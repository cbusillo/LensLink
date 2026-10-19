import Foundation
import AVFoundation

/// Offline test-set recorder (this fork): saves what the green screen sees
/// so cutout ideas can be tried on the Mac without rebuilding the app.
/// Each take is a folder under Documents/datasets/ holding
///  - video.mov: the camera frames as HEVC, original timestamps;
///  - depth.bin: every depth map as it arrived — per record: pts (u64
///    nanoseconds, the wire clock), width u32, height u32, bytesPerRow
///    u32, then height×bytesPerRow bytes of DepthFloat16 metres; all
///    little-endian;
///  - calibration.json: intrinsics and their reference size, once.
///
/// Started and stopped by the `record_dataset` CONTROL command. Appends
/// come from the capture queue and hop to the recorder's own queue, so
/// file I/O never runs on the capture path; a writer that is not ready
/// drops the frame rather than queueing it.
final class DatasetRecorder {

    private let queue = DispatchQueue(label: "lenslink.dataset")
    private let lock = NSLock()
    private var active = false

    private var folder: URL?
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var depthHandle: FileHandle?
    private var wroteCalibration = false
    private var stopAt: CFAbsoluteTime = 0
    private var frames = 0
    private var depthMaps = 0

    /// True while a take is running. Any queue.
    var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    /// Starts a take of `seconds` (it also ends early on `stop()`).
    func start(seconds: Double, label: String) {
        queue.async { [self] in
            guard !isRecording else { return }
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            let safe = label.filter { $0.isLetter || $0.isNumber || $0 == "-" }
            let name = safe.isEmpty ? stamp : "\(stamp)-\(safe)"
            guard let documents = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask).first else { return }
            let dir = documents.appendingPathComponent("datasets/\(name)")
            do {
                try FileManager.default.createDirectory(
                    at: dir, withIntermediateDirectories: true)
                FileManager.default.createFile(
                    atPath: dir.appendingPathComponent("depth.bin").path,
                    contents: nil)
                depthHandle = try FileHandle(
                    forWritingTo: dir.appendingPathComponent("depth.bin"))
                writer = try AVAssetWriter(
                    outputURL: dir.appendingPathComponent("video.mov"),
                    fileType: .mov)
            } catch {
                print("DatasetRecorder: cannot start (\(error.localizedDescription))")
                return
            }
            folder = dir
            input = nil
            wroteCalibration = false
            frames = 0
            depthMaps = 0
            stopAt = CFAbsoluteTimeGetCurrent() + max(1, min(seconds, 600))
            setActive(true)
            print("DatasetRecorder: recording \(dir.lastPathComponent)")
        }
    }

    func stop() {
        queue.async { [self] in finish() }
    }

    /// Capture queue.
    func append(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        queue.async { [self] in
            guard isRecording, let writer else { return }
            if CFAbsoluteTimeGetCurrent() >= stopAt {
                finish()
                return
            }
            if input == nil {
                guard let format = CMSampleBufferGetFormatDescription(sampleBuffer)
                else { return }
                let size = CMVideoFormatDescriptionGetDimensions(format)
                let settings: [String: Any] = [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: Int(size.width),
                    AVVideoHeightKey: Int(size.height),
                    AVVideoCompressionPropertiesKey: [
                        // High enough that edges and hair survive for
                        // offline matting; a minute is about 600 MB.
                        AVVideoAverageBitRateKey: 80_000_000,
                    ],
                ]
                let made = AVAssetWriterInput(mediaType: .video,
                                              outputSettings: settings)
                made.expectsMediaDataInRealTime = true
                guard writer.canAdd(made) else {
                    print("DatasetRecorder: cannot add video input")
                    finish()
                    return
                }
                writer.add(made)
                input = made
                writer.startWriting()
                writer.startSession(atSourceTime:
                    CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
            }
            guard let input, input.isReadyForMoreMediaData else { return }
            if input.append(sampleBuffer) { frames += 1 }
        }
    }

    /// Capture queue. `ptsNanoseconds` is on the same clock as video.
    func append(depth: AVDepthData, ptsNanoseconds: UInt64) {
        guard isRecording else { return }
        let wanted = kCVPixelFormatType_DepthFloat16
        let data = depth.depthDataType == wanted
            ? depth : depth.converting(toDepthDataType: wanted)
        let map = data.depthDataMap
        let calibration = data.cameraCalibrationData
        queue.async { [self] in
            guard isRecording, let depthHandle else { return }
            if !wroteCalibration, let calibration, let folder {
                wroteCalibration = true
                writeCalibration(calibration, to: folder)
            }
            CVPixelBufferLockBaseAddress(map, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(map) else { return }
            let width = UInt32(CVPixelBufferGetWidth(map))
            let height = UInt32(CVPixelBufferGetHeight(map))
            let rowBytes = UInt32(CVPixelBufferGetBytesPerRow(map))
            var record = Data(capacity: 20 + Int(height * rowBytes))
            withUnsafeBytes(of: ptsNanoseconds.littleEndian) { record.append(contentsOf: $0) }
            withUnsafeBytes(of: width.littleEndian) { record.append(contentsOf: $0) }
            withUnsafeBytes(of: height.littleEndian) { record.append(contentsOf: $0) }
            withUnsafeBytes(of: rowBytes.littleEndian) { record.append(contentsOf: $0) }
            record.append(base.assumingMemoryBound(to: UInt8.self),
                          count: Int(height * rowBytes))
            depthHandle.write(record)
            depthMaps += 1
        }
    }

    // MARK: - Private (recorder queue)

    private func setActive(_ value: Bool) {
        lock.lock()
        active = value
        lock.unlock()
    }

    private func finish() {
        guard isRecording else { return }
        setActive(false)
        try? depthHandle?.close()
        depthHandle = nil
        let name = folder?.lastPathComponent ?? "?"
        let counts = "\(frames) frames, \(depthMaps) depth maps"
        if let writer, writer.status == .writing {
            input?.markAsFinished()
            writer.finishWriting {
                print("DatasetRecorder: saved \(name): \(counts)")
            }
        } else {
            print("DatasetRecorder: ended \(name) without video: \(counts)")
        }
        writer = nil
        input = nil
        folder = nil
    }

    private func writeCalibration(_ calibration: AVCameraCalibrationData,
                                  to folder: URL) {
        let k = calibration.intrinsicMatrix
        let reference = calibration.intrinsicMatrixReferenceDimensions
        let json: [String: Any] = [
            "fx": k.columns.0.x, "fy": k.columns.1.y,
            "cx": k.columns.2.x, "cy": k.columns.2.y,
            "referenceWidth": reference.width,
            "referenceHeight": reference.height,
            "pixelSize": calibration.pixelSize,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: json,
                                                  options: [.prettyPrinted]) {
            try? data.write(to: folder.appendingPathComponent("calibration.json"))
        }
    }
}
