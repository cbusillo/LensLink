// Replays a take's video through Apple Vision person segmentation.
//
//   swiftc -O vision-masks.swift -o vision-masks
//   ./vision-masks TAKE_DIR fast|balanced|accurate [MAX_FRAMES]
//
// Writes TAKE_DIR/vision-<quality>.bin: a header of width u32, height u32
// and frame count u32 (little-endian, count patched at the end), then one
// width×height 8-bit mask per frame in file order, and prints the mean and
// worst time per frame. A stateful request with a sequence handler, as in
// the app, so Vision's temporal smoothing matches the live path.

import AVFoundation
import Foundation
import Vision

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: vision-masks TAKE_DIR fast|balanced|accurate [MAX_FRAMES]\n".utf8))
    exit(2)
}
let take = URL(fileURLWithPath: args[1])
let qualityName = args[2]
let maxFrames = args.count > 3 ? Int(args[3]) ?? Int.max : Int.max
let quality: VNGeneratePersonSegmentationRequest.QualityLevel
switch qualityName {
case "fast": quality = .fast
case "balanced": quality = .balanced
case "accurate": quality = .accurate
default:
    FileHandle.standardError.write(Data("unknown quality \(qualityName)\n".utf8))
    exit(2)
}

let asset = AVURLAsset(url: take.appendingPathComponent("video.mov"))
let semaphore = DispatchSemaphore(value: 0)
var track: AVAssetTrack?
Task {
    track = try? await asset.loadTracks(withMediaType: .video).first
    semaphore.signal()
}
semaphore.wait()
guard let track else { fatalError("no video track") }
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
])
reader.add(output)
reader.startReading()

let request = VNGeneratePersonSegmentationRequest()
request.qualityLevel = quality
request.outputPixelFormat = kCVPixelFormatType_OneComponent8
let handler = VNSequenceRequestHandler()

let outURL = take.appendingPathComponent("vision-\(qualityName).bin")
FileManager.default.createFile(atPath: outURL.path, contents: nil)
let out = try FileHandle(forWritingTo: outURL)
var frames: UInt32 = 0
var maskW: UInt32 = 0, maskH: UInt32 = 0
var total = 0.0, worst = 0.0

while frames < maxFrames, let sample = output.copyNextSampleBuffer(),
      let pixels = CMSampleBufferGetImageBuffer(sample) {
    let start = CFAbsoluteTimeGetCurrent()
    try handler.perform([request], on: pixels, orientation: .up)
    let spent = CFAbsoluteTimeGetCurrent() - start
    total += spent
    worst = max(worst, spent)
    guard let mask = request.results?.first?.pixelBuffer else { fatalError("no mask") }
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    let w = CVPixelBufferGetWidth(mask), h = CVPixelBufferGetHeight(mask)
    if frames == 0 {
        maskW = UInt32(w)
        maskH = UInt32(h)
        var header = Data()
        for v in [maskW, maskH, 0] { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        out.write(header)
    }
    let base = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(mask)
    var packed = Data(capacity: w * h)
    for y in 0..<h { packed.append(base + y * row, count: w) }
    CVPixelBufferUnlockBaseAddress(mask, .readOnly)
    out.write(packed)
    frames += 1
}
out.seek(toFileOffset: 8)
withUnsafeBytes(of: frames.littleEndian) { out.write(Data($0)) }
try out.close()
print(String(format: "%@ %@: %u frames, mask %ux%u, mean %.1f ms, worst %.1f ms",
             take.lastPathComponent, qualityName, frames, maskW, maskH,
             total / Double(max(frames, 1)) * 1000, worst * 1000))
