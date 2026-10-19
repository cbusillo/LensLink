import Foundation
import AVFoundation
import Compression
import Metal
import Vision

/// Virtual green screen: keeps the person (Vision segmentation, optionally
/// gated by a depth-based max-subject-distance cutoff) and paints
/// everything else chroma green in one Metal compute pass, so the wire
/// carries ordinary video and OBS chroma-keys it exactly like a physical
/// green screen.
///
/// Alpha-matte mode (this fork): when the plugin asks for it, the frame
/// goes to the encoder untouched and a small matte travels beside it
/// as a MATTE packet (docs/PROTOCOL.md §13). The matte keeps the person
/// mask inside the depth cutoff AND anything nearer than the cutoff
/// (the chair, the desk), so the plugin can cut the picture out with
/// real alpha instead of keying green.
///
/// App target ONLY — deliberately absent from `LensLinkBroadcast.sources`
/// (ios-app/project.yml is an explicit list), so the broadcast extension
/// never links Vision or Metal.
///
/// THREADING: `composite(sampleBuffer:)`, `updateDepth(_:)` and
/// `maxDistance` are all called from CameraManager's capture queue
/// ("obscam.video") only — no locking needed. Segmentation runs
/// synchronously on that queue by design: when a frame takes too long the
/// capture output sheds the next one (alwaysDiscardsLateVideoFrames)
/// instead of building a queue — docs/PERFORMANCE.md's drop-don't-queue
/// rule.
final class FrameCompositor {

    /// Person segmentation exists on every OS this app runs on
    /// (VNGeneratePersonSegmentationRequest is iOS 15.0+, the deployment
    /// floor). Kept as an explicit gate for symmetry with
    /// `hdrAvailable` / `appleLogCaptureAvailable`, so a future
    /// capability split or floor change has one place to land.
    static var supportsSegmentation: Bool { true }

    /// Max subject distance in meters: anything farther is treated as
    /// background even where the person mask disagrees (iOS 15's
    /// segmentation is ONE mask covering all people, so this is the only
    /// tool against a passer-by behind the subject). 0 disables the
    /// cutoff; a negative value is Auto, which sets the cutoff behind
    /// the person from the depth map itself. Capture queue only.
    var maxDistance: Float = 0

    /// Auto's cutoff, reported whenever it moves by a tenth of a metre so
    /// the Live screen can show where it landed. Capture queue.
    var onAutoCutoff: ((Float) -> Void)?

    /// Alpha-matte mode switch, read once per frame on the capture
    /// queue (the plugin turns it on per connection). Set before the
    /// compositor is wired to the camera.
    var matteRequested: () -> Bool = { false }

    /// Receives each MATTE payload with its frame's wire pts. Capture
    /// queue; the receiver hops to the network queue itself.
    var onMatte: ((Data, UInt64) -> Void)?

    /// The matte is no wider than this: its inputs (the person mask,
    /// ~256-512 wide, and the depth map, ~320 wide) carry no more
    /// detail, and the plugin upsamples it to the frame.
    static let matteMaxWidth = 960

    // MARK: - Members (all capture-queue confined after init)

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let mattePipeline: MTLComputePipelineState
    private let guidedCoeffsPipeline: MTLComputePipelineState
    private let guidedApplyPipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    /// Metal validation requires every declared texture argument bound
    /// even when the kernel's runtime branch never samples it; this 1×1
    /// NaN texel stands in when no depth map exists (NaN reads as "no
    /// depth opinion" → gate 1 in the kernel, so even a stray sample is
    /// harmless).
    private let dummyDepthTexture: MTLTexture

    /// ONE stateful request for the stream's life: Vision smooths
    /// temporal changes between frames only when the same request
    /// instance sees the sequence in order (it is a VNStatefulRequest).
    private let request: VNGeneratePersonSegmentationRequest
    private let sequenceHandler = VNSequenceRequestHandler()

    /// App-owned output pool — the capture buffer must never be written
    /// in place (the hardware encoder still reads the previous frame
    /// asynchronously), so the composite pass IS the one copy.
    private var pool: CVPixelBufferPool?
    private var poolWidth = 0
    private var poolHeight = 0
    /// Buffers from one pool share a format description; recreated only
    /// when CMVideoFormatDescriptionMatchesImageBuffer says otherwise.
    private var cachedFormatDescription: CMVideoFormatDescription?

    /// Depth runs slower than video (~15 Hz vs 30/60), so the newest map
    /// is kept and reused until the next one lands.
    private var latestDepthPixelBuffer: CVPixelBuffer?
    /// Auto's working cutoff (0 until a person has been measured), and
    /// the depth map it was last measured against: one measurement per
    /// depth frame, not per video frame.
    private var autoCutoff: Float = 0
    /// The person's eased median distance (0 until measured): the
    /// matte keeps near things only from a little in front of it back.
    private var personDepth: Float = 0
    /// How far in front of the person the matte's near band starts, so
    /// a chair behind stays and a desk in front goes.
    private static let nearBandFront: Float = 0.3
    /// Guided-filter window radius (matte pixels) and regularization:
    /// the coarse matte snaps to edges in the picture within this reach.
    private static let guidedRadius: Float = 6
    private static let guidedEps: Float = 0.001
    private var matteRaw: MTLTexture?
    private var matteCoeffs: MTLTexture?
    private var autoMeasuredDepth: CVPixelBuffer?
    private var reportedAutoCutoff: Float = 0

    /// The matte target: one buffer, reused — the pass waits for the
    /// GPU and compresses before returning, so nothing else holds it.
    private var matteBuffer: CVPixelBuffer?
    /// Contiguous copy of the matte rows (the buffer's rows may be
    /// padded) and the deflate output, both sized for the matte.
    private var matteRows = Data()
    private var matteCompressed = Data()
    private let compressionScratch = UnsafeMutableRawPointer.allocate(
        byteCount: max(1, compression_encode_scratch_buffer_size(COMPRESSION_ZLIB)),
        alignment: 16)

    private var loggedStages = Set<String>()
    private var loggedMaskDims = false

    // MARK: - Init

    /// - Parameter targetFps: the stream's configured frame rate. It
    ///   picks the segmentation model once — the request is stateful and
    ///   lives for the whole stream: `.fast` is Apple's "streaming"
    ///   model (the only safe choice above 30 fps), `.balanced` the
    ///   30 fps video model (WWDC21 session 10040).
    /// - Returns: nil when Metal is unavailable or the kernel fails to
    ///   compile — the caller then streams uncomposited frames.
    /// The Metal objects are process-wide constants (the shader source
    /// never varies — only the Vision request depends on init
    /// parameters), so they compile ONCE per process: a fresh compositor
    /// is created on every (re)configure while armed, and recompiling
    /// tens of milliseconds of MSL on the main thread per lens switch
    /// was a per-reconfigure UI hitch for nothing.
    private struct SharedMetal {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let pipeline: MTLComputePipelineState
        let mattePipeline: MTLComputePipelineState
        let guidedCoeffsPipeline: MTLComputePipelineState
        let guidedApplyPipeline: MTLComputePipelineState
    }

    private static let sharedMetal: SharedMetal? = {
        guard let metalDevice = MTLCreateSystemDefaultDevice(),
              let queue = metalDevice.makeCommandQueue() else {
            print("FrameCompositor: Metal unavailable — green screen disabled")
            return nil
        }
        // The kernel ships as Swift-embedded MSL compiled here once per
        // process via makeLibrary(source:) — keeping the shader next to
        // the code that dispatches it and avoiding .metal/metallib build
        // plumbing in project.yml.
        guard let library = try? metalDevice.makeLibrary(
                source: shaderSource, options: nil),
              let function = library.makeFunction(name: "greenScreenComposite"),
              let pipeline = try? metalDevice.makeComputePipelineState(
                function: function),
              let matteFunction = library.makeFunction(name: "matteComposite"),
              let mattePipeline = try? metalDevice.makeComputePipelineState(
                function: matteFunction),
              let coeffsFunction = library.makeFunction(name: "guidedCoeffs"),
              let guidedCoeffsPipeline = try? metalDevice.makeComputePipelineState(
                function: coeffsFunction),
              let applyFunction = library.makeFunction(name: "guidedApply"),
              let guidedApplyPipeline = try? metalDevice.makeComputePipelineState(
                function: applyFunction) else {
            print("FrameCompositor: shader compile failed — green screen disabled")
            return nil
        }
        return SharedMetal(device: metalDevice, queue: queue,
                           pipeline: pipeline, mattePipeline: mattePipeline,
                           guidedCoeffsPipeline: guidedCoeffsPipeline,
                           guidedApplyPipeline: guidedApplyPipeline)
    }()

    init?(targetFps: Int32) {
        guard let shared = Self.sharedMetal else { return nil }
        let metalDevice = shared.device
        let queue = shared.queue
        let pipeline = shared.pipeline
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice,
                                  nil, &cache)
        guard let cache else {
            print("FrameCompositor: texture cache creation failed — green screen disabled")
            return nil
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: 1, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        guard let dummy = metalDevice.makeTexture(descriptor: descriptor) else {
            print("FrameCompositor: dummy texture creation failed — green screen disabled")
            return nil
        }
        var nan: UInt16 = 0x7E00 // IEEE 754 half-precision NaN
        dummy.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                      withBytes: &nan,
                      bytesPerRow: MemoryLayout<UInt16>.size)

        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .fast
        // OneComponent8 maps straight onto an r8Unorm texture.
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8

        self.device = metalDevice
        self.commandQueue = queue
        self.pipeline = pipeline
        self.mattePipeline = shared.mattePipeline
        self.guidedCoeffsPipeline = shared.guidedCoeffsPipeline
        self.guidedApplyPipeline = shared.guidedApplyPipeline
        self.textureCache = cache
        self.dummyDepthTexture = dummy
        self.request = request
    }

    deinit {
        compressionScratch.deallocate()
    }

    // MARK: - Depth

    /// Store the newest depth frame (capture queue only). Converts to
    /// DepthFloat16 defensively — CameraManager pins that format, so the
    /// conversion is normally a no-op — because the kernel samples the
    /// map as r16Float meters.
    func updateDepth(_ depth: AVDepthData) {
        let wanted = kCVPixelFormatType_DepthFloat16
        let data = depth.depthDataType == wanted
            ? depth : depth.converting(toDepthDataType: wanted)
        latestDepthPixelBuffer = data.depthDataMap
    }

    // MARK: - Composite

    /// Capture queue only. Returns:
    ///  - a NEW sample buffer with the person kept and everything else
    ///    painted chroma green (source pts/duration/attachments carried
    ///    over verbatim, so timesync/lip-sync never notice), or
    ///  - the ORIGINAL buffer when any stage fails — fail-open: a Vision
    ///    or Metal hiccup must never blank the stream, or
    ///  - nil when the output pool is exhausted (every in-flight buffer
    ///    is still referenced downstream): the caller drops the frame,
    ///    per the repo-wide drop-don't-queue rule.
    func composite(sampleBuffer: CMSampleBuffer) -> CMSampleBuffer? {
        let started = CMClockGetTime(CMClockGetHostTimeClock())
        let age = CMTimeGetSeconds(CMTimeSubtract(
            started, CMSampleBufferGetPresentationTimeStamp(sampleBuffer)))
        defer {
            let spent = CMTimeGetSeconds(CMTimeSubtract(
                CMClockGetTime(CMClockGetHostTimeClock()), started))
            noteTiming(age: age, spent: spent)
        }
        // The depth synchronizer queues frames instead of shedding them
        // when this pass runs long (Vision at 4K is close to a 30 fps
        // frame), and every queued frame is delay that never comes back.
        // A frame already this stale is dropped here instead.
        if age > Self.maxFrameAge {
            droppedStale += 1
            return nil
        }
        guard let source = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return sampleBuffer
        }
        // Green screen is SDR-only; the composite green constants are
        // BT.709 video-range and the kernel reads 8-bit biplanar 420v.
        guard CVPixelBufferGetPixelFormatType(source)
                == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else {
            logOnce("pixel-format",
                    "FrameCompositor: non-420v input — green screen is SDR-only; passing frames through")
            return sampleBuffer
        }

        // 1. Person mask. Synchronous on the capture queue: overload
        // sheds the NEXT frame via alwaysDiscardsLateVideoFrames rather
        // than queueing work. Buffers are upright sensor-native
        // landscape (the connection sets .landscapeRight and mirrors the
        // front camera), so .up keeps the mask in buffer coordinates.
        let visionStarted = CMClockGetTime(CMClockGetHostTimeClock())
        do {
            try sequenceHandler.perform([request], on: source,
                                        orientation: .up)
            timingVision += CMTimeGetSeconds(CMTimeSubtract(
                CMClockGetTime(CMClockGetHostTimeClock()), visionStarted))
        } catch {
            logOnce("vision",
                    "FrameCompositor: segmentation failed (\(error.localizedDescription)) — passing frames through")
            return sampleBuffer
        }
        guard let maskBuffer = request.results?.first?.pixelBuffer else {
            logOnce("vision-results",
                    "FrameCompositor: segmentation returned no mask — passing frames through")
            return sampleBuffer
        }

        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        if !loggedMaskDims {
            loggedMaskDims = true
            // Mask resolution per quality level is undocumented — log the
            // observed value once (it feeds the feather-quality picture).
            print("FrameCompositor: mask "
                  + "\(CVPixelBufferGetWidth(maskBuffer))x\(CVPixelBufferGetHeight(maskBuffer))"
                  + " for \(width)x\(height) input")
        }

        if maxDistance != 0, let depth = latestDepthPixelBuffer,
           depth !== autoMeasuredDepth {
            autoMeasuredDepth = depth
            measureAutoCutoff(mask: maskBuffer, depth: depth)
        }

        // Alpha-matte mode: the picture stays untouched and the matte
        // rides beside it. No matte (a failed pass) still sends the
        // frame — the plugin then shows it opaque, never blank.
        if let onMatte, matteRequested() {
            if let payload = makeMatte(mask: maskBuffer, source: source,
                                       width: width, height: height) {
                // VideoEncoder's exact conversion, so the plugin pairs
                // matte and frame by equal pts.
                let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                onMatte(payload,
                        UInt64(max(0, CMTimeGetSeconds(pts)) * 1_000_000_000))
            }
            return sampleBuffer
        }

        // 2. Output buffer from our pool (rebuilt on dimension change).
        if pool == nil || width != poolWidth || height != poolHeight {
            rebuildPool(width: width, height: height)
        }
        guard let pool else { return sampleBuffer } // rebuild already logged
        var allocated: CVPixelBuffer?
        let aux = [kCVPixelBufferPoolAllocationThresholdKey as String: 6]
        let poolStatus = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault, pool, aux as CFDictionary, &allocated)
        if poolStatus == kCVReturnWouldExceedAllocationThreshold {
            // All in-flight buffers still referenced downstream — the
            // GPU/encoder is behind. Drop, don't queue.
            return nil
        }
        guard poolStatus == kCVReturnSuccess, let output = allocated else {
            logOnce("pool-alloc",
                    "FrameCompositor: pool allocation failed (\(poolStatus)) — passing frames through")
            return sampleBuffer
        }

        // 3. GPU composite.
        guard runKernel(source: source, mask: maskBuffer, output: output) else {
            logOnce("metal",
                    "FrameCompositor: Metal composite failed — passing frames through")
            return sampleBuffer
        }

        // 4. Wrap for the encoder. Attachments (colour primaries /
        // transfer / matrix — the BT.709 tags) must travel with the new
        // buffer or colours shift on the wire.
        CVBufferPropagateAttachments(source, output)
        if let cached = cachedFormatDescription,
           !CMVideoFormatDescriptionMatchesImageBuffer(cached,
                                                       imageBuffer: output) {
            cachedFormatDescription = nil
        }
        if cachedFormatDescription == nil {
            var created: CMVideoFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault, imageBuffer: output,
                formatDescriptionOut: &created)
            cachedFormatDescription = created
        }
        guard let formatDescription = cachedFormatDescription else {
            logOnce("format-description",
                    "FrameCompositor: format description creation failed — passing frames through")
            return sampleBuffer
        }
        // Source timing verbatim: pts drives the wire timestamps.
        var timing = CMSampleTimingInfo(
            duration: CMSampleBufferGetDuration(sampleBuffer),
            presentationTimeStamp:
                CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            decodeTimeStamp: .invalid)
        var wrapped: CMSampleBuffer?
        let status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: output,
            formatDescription: formatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &wrapped)
        guard status == noErr, let wrapped else {
            logOnce("sample-buffer",
                    "FrameCompositor: sample buffer wrap failed (\(status)) — passing frames through")
            return sampleBuffer
        }
        return wrapped
    }

    // MARK: - Metal pass

    private func runKernel(source: CVPixelBuffer, mask: CVPixelBuffer,
                           output: CVPixelBuffer) -> Bool {
        // The CVMetalTextures must outlive the GPU pass (releasing one
        // recycles its backing); collected here and held past the wait.
        var retained: [CVMetalTexture] = []
        guard let srcY = makeTexture(from: source, pixelFormat: .r8Unorm,
                                     planeIndex: 0, retaining: &retained),
              let srcCbCr = makeTexture(from: source, pixelFormat: .rg8Unorm,
                                        planeIndex: 1, retaining: &retained),
              let maskTexture = makeTexture(from: mask, pixelFormat: .r8Unorm,
                                            planeIndex: 0, retaining: &retained),
              let outY = makeTexture(from: output, pixelFormat: .r8Unorm,
                                     planeIndex: 0, retaining: &retained),
              let outCbCr = makeTexture(from: output, pixelFormat: .rg8Unorm,
                                        planeIndex: 1, retaining: &retained)
        else { return false }

        let (depthTexture, depthParams) = depthInputs(retaining: &retained)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(srcY, index: 0)
        encoder.setTexture(srcCbCr, index: 1)
        encoder.setTexture(maskTexture, index: 2)
        encoder.setTexture(depthTexture, index: 3)
        encoder.setTexture(outY, index: 4)
        encoder.setTexture(outCbCr, index: 5)
        var params = depthParams
        encoder.setBytes(&params, length: MemoryLayout<SIMD2<Float>>.size,
                         index: 0)

        // One thread per 2×2 luma quad. Plain dispatchThreadgroups with
        // an in-kernel bounds check — non-uniform threadgroups need
        // A11+, and iOS 15 still runs on A9/A10.
        let chromaWidth = CVPixelBufferGetWidthOfPlane(output, 1)
        let chromaHeight = CVPixelBufferGetHeightOfPlane(output, 1)
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (chromaWidth + 15) / 16,
                             height: (chromaHeight + 15) / 16,
                             depth: 1)
        encoder.dispatchThreadgroups(groups,
                                     threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        // Blocking (~0.5 ms) is the simplest correct ordering before the
        // encoder reads the buffer; overload sheds upstream via
        // alwaysDiscardsLateVideoFrames. An MTLSharedEvent pipeline is a
        // later optimization only if measurements demand it.
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(retained) {}
        return commandBuffer.error == nil
    }

    /// The depth texture and kernel parameters (cutoff, hasDepth) for
    /// this frame. Depth is optional per frame: no map yet, no cutoff
    /// configured, or a failed wrap all degrade to segmentation-only.
    private func depthInputs(retaining retained: inout [CVMetalTexture])
        -> (MTLTexture, SIMD2<Float>) {
        let cutoff = maxDistance < 0 ? autoCutoff : maxDistance
        guard cutoff > 0, let depthBuffer = latestDepthPixelBuffer else {
            return (dummyDepthTexture, SIMD2<Float>(cutoff, 0))
        }
        guard let wrappedDepth = makeTexture(from: depthBuffer,
                                             pixelFormat: .r16Float,
                                             planeIndex: 0,
                                             retaining: &retained) else {
            logOnce("depth-wrap",
                    "FrameCompositor: depth texture wrap failed — segmentation-only")
            return (dummyDepthTexture, SIMD2<Float>(cutoff, 0))
        }
        return (wrappedDepth, SIMD2<Float>(cutoff, 1))
    }

    // MARK: - Matte pass

    /// Renders the matte (frame size divided down to at most
    /// `matteMaxWidth` wide) and packs it as a MATTE payload:
    /// width u16, height u16, encoding u8 (0 raw, 1 raw deflate),
    /// reserved u8, then width×height bytes of alpha, row-major,
    /// 255 = keep. Three passes: the coarse matte (with the frame's luma
    /// beside it), then a guided filter that snaps it to edges in the
    /// picture. nil when any stage fails (logged once).
    private func makeMatte(mask: CVPixelBuffer, source: CVPixelBuffer,
                           width: Int, height: Int) -> Data? {
        let scale = max(1, (width + Self.matteMaxWidth - 1) / Self.matteMaxWidth)
        let matteWidth = (width + scale - 1) / scale
        let matteHeight = (height + scale - 1) / scale
        if let existing = matteBuffer,
           CVPixelBufferGetWidth(existing) != matteWidth
            || CVPixelBufferGetHeight(existing) != matteHeight {
            matteBuffer = nil
        }
        if matteBuffer == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, matteWidth, matteHeight,
                                kCVPixelFormatType_OneComponent8,
                                attributes as CFDictionary, &created)
            matteBuffer = created
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rg16Float, width: matteWidth,
                height: matteHeight, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            matteRaw = device.makeTexture(descriptor: descriptor)
            descriptor.pixelFormat = .rg32Float
            matteCoeffs = device.makeTexture(descriptor: descriptor)
            matteRows = Data(count: matteWidth * matteHeight)
            matteCompressed = Data(count: matteWidth * matteHeight)
        }
        guard let output = matteBuffer, let raw = matteRaw,
              let coeffs = matteCoeffs else {
            logOnce("matte-buffer", "FrameCompositor: matte buffer creation failed")
            return nil
        }

        var retained: [CVMetalTexture] = []
        guard let maskTexture = makeTexture(from: mask, pixelFormat: .r8Unorm,
                                            planeIndex: 0, retaining: &retained),
              let lumaTexture = makeTexture(from: source, pixelFormat: .r8Unorm,
                                            planeIndex: 0, retaining: &retained),
              let outTexture = makeTexture(from: output, pixelFormat: .r8Unorm,
                                           planeIndex: 0, retaining: &retained),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            logOnce("matte-metal", "FrameCompositor: matte pass setup failed")
            return nil
        }
        let (depthTexture, depthParams) = depthInputs(retaining: &retained)
        let nearMin = personDepth > 0 ? max(0, personDepth - Self.nearBandFront) : 0
        var params = SIMD4<Float>(depthParams.x, depthParams.y, nearMin, 0)
        var guided = SIMD2<Float>(Self.guidedRadius, Self.guidedEps)
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (matteWidth + 15) / 16,
                             height: (matteHeight + 15) / 16, depth: 1)

        encoder.setComputePipelineState(mattePipeline)
        encoder.setTexture(maskTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setTexture(lumaTexture, index: 2)
        encoder.setTexture(raw, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.size,
                         index: 0)
        encoder.dispatchThreadgroups(groups,
                                     threadsPerThreadgroup: threadsPerGroup)

        encoder.setComputePipelineState(guidedCoeffsPipeline)
        encoder.setTexture(raw, index: 0)
        encoder.setTexture(coeffs, index: 1)
        encoder.setBytes(&guided, length: MemoryLayout<SIMD2<Float>>.size,
                         index: 0)
        encoder.dispatchThreadgroups(groups,
                                     threadsPerThreadgroup: threadsPerGroup)

        encoder.setComputePipelineState(guidedApplyPipeline)
        encoder.setTexture(raw, index: 0)
        encoder.setTexture(coeffs, index: 1)
        encoder.setTexture(outTexture, index: 2)
        encoder.setBytes(&guided, length: MemoryLayout<SIMD2<Float>>.size,
                         index: 0)
        encoder.dispatchThreadgroups(groups,
                                     threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(retained) {}
        guard commandBuffer.error == nil else {
            logOnce("matte-gpu", "FrameCompositor: matte pass failed")
            return nil
        }

        CVPixelBufferLockBaseAddress(output, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(output) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(output)
        let size = matteWidth * matteHeight
        let compressedSize = matteRows.withUnsafeMutableBytes { rows -> Int in
            let rowsBase = rows.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for row in 0..<matteHeight {
                memcpy(rowsBase + row * matteWidth, base + row * rowBytes,
                       matteWidth)
            }
            return matteCompressed.withUnsafeMutableBytes { packed in
                compression_encode_buffer(
                    packed.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    size, rowsBase, size, compressionScratch, COMPRESSION_ZLIB)
            }
        }
        // 0 means deflate could not beat raw within the buffer: send raw.
        let deflated = compressedSize > 0
        timingMatteBytes += deflated ? compressedSize : size
        var payload = Data(capacity: 6 + (deflated ? compressedSize : size))
        payload.append(UInt8(matteWidth >> 8))
        payload.append(UInt8(matteWidth & 0xFF))
        payload.append(UInt8(matteHeight >> 8))
        payload.append(UInt8(matteHeight & 0xFF))
        payload.append(deflated ? 1 : 0)
        payload.append(0)
        payload.append(deflated ? matteCompressed.prefix(compressedSize)
                                : matteRows.prefix(size))
        return payload
    }

    private func makeTexture(from buffer: CVPixelBuffer,
                             pixelFormat: MTLPixelFormat,
                             planeIndex: Int,
                             retaining retained: inout [CVMetalTexture])
        -> MTLTexture? {
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar
            ? CVPixelBufferGetWidthOfPlane(buffer, planeIndex)
            : CVPixelBufferGetWidth(buffer)
        let height = planar
            ? CVPixelBufferGetHeightOfPlane(buffer, planeIndex)
            : CVPixelBufferGetHeight(buffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, nil, pixelFormat,
            width, height, planeIndex, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else {
            return nil
        }
        retained.append(cvTexture)
        return texture
    }

    // MARK: - Auto cutoff

    /// Metres kept behind the person: their own depth plus a margin.
    private static let autoMargin: Float = 0.6

    /// Samples the person's distance on a coarse grid (where the mask
    /// says person and the depth map has a reading), takes the median,
    /// and eases the cutoff toward that plus the margin, so a hand
    /// thrown forward or a missed frame doesn't jerk the matte. With no
    /// one in frame the last cutoff holds.
    private func measureAutoCutoff(mask: CVPixelBuffer, depth: CVPixelBuffer) {
        guard CVPixelBufferGetPixelFormatType(depth)
                == kCVPixelFormatType_DepthFloat16 else { return }
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depth, .readOnly)
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }
        guard let maskBase = CVPixelBufferGetBaseAddress(mask),
              let depthBase = CVPixelBufferGetBaseAddress(depth) else { return }
        let maskW = CVPixelBufferGetWidth(mask)
        let maskH = CVPixelBufferGetHeight(mask)
        let maskRow = CVPixelBufferGetBytesPerRow(mask)
        let depthW = CVPixelBufferGetWidth(depth)
        let depthH = CVPixelBufferGetHeight(depth)
        let depthRow = CVPixelBufferGetBytesPerRow(depth)

        let columns = 32, rows = 18
        var samples: [Float] = []
        samples.reserveCapacity(columns * rows)
        for gy in 0..<rows {
            let v = (Float(gy) + 0.5) / Float(rows)
            for gx in 0..<columns {
                let u = (Float(gx) + 0.5) / Float(columns)
                let mx = min(Int(u * Float(maskW)), maskW - 1)
                let my = min(Int(v * Float(maskH)), maskH - 1)
                guard maskBase.load(fromByteOffset: my * maskRow + mx,
                                    as: UInt8.self) > 127 else { continue }
                let dx = min(Int(u * Float(depthW)), depthW - 1)
                let dy = min(Int(v * Float(depthH)), depthH - 1)
                let bits = depthBase.load(fromByteOffset: dy * depthRow + dx * 2,
                                          as: UInt16.self)
                let d = Float(Float16(bitPattern: bits))
                if d.isFinite, d > 0.1 { samples.append(d) }
            }
        }
        guard samples.count >= 8 else { return }
        samples.sort()
        let median = samples[samples.count / 2]
        personDepth = personDepth == 0 ? median
            : personDepth + (median - personDepth) * 0.25
        let target = min(median + Self.autoMargin, 5)
        autoCutoff = autoCutoff == 0 ? target : autoCutoff + (target - autoCutoff) * 0.25
        let rounded = (autoCutoff * 10).rounded() / 10
        if rounded != reportedAutoCutoff {
            reportedAutoCutoff = rounded
            onAutoCutoff?(rounded)
        }
    }

    // MARK: - Pool

    private func rebuildPool(width: Int, height: Int) {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            // IOSurface backing keeps the VideoToolbox handoff zero-copy
            // and lets the texture cache wrap the planes for writing.
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var created: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, nil,
                                             attributes as CFDictionary,
                                             &created)
        pool = status == kCVReturnSuccess ? created : nil
        poolWidth = width
        poolHeight = height
        cachedFormatDescription = nil
        if pool == nil {
            logOnce("pool-create",
                    "FrameCompositor: pool creation failed (\(status)) — passing frames through")
        }
    }

    // MARK: - Timing (diagnostic)

    /// Frames older than this on arrival are dropped (two 30 fps frames).
    private static let maxFrameAge = 0.08
    private var droppedStale = 0
    private var timingVision = 0.0
    private var timingMatteBytes = 0
    private var timingFrames = 0
    private var timingAge = 0.0
    private var timingSpent = 0.0
    private var timingSpentMax = 0.0

    /// Every 150 frames: how old frames are when they reach the
    /// compositor (capture → callback) and how long it holds them.
    private func noteTiming(age: Double, spent: Double) {
        timingFrames += 1
        timingAge += age
        timingSpent += spent
        timingSpentMax = max(timingSpentMax, spent)
        guard timingFrames == 150 else { return }
        print(String(format: "FrameCompositor timing: age %.0f ms, spent %.1f ms (max %.1f), vision %.1f ms, dropped stale %d, matte %d KB/frame",
                     timingAge / 150 * 1000, timingSpent / 150 * 1000,
                     timingSpentMax * 1000, timingVision / 150 * 1000,
                     droppedStale, timingMatteBytes / 150 / 1024))
        timingMatteBytes = 0
        timingVision = 0
        droppedStale = 0
        timingFrames = 0
        timingAge = 0
        timingSpent = 0
        timingSpentMax = 0
    }

    // MARK: - Logging

    /// Capture queue only. Each distinct failure stage logs once — a
    /// per-frame print at 60 fps would itself be a performance bug.
    private func logOnce(_ stage: String, _ message: String) {
        guard loggedStages.insert(stage).inserted else { return }
        print(message)
    }

    // MARK: - Kernel source

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Chroma green, BT.709 limited (video) range — THE single tunable
    // set of green constants for the whole feature.
    // sRGB #00B140 (0, 177, 64) is the industry chroma-key green; it
    // survives 4:2:0 subsampling + H.264/HEVC quantization better than
    // primary green (0, 255, 0). Derivation with BT.709 coefficients
    // (Kr = 0.2126, Kg = 0.7152, Kb = 0.0722):
    //   Y' = 0.5146, Cb' = -0.1420, Cr' = -0.3267
    //   Y  = 16 + 219 * Y'  = 129
    //   Cb = 128 + 224 * Cb' = 96
    //   Cr = 128 + 224 * Cr' = 55
    // Normalized for r8Unorm / rg8Unorm writes:
    constant float GREEN_Y  = 129.0 / 255.0;
    constant float GREEN_CB =  96.0 / 255.0;
    constant float GREEN_CR =  55.0 / 255.0;

    // Depth gate feather, meters: alpha ramps 1 -> 0 over
    // [maxDistance - FEATHER, maxDistance].
    constant float DEPTH_FEATHER_M = 0.15;

    struct Params {
        float maxDistance; // meters; <= 0 disables the depth cutoff
        float hasDepth;    // > 0.5 when the depth texture is real
    };

    // One thread per 2x2 luma quad (grid = chroma-plane dimensions).
    kernel void greenScreenComposite(
        texture2d<float, access::read>   srcY    [[texture(0)]],
        texture2d<float, access::read>   srcCbCr [[texture(1)]],
        texture2d<float, access::sample> mask    [[texture(2)]],
        texture2d<float, access::sample> depth   [[texture(3)]],
        texture2d<float, access::write>  outY    [[texture(4)]],
        texture2d<float, access::write>  outCbCr [[texture(5)]],
        constant Params &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        const uint chromaW = outCbCr.get_width();
        const uint chromaH = outCbCr.get_height();
        if (gid.x >= chromaW || gid.y >= chromaH)
            return;

        constexpr sampler bilinear(coord::normalized,
                                   address::clamp_to_edge,
                                   filter::linear);

        const uint lumaW = outY.get_width();
        const uint lumaH = outY.get_height();
        const float2 lumaSize = float2(lumaW, lumaH);

        float alphaSum = 0.0;
        for (uint i = 0; i < 4; ++i) {
            const uint2 px = uint2(min(2 * gid.x + (i & 1u), lumaW - 1),
                                   min(2 * gid.y + (i >> 1), lumaH - 1));
            // The mask is far smaller than the frame (~256 wide): this
            // bilinear upsample IS the edge feather.
            const float2 uv = (float2(px) + 0.5) / lumaSize;
            float alpha = mask.sample(bilinear, uv).r;

            // Depth gate: 1 inside the cutoff, ramping to 0 at it.
            // NaN (depth holes / no reading) must NEVER classify as
            // background — every comparison against NaN is false, so
            // test isnan() FIRST and fall through to "no depth
            // opinion" (gate stays 1).
            if (params.hasDepth > 0.5 && params.maxDistance > 0.0) {
                const float d = depth.sample(bilinear, uv).r;
                if (!isnan(d)) {
                    alpha *= 1.0 - smoothstep(
                        params.maxDistance - DEPTH_FEATHER_M,
                        params.maxDistance, d);
                }
            }

            const float y = srcY.read(px).r;
            outY.write(float4(mix(GREEN_Y, y, alpha)), px);
            alphaSum += alpha;
        }

        const float alphaMean = alphaSum * 0.25;
        const float2 cbcr = srcCbCr.read(gid).rg;
        const float2 outC = mix(float2(GREEN_CB, GREEN_CR), cbcr, alphaMean);
        outCbCr.write(float4(outC, 0.0, 0.0), gid);
    }

    // Alpha matte, one thread per matte pixel, written as (alpha,
    // luma) for the guided filter below. The person mask is kept inside
    // the depth cutoff (as above). Near things are kept even where the
    // mask says no — the chair — but only within a band from a little
    // in front of the person (nearMin) back to the cutoff, so a desk in
    // front goes. That test uses the farthest and nearest of the 3x3
    // depth texels around the pixel, so the blurred depth edge never
    // pulls the background in; a hole anywhere in the window gives it
    // no opinion (the mask alone decides).
    struct MatteParams {
        float maxDistance;
        float hasDepth;
        float nearMin; // metres; 0 = no lower bound
        float unused;
    };

    kernel void matteComposite(
        texture2d<float, access::sample> mask  [[texture(0)]],
        texture2d<float, access::sample> depth [[texture(1)]],
        texture2d<float, access::sample> luma  [[texture(2)]],
        texture2d<float, access::write>  outRaw [[texture(3)]],
        constant MatteParams &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        const uint w = outRaw.get_width();
        const uint h = outRaw.get_height();
        if (gid.x >= w || gid.y >= h)
            return;

        constexpr sampler bilinear(coord::normalized,
                                   address::clamp_to_edge,
                                   filter::linear);
        constexpr sampler nearest(coord::normalized,
                                  address::clamp_to_edge,
                                  filter::nearest);

        const float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float alpha = mask.sample(bilinear, uv).r;

        if (params.hasDepth > 0.5 && params.maxDistance > 0.0) {
            const float cut = params.maxDistance;
            const float d = depth.sample(bilinear, uv).r;
            if (!isnan(d))
                alpha *= 1.0 - smoothstep(cut - DEPTH_FEATHER_M, cut, d);

            const float2 texel =
                1.0 / float2(depth.get_width(), depth.get_height());
            float farthest = 0.0;
            float closest = 1e6;
            bool hole = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const float s = depth.sample(
                        nearest, uv + float2(dx, dy) * texel).r;
                    if (isnan(s) || s <= 0.0) {
                        hole = true;
                    } else {
                        farthest = max(farthest, s);
                        closest = min(closest, s);
                    }
                }
            }
            if (!hole) {
                float keepNear = 1.0 - smoothstep(
                    cut - DEPTH_FEATHER_M, cut, farthest);
                if (params.nearMin > 0.0)
                    keepNear *= smoothstep(params.nearMin - DEPTH_FEATHER_M,
                                       params.nearMin, closest);
                alpha = max(alpha, keepNear);
            }
        }

        outRaw.write(float4(alpha, luma.sample(bilinear, uv).r, 0.0, 0.0),
                     gid);
    }

    // Guided filter (He et al.), luma as the guide: per window the
    // matte is fit as a*I + b, so its edges follow edges in the picture
    // instead of the coarse mask and depth grids.
    struct GuidedParams {
        float radius;
        float eps;
    };

    kernel void guidedCoeffs(
        texture2d<float, access::read>  raw    [[texture(0)]],
        texture2d<float, access::write> coeffs [[texture(1)]],
        constant GuidedParams &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        const int w = int(raw.get_width());
        const int h = int(raw.get_height());
        if (int(gid.x) >= w || int(gid.y) >= h)
            return;
        const int r = int(params.radius);
        float sumI = 0.0, sumP = 0.0, sumIP = 0.0, sumII = 0.0, n = 0.0;
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                const uint2 q = uint2(clamp(int(gid.x) + dx, 0, w - 1),
                                      clamp(int(gid.y) + dy, 0, h - 1));
                const float2 v = raw.read(q).rg;
                sumP += v.x;
                sumI += v.y;
                sumIP += v.x * v.y;
                sumII += v.y * v.y;
                n += 1.0;
            }
        }
        const float meanI = sumI / n;
        const float meanP = sumP / n;
        const float cov = sumIP / n - meanI * meanP;
        const float variance = sumII / n - meanI * meanI;
        const float a = cov / (variance + params.eps);
        coeffs.write(float4(a, meanP - a * meanI, 0.0, 0.0), gid);
    }

    kernel void guidedApply(
        texture2d<float, access::read>  raw    [[texture(0)]],
        texture2d<float, access::read>  coeffs [[texture(1)]],
        texture2d<float, access::write> outA   [[texture(2)]],
        constant GuidedParams &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        const int w = int(raw.get_width());
        const int h = int(raw.get_height());
        if (int(gid.x) >= w || int(gid.y) >= h)
            return;
        const int r = int(params.radius);
        float2 sum = float2(0.0);
        float n = 0.0;
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                const uint2 q = uint2(clamp(int(gid.x) + dx, 0, w - 1),
                                      clamp(int(gid.y) + dy, 0, h - 1));
                sum += coeffs.read(q).rg;
                n += 1.0;
            }
        }
        const float I = raw.read(gid).g;
        // Snap the filter's faint tails to exactly 0 and 1: only the edge
        // band keeps gradients, so the matte still deflates to a few KB.
        const float q = (sum.x / n) * I + sum.y / n;
        outA.write(float4(smoothstep(0.03, 0.97, q)), gid);
    }
    """
}
