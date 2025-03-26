//
// VideoCapture.swift
// mosaic
//
// Created by Christian Bator on 03/24/2025
//

import AVFoundation
import Accelerate


// MARK: VideoCaptureDimensions

struct VideoCaptureDimensions {
    var width: CInt
    var height: CInt
}

// MARK: - C Interface

@_cdecl("initialize")
public func initialize() -> OpaquePointer {
    let videoCapture = VideoCapture()
    let rawPointer = Unmanaged<VideoCapture>.passUnretained(videoCapture).toOpaque()

    return OpaquePointer(rawPointer)
}

@_cdecl("open")
public func open(pointer: OpaquePointer, dimensions: UnsafeMutableRawPointer) -> CInt {
    let dimensions = dimensions.assumingMemoryBound(to: VideoCaptureDimensions.self)
    
    return videoCapture(from: pointer).open(dimensions: dimensions)
}

@_cdecl("start")
public func start(pointer: OpaquePointer, frameBuffer: UnsafeMutablePointer<UInt8>, isNextFrameAvailable: UnsafeMutablePointer<CInt>) {
    videoCapture(from: pointer).start(frameBuffer: frameBuffer, isNextFrameAvailable: isNextFrameAvailable)
}

@_cdecl("stop")
public func stop(pointer: OpaquePointer) {
    videoCapture(from: pointer).stop()
}

@_cdecl("deinitialize")
public func deinitialize(pointer: OpaquePointer) {
    videoCapture(from: pointer).release()
}

// MARK: Pointer Conversion

private func videoCapture(from pointer: OpaquePointer) -> VideoCapture {
    let rawPointer = UnsafeRawPointer(pointer)
    return Unmanaged<VideoCapture>.fromOpaque(rawPointer).takeUnretainedValue()
}

// MARK: - VideoCapture

class VideoCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private var session: AVCaptureSession?
    private let queue = DispatchQueue(label: "mosaic.video_capture")
    private var destBuffer: vImage_Buffer?
    
    private var frameBuffer: UnsafeMutableRawPointer?
    private var isNextFrameAvailable: UnsafeMutablePointer<CInt>?

    override init() {
        super.init()
        
        _ = Unmanaged<VideoCapture>.passRetained(self)
    }
    
    func open(dimensions: UnsafeMutablePointer<VideoCaptureDimensions>) -> CInt {
        let session = AVCaptureSession()
        
        session.beginConfiguration()
        
        guard let device = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices.first else {
            print("Error initializing capture device: no devices available")
            return 0
        }

        let activeFormat = device.activeFormat
        let activeFormatDimensions = CMVideoFormatDescriptionGetDimensions(activeFormat.formatDescription)

        let height = Int(activeFormatDimensions.height)
        let width = Int(activeFormatDimensions.width)
        
        do {
            let input = try AVCaptureDeviceInput(device: device)

            if session.canAddInput(input) {
                session.addInput(input)
            }
            else {
                print("Error initializing capture device: failed to add input")
                return 0
            }
            
            let output = AVCaptureVideoDataOutput()

            output.videoSettings = [
                (kCVPixelBufferPixelFormatTypeKey as String): Int(kCVPixelFormatType_32ARGB)
            ]

            output.setSampleBufferDelegate(self, queue: queue)

            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            else {
                print("Error initializing capture device: failed to add output")
                return 0
            }
            
            session.commitConfiguration()

            self.session = session
            
            destBuffer = vImage_Buffer(
                data: malloc(height * width * 3),
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: width * 3
            )
            
            dimensions.pointee.width = CInt(width)
            dimensions.pointee.height = CInt(height)
            
            return 1
        }
        catch {
            print("Error initializing capture device: \(error)")
            return 0
        }
    }
    
    func start(frameBuffer: UnsafeMutablePointer<UInt8>, isNextFrameAvailable: UnsafeMutablePointer<CInt>) {
        self.frameBuffer = UnsafeMutableRawPointer(frameBuffer)
        self.isNextFrameAvailable = isNextFrameAvailable
        session?.startRunning()
    }
    
    func stop() {
        session?.stopRunning()
    }

    func release() {
        stop()
        session = nil
        
        if let destBufferData = destBuffer?.data {
            free(destBufferData)
        }
        
        Unmanaged<VideoCapture>.passUnretained(self).release()
    }

    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard var destBuffer = destBuffer else {
            print("No destBuffer set")
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer")
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)

        guard let data = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to get base address")
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            return 
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var sourceBuffer = vImage_Buffer(
            data: data,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
        )

        let error = vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destBuffer, vImage_Flags(kvImageNoFlags))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        guard error == kvImageNoError else {
            print("Error during conversion: \(error)")
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self = self, let capturedFrameBuffer = self.frameBuffer, let capturedDestBuffer = self.destBuffer else {
                return
            }
            
            memcpy(capturedFrameBuffer, capturedDestBuffer.data, height * width * 3)
            
            self.isNextFrameAvailable?.pointee = 1
        }
    }
}
