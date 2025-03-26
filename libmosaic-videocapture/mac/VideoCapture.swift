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

@MainActor
@_cdecl("initialize")
public func initialize() -> OpaquePointer {
    let videoCapture = VideoCapture()
    let rawPointer = Unmanaged<VideoCapture>.passUnretained(videoCapture).toOpaque()

    return OpaquePointer(rawPointer)
}

@MainActor
@_cdecl("open")
public func open(pointer: OpaquePointer, dimensions: UnsafeMutableRawPointer) -> CBool {
    let dimensions = dimensions.assumingMemoryBound(to: VideoCaptureDimensions.self)
    
    return videoCapture(from: pointer).open(dimensions: dimensions)
}

@MainActor
@_cdecl("start")
public func start(pointer: OpaquePointer, frameBuffer: UnsafeMutablePointer<UInt8>) {
    videoCapture(from: pointer).start(frameBuffer: frameBuffer)
}

@MainActor
@_cdecl("is_next_frame_available")
public func isNextFrameAvailable(pointer: OpaquePointer) -> CBool {
    return videoCapture(from: pointer).isNextFrameAvailable
}

@MainActor
@_cdecl("did_read_next_frame")
public func didReadNextFrame(pointer: OpaquePointer) {
    return videoCapture(from: pointer).didReadNextFrame()
}

@MainActor
@_cdecl("stop")
public func stop(pointer: OpaquePointer) {
    videoCapture(from: pointer).stop()
}

@MainActor
@_cdecl("deinitialize")
public func deinitialize(pointer: OpaquePointer) {
    videoCapture(from: pointer).release()
}

// MARK: Pointer Conversion

@MainActor
private func videoCapture(from pointer: OpaquePointer) -> VideoCapture {
    let rawPointer = UnsafeRawPointer(pointer)
    return Unmanaged<VideoCapture>.fromOpaque(rawPointer).takeUnretainedValue()
}

// MARK: - VideoCapture

@MainActor
class VideoCapture: NSObject, @preconcurrency AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private(set) var isNextFrameAvailable: CBool = false
    
    private var session: AVCaptureSession?
    private var destBuffer: vImage_Buffer?

    override init() {
        super.init()
        
        _ = Unmanaged<VideoCapture>.passRetained(self)
    }
    
    func open(dimensions: UnsafeMutablePointer<VideoCaptureDimensions>) -> CBool {
        let session = AVCaptureSession()
        
        session.beginConfiguration()
        
        guard let device = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices.first else {
            print("Error initializing capture device: no devices available")
            return false
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
                return false
            }
            
            let output = AVCaptureVideoDataOutput()

            output.videoSettings = [
                (kCVPixelBufferPixelFormatTypeKey as String): Int(kCVPixelFormatType_32ARGB)
            ]

            output.setSampleBufferDelegate(self, queue: DispatchQueue.main)

            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            else {
                print("Error initializing capture device: failed to add output")
                return false
            }
            
            session.commitConfiguration()

            self.session = session
            
            destBuffer = vImage_Buffer(
                data: nil,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: width * 3
            )
            
            dimensions.pointee.width = CInt(width)
            dimensions.pointee.height = CInt(height)
            
            return true
        }
        catch {
            print("Error initializing capture device: \(error)")
            return false
        }
    }
    
    func start(frameBuffer: UnsafeMutablePointer<UInt8>) {
        destBuffer?.data = UnsafeMutableRawPointer(frameBuffer)
        isNextFrameAvailable = true
        
        session?.startRunning()
    }
    
    func didReadNextFrame() {
        isNextFrameAvailable = false
    }
    
    func stop() {
        session?.stopRunning()
    }

    func release() {
        stop()
        session = nil
        
        Unmanaged<VideoCapture>.passUnretained(self).release()
    }

    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard var destBuffer = destBuffer, destBuffer.data != nil else {
            print("Destination buffer not set")
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer")
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)

        guard let data = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to get base address of pixel buffer")
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

        if error == kvImageNoError {
            isNextFrameAvailable = true
        }
        else {
            print("Error during conversion to RGB: \(error)")
        }
    }
}
