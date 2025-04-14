//
// VideoCapture.swift
// mosaic
//
// Created by Christian Bator on 03/24/2025
//

import AVFoundation
import Accelerate


// MARK: - C Interface

enum DeviceQuery {
    case index(Int)
    case name(String)
}

public enum ColorSpace: CInt {
    case greyscale
    case rgb
    
    var channels: Int {
        switch self {
        case .greyscale:
            return 1
        case .rgb:
            return 3
        }
    }
}

struct VideoCaptureDimensions {
    var height: CInt
    var width: CInt
}

@_cdecl("initialize_with_index")
public func initialize(index: CInt) -> OpaquePointer {
    let videoCapture = VideoCapture(deviceQuery: .index(Int(index)))
    let rawPointer = Unmanaged<VideoCapture>.passUnretained(videoCapture).toOpaque()

    return OpaquePointer(rawPointer)
}

@_cdecl("initialize_with_name")
public func initialize(name: UnsafePointer<CChar>) -> OpaquePointer {
    let videoCapture = VideoCapture(deviceQuery: .name(String(cString: name)))
    let rawPointer = Unmanaged<VideoCapture>.passUnretained(videoCapture).toOpaque()

    return OpaquePointer(rawPointer)
}

@_cdecl("open")
public func open(pointer: OpaquePointer, colorSpace: CInt, dimensions: UnsafeMutableRawPointer) -> CBool {
    let dimensions = dimensions.assumingMemoryBound(to: VideoCaptureDimensions.self)
    
    return videoCapture(from: pointer).open(colorSpace: ColorSpace(rawValue: colorSpace)!, dimensions: dimensions)
}

@_cdecl("start")
public func start(pointer: OpaquePointer, frameBuffer: UnsafeMutablePointer<UInt8>) {
    videoCapture(from: pointer).start(frameBuffer: frameBuffer)
}

@_cdecl("is_next_frame_available")
public func isNextFrameAvailable(pointer: OpaquePointer) -> CBool {
    return videoCapture(from: pointer).isNextFrameAvailable
}

@_cdecl("did_read_next_frame")
public func didReadNextFrame(pointer: OpaquePointer) {
    return videoCapture(from: pointer).didReadNextFrame()
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
    
    private(set) var isNextFrameAvailable: Bool = false
    
    private let deviceQuery: DeviceQuery
    private var session: AVCaptureSession!
    private var colorSpace: ColorSpace!
    private var height: Int = 0
    private var width: Int = 0
    private var destBuffer: vImage_Buffer!

    init(deviceQuery: DeviceQuery) {
        self.deviceQuery = deviceQuery
        super.init()
        
        _ = Unmanaged<VideoCapture>.passRetained(self)
    }
    
    func open(colorSpace: ColorSpace, dimensions: UnsafeMutablePointer<VideoCaptureDimensions>) -> Bool {
        do {
            let session = AVCaptureSession()
            session.beginConfiguration()
            
            let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices
            
            guard let device: AVCaptureDevice = {
                switch deviceQuery {
                case .index(let index):
                    let sortedDevices = devices.sorted { $0.localizedName < $1.localizedName }
                    
                    guard sortedDevices.indices.contains(index) else {
                        print("Error initializing capture device at index \(index): device not found")
                        return nil
                    }
                    
                    return sortedDevices[index]
                case .name(let name):
                    guard let device = devices.first(where: { $0.localizedName == name }) else {
                        print("Error initializing capture device with name '\(name)': device not found")
                        return nil
                    }
                    
                    return device
                }
            }() else {
                return false
            }
            
            let input = try AVCaptureDeviceInput(device: device)
            
            if session.canAddInput(input) {
                session.addInput(input)
            }
            else {
                print("Error initializing capture device: failed to add input")
                return false
            }
            
            let output = AVCaptureVideoDataOutput()

            switch colorSpace {
            case .greyscale:
                if output.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
                    output.videoSettings = [
                        (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                    ]
                }
                else if output.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
                    output.videoSettings = [
                        (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                    ]
                }
                else {
                    return false
                }
            case .rgb:
                guard output.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_32ARGB) else {
                    return false
                }
                
                output.videoSettings = [
                    (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32ARGB
                ]
            }
            
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
            self.colorSpace = colorSpace
            
            let activeFormatDimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
            
            self.height = Int(activeFormatDimensions.height)
            self.width = Int(activeFormatDimensions.width)
            
            dimensions.pointee.height = activeFormatDimensions.height
            dimensions.pointee.width = activeFormatDimensions.width
            
            return true
        }
        catch {
            print("Error initializing capture device: \(error)")
            return false
        }
    }
    
    func start(frameBuffer: UnsafeMutablePointer<UInt8>) {
        destBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(frameBuffer),
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * colorSpace.channels
        )
        
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

        let height = CVPixelBufferGetHeight(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        
        switch colorSpace! {
        case .greyscale:
            guard let greyscaleData = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
                print("Failed to get base address of greyscale plane")
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                return
            }
            
            memcpy(destBuffer.data, greyscaleData, width * height)
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            
            isNextFrameAvailable = true
        case .rgb:
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
}

// MARK: - Utilities
                             
func printDeviceInfo(_ device: AVCaptureDevice) {
    print("Device: \(device.localizedName) (\(device.uniqueID))")
    print("Format: \(pixelFormatName(device.activeFormat.formatDescription.mediaSubType))")
    
    var frameRates = String()
    for (index, frameRateRange) in device.activeFormat.videoSupportedFrameRateRanges.enumerated() {
        frameRates.append(String(format: "%.2f", frameRateRange.maxFrameRate))
        
        if index != device.activeFormat.videoSupportedFrameRateRanges.count - 1 {
            frameRates.append(" ")
        }
    }
    
    print("Available frame rates: { \(frameRates) fps }")
}

func pixelFormatName(_ pixelFormat: CMFormatDescription.MediaSubType) -> String {
    switch pixelFormat.rawValue {
         case kCVPixelFormatType_1Monochrome:                   return "kCVPixelFormatType_1Monochrome"
         case kCVPixelFormatType_2Indexed:                      return "kCVPixelFormatType_2Indexed"
         case kCVPixelFormatType_4Indexed:                      return "kCVPixelFormatType_4Indexed"
         case kCVPixelFormatType_8Indexed:                      return "kCVPixelFormatType_8Indexed"
         case kCVPixelFormatType_1IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_1IndexedGray_WhiteIsZero"
         case kCVPixelFormatType_2IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_2IndexedGray_WhiteIsZero"
         case kCVPixelFormatType_4IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_4IndexedGray_WhiteIsZero"
         case kCVPixelFormatType_8IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_8IndexedGray_WhiteIsZero"
         case kCVPixelFormatType_16BE555:                       return "kCVPixelFormatType_16BE555"
         case kCVPixelFormatType_16LE555:                       return "kCVPixelFormatType_16LE555"
         case kCVPixelFormatType_16LE5551:                      return "kCVPixelFormatType_16LE5551"
         case kCVPixelFormatType_16BE565:                       return "kCVPixelFormatType_16BE565"
         case kCVPixelFormatType_16LE565:                       return "kCVPixelFormatType_16LE565"
         case kCVPixelFormatType_24RGB:                         return "kCVPixelFormatType_24RGB"
         case kCVPixelFormatType_24BGR:                         return "kCVPixelFormatType_24BGR"
         case kCVPixelFormatType_32ARGB:                        return "kCVPixelFormatType_32ARGB"
         case kCVPixelFormatType_32BGRA:                        return "kCVPixelFormatType_32BGRA"
         case kCVPixelFormatType_32ABGR:                        return "kCVPixelFormatType_32ABGR"
         case kCVPixelFormatType_32RGBA:                        return "kCVPixelFormatType_32RGBA"
         case kCVPixelFormatType_64ARGB:                        return "kCVPixelFormatType_64ARGB"
         case kCVPixelFormatType_48RGB:                         return "kCVPixelFormatType_48RGB"
         case kCVPixelFormatType_32AlphaGray:                   return "kCVPixelFormatType_32AlphaGray"
         case kCVPixelFormatType_16Gray:                        return "kCVPixelFormatType_16Gray"
         case kCVPixelFormatType_30RGB:                         return "kCVPixelFormatType_30RGB"
         case kCVPixelFormatType_422YpCbCr8:                    return "kCVPixelFormatType_422YpCbCr8"
         case kCVPixelFormatType_4444YpCbCrA8:                  return "kCVPixelFormatType_4444YpCbCrA8"
         case kCVPixelFormatType_4444YpCbCrA8R:                 return "kCVPixelFormatType_4444YpCbCrA8R"
         case kCVPixelFormatType_4444AYpCbCr8:                  return "kCVPixelFormatType_4444AYpCbCr8"
         case kCVPixelFormatType_4444AYpCbCr16:                 return "kCVPixelFormatType_4444AYpCbCr16"
         case kCVPixelFormatType_444YpCbCr8:                    return "kCVPixelFormatType_444YpCbCr8"
         case kCVPixelFormatType_422YpCbCr16:                   return "kCVPixelFormatType_422YpCbCr16"
         case kCVPixelFormatType_422YpCbCr10:                   return "kCVPixelFormatType_422YpCbCr10"
         case kCVPixelFormatType_444YpCbCr10:                   return "kCVPixelFormatType_444YpCbCr10"
         case kCVPixelFormatType_420YpCbCr8Planar:              return "kCVPixelFormatType_420YpCbCr8Planar"
         case kCVPixelFormatType_420YpCbCr8PlanarFullRange:     return "kCVPixelFormatType_420YpCbCr8PlanarFullRange"
         case kCVPixelFormatType_422YpCbCr_4A_8BiPlanar:        return "kCVPixelFormatType_422YpCbCr_4A_8BiPlanar"
         case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:  return "kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange"
         case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:   return "kCVPixelFormatType_420YpCbCr8BiPlanarFullRange"
         case kCVPixelFormatType_422YpCbCr8_yuvs:               return "kCVPixelFormatType_422YpCbCr8_yuvs"
         case kCVPixelFormatType_422YpCbCr8FullRange:           return "kCVPixelFormatType_422YpCbCr8FullRange"
         case kCVPixelFormatType_OneComponent8:                 return "kCVPixelFormatType_OneComponent8"
         case kCVPixelFormatType_TwoComponent8:                 return "kCVPixelFormatType_TwoComponent8"
         case kCVPixelFormatType_30RGBLEPackedWideGamut:        return "kCVPixelFormatType_30RGBLEPackedWideGamut"
         case kCVPixelFormatType_OneComponent16Half:            return "kCVPixelFormatType_OneComponent16Half"
         case kCVPixelFormatType_OneComponent32Float:           return "kCVPixelFormatType_OneComponent32Float"
         case kCVPixelFormatType_TwoComponent16Half:            return "kCVPixelFormatType_TwoComponent16Half"
         case kCVPixelFormatType_TwoComponent32Float:           return "kCVPixelFormatType_TwoComponent32Float"
         case kCVPixelFormatType_64RGBAHalf:                    return "kCVPixelFormatType_64RGBAHalf"
         case kCVPixelFormatType_128RGBAFloat:                  return "kCVPixelFormatType_128RGBAFloat"
         case kCVPixelFormatType_14Bayer_GRBG:                  return "kCVPixelFormatType_14Bayer_GRBG"
         case kCVPixelFormatType_14Bayer_RGGB:                  return "kCVPixelFormatType_14Bayer_RGGB"
         case kCVPixelFormatType_14Bayer_BGGR:                  return "kCVPixelFormatType_14Bayer_BGGR"
         case kCVPixelFormatType_14Bayer_GBRG:                  return "kCVPixelFormatType_14Bayer_GBRG"
         default:                                               return "UNKNOWN"
     }
 }
