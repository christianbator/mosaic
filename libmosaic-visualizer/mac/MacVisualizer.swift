//
//  MacVisualizer.swift
//  mosaic
//
//  Created by Christian Bator on 02/09/2025
//

import AppKit

@MainActor
let application = NSApplication.shared

@MainActor
let appDelegate = AppDelegate(application: application)

@MainActor
@_cdecl("show")
public func show(data: UnsafeMutablePointer<UInt8>, height: CInt, width: CInt, channels: CInt, windowTitle: UnsafePointer<CChar>) {
    autoreleasepool {
        let count = Int(height * width * channels)
        let dataCopy = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        memcpy(dataCopy, data, count)

        let imageData = ImageData(data: dataCopy, height: Int(height), width: Int(width), channels: Int(channels))
        let windowTitleString = String(cString: windowTitle)
    
        appDelegate.show(imageData: imageData, inWindowTitled: windowTitleString)
    }
}

@MainActor
@_cdecl("wait")
public func wait(timeout: CFloat) -> CBool {
    let timeoutInterval = TimeInterval(timeout)
    let start = Date()
    
    if !appDelegate.isRunning {
        application.finishLaunching()
        application.nextEvent(matching: .any, until: nil, inMode: .default, dequeue: true)
    }

    while true {
        autoreleasepool {
            if let event = application.nextEvent(matching: .any, until: nil, inMode: .default, dequeue: true) {
                application.sendEvent(event)
                application.updateWindows()
            }
        }
            
        if application.windows.isEmpty {
            return false
        }
        
        if Date().timeIntervalSince(start) >= timeoutInterval {
            return true
        }
    }
}
