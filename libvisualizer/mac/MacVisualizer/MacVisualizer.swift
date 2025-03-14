//
//  MacVisualizer.swift
//  MacVisualizer
//
//  Created by Christian Bator on 2/9/2025.
//

import AppKit

@MainActor
let application = NSApplication.shared

@MainActor
let appDelegate = AppDelegate(application: application)

@MainActor
@_cdecl("show")
public func show(data: UnsafePointer<UInt8>, width: CInt, height: CInt, channels: CInt, windowTitle: UnsafePointer<CChar>) {
    autoreleasepool {
        let imageData = ImageData(data: data, width: Int(width), height: Int(height), channels: Int(channels))
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
