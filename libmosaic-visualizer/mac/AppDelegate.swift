//
//  AppDelegate.swift
//  mosaic
//
//  Created by Christian Bator on 12/14/2024
//

import AppKit

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {

    private(set) var isRunning: Bool = false
    
    private var application: NSApplication {
        return NSApplication.shared
    }
    
    // MARK: Initialization

    init(application: NSApplication) {
        super.init()
        
        application.delegate = self
        application.setActivationPolicy(.regular)
        application.mainMenu = createMenu()
    }

    private func createMenu() -> NSMenu {
        let mainMenu = NSMenu()

        let mainMenuItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        mainMenu.addItem(mainMenuItem)

        let appMenu = NSMenu()

        let resetWindowSizeMenuItem = NSMenuItem(title: "Reset Window Size", action: #selector(resetWindowSize), keyEquivalent: "r")
        appMenu.addItem(resetWindowSizeMenuItem)
        
        let closeWindowMenuItem = NSMenuItem(title: "Close Window", action: #selector(closeWindow), keyEquivalent: "w")
        appMenu.addItem(closeWindowMenuItem)
        
        let closeAllWindowsMenuItem = NSMenuItem(title: "Close All Windows", action: #selector(closeAllWindows), keyEquivalent: "w")
        closeAllWindowsMenuItem.keyEquivalentModifierMask = [.command, .shift]
        appMenu.addItem(closeAllWindowsMenuItem)

        mainMenuItem.submenu = appMenu
        
        return mainMenu
    }

    @objc
    private func resetWindowSize() {
        guard let window = application.keyWindow, let imageViewController = window.contentViewController as? ImageViewController else {
            return
        }

        let scaledIntrinsicContentSize = scaledWindowContentSize(for: imageViewController.imageSize)
        window.setContentSize(scaledIntrinsicContentSize)
    }

    @objc
    private func closeWindow() {
        application.keyWindow?.close()
    }
    
    @objc
    private func closeAllWindows() {
        for window in application.windows {
            window.close()
        }
    }

    // MARK: NSApplicationDelegate
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        application.activate()
        isRunning = true
    }
    
    func applicationWillTerminate(_ notification: Notification) {
        isRunning = false
    }

    // MARK: Window Management

    func show(imageData: ImageData, inWindowTitled windowTitle: String) {
        if let window = window(titled: windowTitle) {
            let imageViewController = window.contentViewController as! ImageViewController
            imageViewController.update(with: imageData)
        }
        else {
            let viewController = ImageViewController(imageData: imageData)

            let window = NSWindow(contentViewController: viewController)
            window.title = windowTitle
            window.isReleasedWhenClosed = true
            
            let scaledIntrinsicContentSize = scaledWindowContentSize(from: imageData)
            window.setContentSize(scaledIntrinsicContentSize)
            centerWindow(window)
            window.delegate = self
            
            window.makeKeyAndOrderFront(nil)
        }
    }

    private func window(titled windowTitle: String) -> NSWindow? {
        return application.windows.first { window in
            return window.title == windowTitle
        }
    }

    private func scaledWindowContentSize(from imageData: ImageData) -> NSSize {
        return scaledWindowContentSize(for:
            NSSize(width: imageData.width, height: imageData.height)
        )
    }
    
    private func scaledWindowContentSize(for size: NSSize) -> NSSize {
        let width = size.width
        let height = size.height

        if let screenFrame = NSScreen.main?.frame, width > screenFrame.width || height > screenFrame.height {
            let aspectRatio = width / height
            
            var scaledWidth = screenFrame.width
            var scaledHeight = screenFrame.width / aspectRatio
            
            if scaledHeight > screenFrame.height {
                scaledHeight = screenFrame.height
                scaledWidth = scaledHeight * aspectRatio
            }
            
            return NSSize(width: scaledWidth, height: scaledHeight)
        }
        else {
            return NSSize(width: width, height: height)
        }
    }
    
    private func centerWindow(_ window: NSWindow) {
        let screenFrame = NSScreen.main?.frame ?? .zero
        let windowWidth = window.frame.width
        let windowHeight = window.frame.height
        
        let centeredOrigin = NSPoint(
            x: (screenFrame.width - windowWidth) / 2,
            y: 1.2 * (screenFrame.height - windowHeight) / 2
        )
    
        window.setFrameOrigin(centeredOrigin)
    }
    
    // MARK: NSWindowDelegate

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        return true
    }
    
    func windowWillClose(_ notification: Notification) {
        guard let closedWindow = notification.object as? NSWindow else {
            return
        }
        
        if let nextWindow = application.orderedWindows.first(where: { $0 != closedWindow }) {
            nextWindow.makeKeyAndOrderFront(nil)
        }
    }
}
