//
//  ImageViewController.swift
//  MacVisualizer
//
//  Created by Christian Bator on 12/14/2024.
//

import AppKit

public class ImageViewController: ViewController {
    
    private let imageView: ImageView

    init(imageData: ImageData) {
        imageView = ImageView()
        super.init()

        update(with: imageData)
    }

    override public func viewDidLoad() {
        super.viewDidLoad()

        layer.backgroundColor = NSColor.systemGray.cgColor
        
        view.addSubview(imageView)
    
        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
    
    // MARK: ImageView Management

    private func createRepresentation(from imageData: ImageData) -> NSImageRep {
        var mutableData: UnsafeMutablePointer? = UnsafeMutablePointer(mutating: imageData.data)

        let bitmapRepresentation = NSBitmapImageRep(
            bitmapDataPlanes: &mutableData,
            pixelsWide: imageData.width,
            pixelsHigh: imageData.height,
            bitsPerSample: imageData.bitDepth,
            samplesPerPixel: imageData.channels,
            hasAlpha: false,
            isPlanar: false,
            colorSpaceName: imageData.channels == 1 ? .deviceWhite : .deviceRGB,
            bitmapFormat: [],
            bytesPerRow: (imageData.bitDepth / 8 * imageData.channels) * imageData.width,
            bitsPerPixel: imageData.bitDepth * imageData.channels
        )!
        
        return bitmapRepresentation
    }

    func update(with imageData: ImageData) {
        let representation = createRepresentation(from: imageData)
        
        let image = NSImage(size: NSSize(width: imageData.width, height: imageData.height))
        image.addRepresentation(representation)
        
        imageView.image = image
    }
}
