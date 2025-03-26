//
//  ImageView.swift
//  mosaic
//
//  Created by Christian Bator on 12/14/2024
//

import AppKit

class ImageView: NSImageView {

    init() {
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false
        imageScaling = .scaleProportionallyUpOrDown
        setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        setContentCompressionResistancePriority(.defaultLow, for: .vertical)
    }
    
    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("Unimplemented")
    }
}
