//
//  ImageData.swift
//  mosaic
//
//  Created by Christian Bator on 02/09/2025
//

import Foundation

struct ImageData {
    let data: UnsafePointer<UInt8>
    let height: Int
    let width: Int
    let channels: Int
    let bitDepth: Int = 8
}
