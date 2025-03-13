//
//  ImageData.swift
//  MacVisualizer
//
//  Created by Christian Bator on 2/9/2025.
//

import Foundation

struct ImageData {
    let data: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let channels: Int
    let bitDepth: Int = 8
}
