//
//  ViewController.swift
//  MacVisualizer
//
//  Created by Christian Bator on 12/14/2024
//

import AppKit

public class ViewController: NSViewController {

    var layer: CALayer! {
        return view.layer
    }

    init() {
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("Unimplemented")
    }

    override public func viewDidLoad() {
        super.viewDidLoad()

        view.wantsLayer = true
    }
}
