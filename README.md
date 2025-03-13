<h1 align="center">
  Mosaic
</h1>

<h4 align="center">An open source computer vision library in <a href="https://github.com/modular/max/tree/main/mojo">Mojo</a></h4>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#related">Related</a>
</p>
<br>

![CodeQL](https://github.com/christianbator/mosaic/workflows/CodeQL/badge.svg)

## Features

- Type-safe, memory-safe computer vision library
- One language for prototyping and production
  - Faster than python, simpler than c++
- Image reading / writing
  - [x] PNG
  - [x] JPEG
  - [x] HDR (linear float32)
- Data types and color spaces
  - [x] Any Mojo `DType` (UInt8, Int32, Float64, etc.) 
  - [x] Greyscale
  - [x] RGB
- Image processing
  - Transforms
    - [x] Flipping
  - Filtering
    - [x] 2D filtering with any kernel
- Video capture
  - [ ] macOS video capture
  - [ ] Linux video capture
- Visualization
  - [x] macOS visualizer
  - [ ] Linux visualizer

### Description

Mosaic is a cross-platform (currently only macOS) library for computer vision prototyping and production.

The library provides methods to decode and encode image files, represent images in various color spaces and with various numeric data types, process images, visualize results, and more.

## Installation

### Prerequisites

- [Magic](https://docs.modular.com/magic/)
  - Mojo environment and dependency manager

### Add to project

```bash
magic add mosaic
```

**NOTE: not currently available in the modular community channel. Coming soon...**

## Examples

To run the examples, clone the repo and build Mosaic:

```bash
git clone git@github.com:christianbator/mosaic.git
cd mosaic
magic run build
```

Then you can run the examples:

```zsh
cd examples
magic run mojo show_image.mojo
```

The following examples are available:

- [Show image](examples/show_image.mojo)
    - Decodes a PNG into a UInt8 representation in the RGB color space and displays it on screen
    - Usage: `magic run mojo show_image.mojo`

## Related

- [Mosaic website](https://mosaiclib.org)
- [Mojo website](https://docs.modular.com/mojo/manual/get-started/)
