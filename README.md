<div align="center">
  <img src="assets/logo.svg" alt="Mosaic Logo" width="280" height="100">
  <p>
    An open source computer vision library in Mojo
    
  ![Language Badge](https://img.shields.io/badge/language-mojo-orange)
  ![GitHub License Badge](https://img.shields.io/github/license/christianbator/mosaic)
  ![CodeQL](https://github.com/christianbator/mosaic/workflows/CodeQL/badge.svg)
  </p>
</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a>
</p>

<br>

## Overview

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

#### Description

Mosaic is a cross-platform (currently only macOS) library for computer vision prototyping and production.

The library provides methods to decode and encode image files, represent images in various color spaces and with various numeric data types, process images, visualize results, and more.

<br>

## Installation

#### Prerequisites

- [Magic](https://docs.modular.com/magic/) (Mojo environment and package manager)

#### Using the magic cli:

```bash
magic add mosaic
```

<br>

## Examples

Clone the repo and build Mosaic:

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

#### Example List

- [Show image](examples/show_image.mojo)
