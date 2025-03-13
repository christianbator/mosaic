<div align="center">
  <img src="assets/logo.png" alt="Mosaic Logo" width="280" height="100">
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

## Overview

#### Description

Mosaic is a cross-platform (currently only macOS) computer vision library for prototyping and production.

The library provides methods to decode and encode image files, represent images in various color spaces, process images, visualize results, and more.

#### Features

- Type-safe
- Memory-safe
- Image file reading / writing
- Data type conversion
- Color space conversion
- Image processing
  - Transforms
  - Filtering
- Video capture
- Visualization

## Installation

#### Prerequisites

- [Magic](https://docs.modular.com/magic/) (the Mojo environment and package manager)

#### Using the magic cli:

```bash
magic add mosaic
```

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
