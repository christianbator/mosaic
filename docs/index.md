---
layout: home

hero:
  tagline: An image processing library in Mojo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/

features:
  - title: Type-safe images
    details: Color space and data type are compile-time parameters, so conversions are explicit and checked.
  - title: SIMD and parallel on the CPU
    details: Filters, geometric transforms and Fourier transforms vectorize and parallelize across cores.
  - title: GPU acceleration
    details: DeviceImage runs filters on the GPU, using host memory in place on unified memory systems.
---
