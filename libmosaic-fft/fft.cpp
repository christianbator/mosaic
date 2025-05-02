//
// fft.cpp
// mosaic
//
// Created by Christian Bator on 05/02/2025
//

#include "fft.h"

extern "C" {

    __attribute__((visibility("default")))
    void fft_float32(int rows, int cols, int components, const float* data_in, float* data_out, bool inverse) {
        const pocketfft::shape_t shape = {
            static_cast<unsigned long>(rows),
            static_cast<unsigned long>(cols)
        };
        
        const pocketfft::stride_t stride = {
            static_cast<long>(cols * sizeof(std::complex<float>) * components),
            static_cast<long>(sizeof(std::complex<float>) * components)
        };

        const pocketfft::shape_t axes = {0, 1};
        
        const float scaling_factor = inverse ? 1.0 / (rows * cols) : 1.0;

        for (int component = 0; component < components; component++) {
            pocketfft::c2c(
                shape,
                stride,
                stride,
                axes,
                !inverse,
                reinterpret_cast<const std::complex<float>*>(data_in) + component,
                reinterpret_cast<std::complex<float>*>(data_out) + component,
                scaling_factor
            );
        }
    }

    __attribute__((visibility("default")))
    void fft_float64(int rows, int cols, int components, const double* data_in, double* data_out, bool inverse) {
        const pocketfft::shape_t shape = {
            static_cast<unsigned long>(rows),
            static_cast<unsigned long>(cols)
        };
        
        const pocketfft::stride_t stride = {
            static_cast<long>(cols * sizeof(std::complex<float>) * components),
            static_cast<long>(sizeof(std::complex<float>) * components)
        };

        const pocketfft::shape_t axes = {0, 1};
        
        const double scaling_factor = inverse ? 1.0 / (rows * cols) : 1.0;

        for (int component = 0; component < components; component++) {
            pocketfft::c2c(
                shape,
                stride,
                stride,
                axes,
                !inverse,
                reinterpret_cast<const std::complex<double>*>(data_in) + component,
                reinterpret_cast<std::complex<double>*>(data_out) + component,
                scaling_factor
            );
        }
    }
}
