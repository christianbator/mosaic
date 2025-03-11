//
//  libcodec.c
//  mosaic
//
//  Created by Christian Bator on 12/14/2024
//

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "stdio.h"

//
// ImageInfo
//
struct ImageInfo {
    int width;
    int height;
    int bit_depth;
};

//
// Reading
//
__attribute__((visibility("default")))
int decode_image_info(uint8_t* raw_data, int raw_data_length, struct ImageInfo* image_info) {

    int is_valid = stbi_info_from_memory(raw_data, raw_data_length, &image_info->width, &image_info->height, NULL);

    if (!is_valid) {
        return 0;
    }

    if (stbi_is_hdr_from_memory(raw_data, raw_data_length)) {
        image_info->bit_depth = 32;
    }
    else if (stbi_is_16_bit_from_memory(raw_data, raw_data_length)) {
        image_info->bit_depth = 16;
    }
    else {
        image_info->bit_depth = 8;
    }

    return 1;
}

__attribute__((visibility("default")))
int decode_image_data_uint8(uint8_t* raw_data, int raw_data_length, int desired_channels, uint8_t* image_data) {
    
    int width, height;
    uint8_t* decoded_image_data = stbi_load_from_memory(raw_data, raw_data_length, &width, &height, NULL, desired_channels);

    if (decoded_image_data == NULL) {
        return 0;
    }

    int bytes_per_channel = 1;
    int byte_count = width * height * desired_channels * bytes_per_channel;
    memcpy(image_data, decoded_image_data, byte_count);

    stbi_image_free(decoded_image_data);

    return 1;
}

__attribute__((visibility("default")))
int decode_image_data_uint16(uint8_t* raw_data, int raw_data_length, int desired_channels, uint16_t* image_data) {
    
    int width, height;
    uint16_t* decoded_image_data = stbi_load_16_from_memory(raw_data, raw_data_length, &width, &height, NULL, desired_channels);

    if (decoded_image_data == NULL) {
        return 0;
    }

    int bytes_per_channel = 2;
    int byte_count = width * height * desired_channels * bytes_per_channel;
    memcpy(image_data, decoded_image_data, byte_count);

    stbi_image_free(decoded_image_data);

    return 1;
}

__attribute__((visibility("default")))
int decode_image_data_float32(uint8_t* raw_data, int raw_data_length, int desired_channels, float* image_data) {
    assert(sizeof(float) == 4);

    int width, height;
    float* decoded_image_data = stbi_loadf_from_memory(raw_data, raw_data_length, &width, &height, NULL, desired_channels);

    if (decoded_image_data == NULL) {
        return 0;
    }

    int bytes_per_channel = 4;
    int byte_count = width * height * desired_channels * bytes_per_channel;
    memcpy(image_data, decoded_image_data, byte_count);

    stbi_image_free(decoded_image_data);

    return 1;
}

//
// Writing
//
__attribute__((visibility("default")))
int write_image_data_png(const char* filename, uint8_t* data, int width, int height, int channels) {
    return stbi_write_png(filename, width, height, channels, data, width * channels);
}

__attribute__((visibility("default")))
int write_image_data_jpeg(const char* filename, uint8_t* data, int width, int height, int channels) {
    // JPEG quality value in range [1, 100]
    const int quality = 85; 

    return stbi_write_jpg(filename, width, height, channels, data, quality);
}
