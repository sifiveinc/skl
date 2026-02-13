// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file conv2d_nhwc_f32.c
 * @brief Implementation of 2D convolution functions for NHWC layout with
 * float32 data type
 */

#include <assert.h>
#include <math.h> // fma
#include <stddef.h>
#include <stdint.h>

#include "./conv2d_nhwc_f32.h"

#if defined(__riscv_zve32x)
#include "im2row/im2row_hwc_zve32x.h"
#endif

#if defined(__riscv_zve32f)
#include "gemm/rvv/gemm_f32_f32_f32_zve32f_x390.h"
#endif

void conv2d_io_nhwc_filter_hwio_f32_f32_f32_scalar(
    float *output, const float *input, const float *filter, size_t batches,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height) {
  for (size_t batch = 0; batch < batches; ++batch) {
    for (size_t out_h = 0; out_h < output_height; ++out_h) {
      const int32_t in_h_origin =
          (int32_t)(out_h * stride_height) - (int32_t)padding_height;
      for (size_t out_w = 0; out_w < output_width; ++out_w) {
        const int32_t in_w_origin =
            (int32_t)(out_w * stride_width) - (int32_t)padding_width;
        for (size_t out_c = 0; out_c < output_channel; ++out_c) {
          float acc = 0.f;
          for (size_t filter_h = 0; filter_h < filter_height; ++filter_h) {
            const int32_t in_h =
                in_h_origin + (int32_t)(dilation_height * filter_h);
            for (size_t filter_w = 0; filter_w < filter_width; ++filter_w) {
              const int32_t in_w =
                  in_w_origin + (int32_t)(dilation_width * filter_w);

              if ((in_w >= 0) && ((size_t)in_w < input_width) && (in_h >= 0) &&
                  ((size_t)in_h < input_height)) {
                for (size_t in_c = 0; in_c < input_channel; ++in_c) {
                  float input_val =
                      input[((batch * input_height + in_h) * input_width +
                             in_w) *
                                input_channel +
                            in_c];
                  // HWIO
                  size_t filter_offset =
                      ((filter_h * filter_width + filter_w) * input_channel +
                       in_c) *
                          output_channel +
                      out_c;
                  float filter_val = filter[filter_offset];
                  acc = fmaf(input_val, filter_val, acc);
                } // end for
              } // end if
            }
          }

          output[((batch * output_height + out_h) * output_width + out_w) *
                     output_channel +
                 out_c] = acc;
        }
      }
    }
  }
}

void conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390(
    float *output, float *gemm_input, const float *input, const float *filter,
    size_t batches, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t padding_width, size_t padding_height, size_t stride_width,
    size_t stride_height, size_t dilation_width, size_t dilation_height) {
  const size_t m_len = batches * output_height * output_width;
  const size_t n_len = output_channel;
  const size_t k_len = filter_height * filter_width * input_channel;

  size_t output_row_offset = 0;

  const size_t patch_begin_coord[3] = {0, 0, 0};

  for (size_t m = 0; m < m_len; ++m) {
    size_t batch = (m / (output_width * output_height)) % batches;
    size_t out_h = (m / output_width) % output_height;
    size_t out_w = m % output_width;

    const int32_t in_w_origin =
        (int32_t)(out_w * stride_width) - (int32_t)padding_width;
    const int32_t in_h_origin =
        (int32_t)(out_h * stride_height) - (int32_t)padding_height;

    const size_t in_offset = batch * input_height * input_width * input_channel;
    const float *in_batch_tile = input + in_offset;

    float *row_tile = gemm_input + output_row_offset;

    skl_im2row_hwc_zve32x(row_tile, in_batch_tile, sizeof(float), in_w_origin,
                          in_h_origin, input_height, input_width, input_channel,
                          filter_height, filter_width, dilation_width,
                          dilation_height, 0, patch_begin_coord, k_len);

    output_row_offset += k_len;
  }

  skl_gemm_f32_f32_f32_zve32f_x390(
      m_len, n_len, k_len, 1 /* alpha */, gemm_input, k_len /* rsa */, filter,
      n_len /* rsb */, 0 /* beta */, output, n_len /* rsc */);
}

void conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390(
    float *output, const float *input, const float *filter, size_t batches,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height) {

  (void)input_height;
  (void)input_width;

  // Verify this is indeed a 1x1 convolution with stride=1, dilation=1
  assert(filter_height == 1 && filter_width == 1);
  assert(stride_height == 1 && stride_width == 1);
  assert(dilation_height == 1 && dilation_width == 1);
  assert(padding_height == 0 && padding_width == 0);

  (void)filter_height;
  (void)filter_width;
  (void)stride_height;
  (void)stride_width;
  (void)dilation_height;
  (void)dilation_width;
  (void)padding_height;
  (void)padding_width;

  const size_t m_len = batches * output_height * output_width;
  const size_t n_len = output_channel;
  const size_t k_len = input_channel;

  // Direct GEMM: output = input * filter
  skl_gemm_f32_f32_f32_zve32f_x390(m_len, n_len, k_len, 1.0f /* alpha */, input,
                                   k_len /* rsa */, filter, n_len /* rsb */,
                                   0.0f /* beta */, output, n_len /* rsc */);
}
