// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#include "skl-test.h"

// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

enum {
  ALIGN = 4096,
  BATCH = 1,
  INPUT_HEIGHT = 10,
  INPUT_WIDTH = 10,
  INPUT_CHANNEL = 1024,
  FILTER_HEIGHT = 3,
  FILTER_WIDTH = 3,
  OUTPUT_HEIGHT = 5,
  OUTPUT_WIDTH = 5,
  STRIDE_HEIGHT = 2,
  STRIDE_WIDTH = 2,
  DILATION_HEIGHT = 1,
  DILATION_WIDTH = 1,
  PADDING_HEIGHT = 0,
  PADDING_WIDTH = 0,
  INPUT_LEN = BATCH * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL,
  M = BATCH * OUTPUT_HEIGHT * OUTPUT_WIDTH,
  K = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL,
  IM2ROW_OUTPUT_LEN = M * K
};

_Alignas(ALIGN) float input[INPUT_LEN];
_Alignas(ALIGN) float im2row_output[IM2ROW_OUTPUT_LEN];

#if defined(ENABLE_TEST)
float ref_im2row_output[IM2ROW_OUTPUT_LEN],
    test_im2row_output[IM2ROW_OUTPUT_LEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
typedef void (*im2row_hwc)(
    void *out, const void *in_batch, size_t element_size, int32_t in_h_origin,
    int32_t in_w_origin, int32_t in_c_origin, size_t input_height,
    size_t input_width, size_t filter_height, size_t filter_width,
    size_t patch_channel, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_height_stride,
    size_t input_width_stride, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements);

void skl_im2row_hwc_f32_wrapper(float *output, const float *input,
                                size_t batches, size_t input_height,
                                size_t input_width, size_t input_channel,
                                size_t filter_height, size_t filter_width,
                                size_t output_height, size_t output_width,
                                size_t padding_width, size_t padding_height,
                                size_t stride_width, size_t stride_height,
                                size_t dilation_width, size_t dilation_height,
                                im2row_hwc im2row_f) {
  const size_t m_len = batches * output_height * output_width;
  const size_t k_len = filter_height * filter_width * input_channel;

  size_t output_row_offset = 0;

  const size_t patch_begin_coord[3] = {0, 0, 0};

  const size_t input_width_stride = input_channel;
  const size_t input_height_stride = input_channel * input_width;
  const size_t input_batch_stride = input_height * input_width * input_channel;
  for (size_t m = 0; m < m_len; ++m) {
    size_t batch = (m / (output_width * output_height)) % batches;
    size_t out_h = (m / output_width) % output_height;
    size_t out_w = m % output_width;

    const int32_t in_h_origin =
        (int32_t)(out_h * stride_height) - (int32_t)padding_height;
    const int32_t in_w_origin =
        (int32_t)(out_w * stride_width) - (int32_t)padding_width;
    const int32_t in_c_origin = 0;

    const size_t in_offset = batch * input_batch_stride;
    const float *in_batch_tile = input + in_offset;

    float *row_tile = output + output_row_offset;

    im2row_f(row_tile, in_batch_tile, sizeof(float), in_h_origin, in_w_origin,
             in_c_origin, input_height, input_width, filter_height,
             filter_width, input_channel /* patch_channel */, dilation_height,
             dilation_width, input_height_stride, input_width_stride,
             0 /* zero value */, patch_begin_coord, k_len);

    output_row_offset += k_len;
  }
}

typedef void (*im2row_hwc_e32)(
    uint32_t *out, const uint32_t *in_batch, int32_t in_h_origin,
    int32_t in_w_origin, int32_t in_c_origin, size_t input_height,
    size_t input_width, size_t filter_height, size_t filter_width,
    size_t patch_channel, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_height_stride,
    size_t input_width_stride, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements);

void skl_im2row_hwc_f32_typed_wrapper(
    float *output, const float *input, size_t batches, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t padding_width, size_t padding_height, size_t stride_width,
    size_t stride_height, size_t dilation_width, size_t dilation_height,
    im2row_hwc_e32 im2row_f) {
  const size_t m_len = batches * output_height * output_width;
  const size_t k_len = filter_height * filter_width * input_channel;

  size_t output_row_offset = 0;

  const size_t patch_begin_coord[3] = {0, 0, 0};

  const size_t input_width_stride = input_channel;
  const size_t input_height_stride = input_channel * input_width;
  const size_t input_batch_stride = input_height * input_width * input_channel;

  for (size_t m = 0; m < m_len; ++m) {
    size_t batch = (m / (output_width * output_height)) % batches;
    size_t out_h = (m / output_width) % output_height;
    size_t out_w = m % output_width;

    const int32_t in_h_origin =
        (int32_t)(out_h * stride_height) - (int32_t)padding_height;
    const int32_t in_w_origin =
        (int32_t)(out_w * stride_width) - (int32_t)padding_width;
    const int32_t in_c_origin = 0;

    const size_t in_offset = batch * input_batch_stride;
    const float *in_batch_tile = input + in_offset;

    float *row_tile = output + output_row_offset;

    im2row_f((uint32_t *)row_tile, (uint32_t *)in_batch_tile, in_h_origin,
             in_w_origin, in_c_origin, input_height, input_width, filter_height,
             filter_width, input_channel /* patch_channel */, dilation_height,
             dilation_width, input_height_stride, input_width_stride,
             0 /* zero value */, patch_begin_coord, k_len);

    output_row_offset += k_len;
  }
}

int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_im2row_hwc_f32_wrapper(
      ref_im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL,
      FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, PADDING_WIDTH,
      PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT, DILATION_WIDTH,
      DILATION_HEIGHT, skl_im2row_generic_hwc);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < K; ++j) {
      if (test_im2row_output[i * M + j] != ref_im2row_output[i * M + j]) {
        printf("result [%zu, %zu] (%f) != reference (%f)\n", i, j,
               test_im2row_output[i * M + j], ref_im2row_output[i * M + j]);
        return 1;
      }
    }
  }

  return 0;
}
#endif // ENABLE_TEST

#define TEST_LABEL(S) #S ":\n"
#define PRINT_TEST_NAME(S) printf(TEST_LABEL(S));

int main(void) {
  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);

  /* Populate the matrices. */
  skl_test_init_f32(input, INPUT_LEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);
  skl_test_init_f32(im2row_output, IM2ROW_OUTPUT_LEN, SKL_TEST_MIN_F32,
                    SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  /* Make copies of A^T to write the reference and test outputs to. */
  memcpy(ref_im2row_output, im2row_output, IM2ROW_OUTPUT_LEN * sizeof(float));
  memcpy(test_im2row_output, im2row_output, IM2ROW_OUTPUT_LEN * sizeof(float));
  skl_im2row_hwc_f32_typed_wrapper(
      test_im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH,
      INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH,
      PADDING_WIDTH, PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT,
      DILATION_WIDTH, DILATION_HEIGHT, SKL_TEST_NAME);

  res += check_error();
#endif // ENABLE_TEST

#if defined(ENABLE_BENCHMARK)
  /* Warmup run */
  skl_im2row_hwc_f32_typed_wrapper(
      im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL,
      FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, PADDING_WIDTH,
      PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT, DILATION_WIDTH,
      DILATION_HEIGHT, SKL_TEST_NAME);

  /* Benchmark im2row. */
  riscv_fence();
  uint64_t c0 = riscv_read_mcycle();

  skl_im2row_hwc_f32_typed_wrapper(
      im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL,
      FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, PADDING_WIDTH,
      PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT, DILATION_WIDTH,
      DILATION_HEIGHT, SKL_TEST_NAME);

  riscv_fence();
  uint64_t c1 = riscv_read_mcycle();
  uint64_t cycles = c1 - c0;

  printf("Cycle count: %" PRIu64 "\n", cycles);
  printf("\n");
#endif // ENABLE_BENCHMARK

  return res;
}
