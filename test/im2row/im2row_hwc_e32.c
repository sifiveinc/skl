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
_Alignas(ALIGN) float test_im2row_output[IM2ROW_OUTPUT_LEN];

#if defined(ENABLE_TEST)
float ref_im2row_output[IM2ROW_OUTPUT_LEN];
#endif // ENABLE_TEST

void skl_im2row_hwc_f32_scalar_wrapper(
    float *output, const float *input, size_t batches, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t padding_width, size_t padding_height, size_t stride_width,
    size_t stride_height, size_t dilation_width, size_t dilation_height) {
  const size_t input_width_stride = input_channel;
  const size_t input_height_stride = input_channel * input_width;
  const size_t input_batch_stride = input_height * input_width * input_channel;

  for (size_t b = 0; b < batches; b++) {
    skl_im2row_generic_hwc(
        output, input, sizeof(float), input_height, input_width, input_channel,
        input_height_stride, input_width_stride, filter_height, filter_width,
        output_height, output_width, padding_width, padding_height,
        stride_width, stride_height, dilation_width, dilation_height, 0);
    input += input_batch_stride;
  }
}

void skl_im2row_hwc_f32_zve32x_wrapper(
    float *output, const float *input, size_t batches, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t output_height, size_t output_width,
    size_t padding_width, size_t padding_height, size_t stride_width,
    size_t stride_height, size_t dilation_width, size_t dilation_height) {
  const size_t input_width_stride = input_channel;
  const size_t input_height_stride = input_channel * input_width;
  const size_t input_batch_stride = input_height * input_width * input_channel;

  for (size_t b = 0; b < batches; b++) {
    skl_im2row_hwc_e32_zve32x(
        (uint32_t *)output, (const uint32_t *)input, input_height, input_width,
        input_channel, input_height_stride, input_width_stride, filter_height,
        filter_width, output_height, output_width, padding_width,
        padding_height, stride_width, stride_height, dilation_width,
        dilation_height, 0);
    input += input_batch_stride;
  }
}

#if defined(ENABLE_TEST)
int check_error_float(char *func_name) {
  printf("Check Func %s:\n", func_name);

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

#if defined(ENABLE_TEST)
#define GENERATE_GOLDEN(DATA_TYPE)                                             \
  memcpy(ref_im2row_output, im2row_output,                                     \
         IM2ROW_OUTPUT_LEN * sizeof(DATA_TYPE));                               \
  skl_im2row_hwc_f32_scalar_wrapper(                                           \
      ref_im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH,              \
      INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, \
      PADDING_WIDTH, PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT,              \
      DILATION_WIDTH, DILATION_HEIGHT);
#else
#define GENERATE_GOLDEN(DATA_TYPE)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(NAME, DATA_TYPE) res += check_error_##DATA_TYPE(#NAME);
#else
#define CHECK_RESULT(NAME, DATA_TYPE)
#endif

#define RUN(NAME, FUNCTION, DATA_TYPE, ...)                                    \
  memcpy(test_im2row_output, im2row_output,                                    \
         IM2ROW_OUTPUT_LEN * sizeof(DATA_TYPE));                               \
  SKL_BENCHMARK_RUN(#NAME, IM2ROW_OUTPUT_LEN, SKL_TEST_WARMUP, FUNCTION,       \
                    __VA_ARGS__);                                              \
  CHECK_RESULT(NAME, DATA_TYPE);

int main(void) {
  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);

  /* Populate the matrices. */
  skl_test_init_f32(input, INPUT_LEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);
  skl_test_init_f32(im2row_output, IM2ROW_OUTPUT_LEN, SKL_TEST_MIN_F32,
                    SKL_TEST_MAX_F32);

  GENERATE_GOLDEN(float);

  RUN(skl_im2row_hwc_e32_zve32x, skl_im2row_hwc_f32_zve32x_wrapper, float,
      test_im2row_output, input, BATCH, INPUT_HEIGHT, INPUT_WIDTH,
      INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH,
      PADDING_WIDTH, PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT,
      DILATION_WIDTH, DILATION_HEIGHT);

  return res;
}
