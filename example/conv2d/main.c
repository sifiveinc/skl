// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

// convolution 2D test configuration
#ifdef TEST_IM2ROW_GEMM
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
  OUTPUT_CHANNEL = 512,
  STRIDE_HEIGHT = 2,
  STRIDE_WIDTH = 2,
  DILATION_HEIGHT = 1,
  DILATION_WIDTH = 1,
  PADDING_HEIGHT = 0,
  PADDING_WIDTH = 0,
  INPUT_SIZE = BATCH * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL,
  FILTER_SIZE = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL * OUTPUT_CHANNEL,
  OUTPUT_SIZE = BATCH * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL,
  M = BATCH * OUTPUT_HEIGHT * OUTPUT_WIDTH,
  K = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL,
  IM2ROW_SIZE = M * K
};
#else
enum {
  ALIGN = 4096,
  BATCH = 1,
  INPUT_HEIGHT = 10,
  INPUT_WIDTH = 10,
  INPUT_CHANNEL = 1024,
  FILTER_HEIGHT = 1,
  FILTER_WIDTH = 1,
  OUTPUT_HEIGHT = 10,
  OUTPUT_WIDTH = 10,
  OUTPUT_CHANNEL = 512,
  STRIDE_HEIGHT = 1,
  STRIDE_WIDTH = 1,
  DILATION_HEIGHT = 1,
  DILATION_WIDTH = 1,
  PADDING_HEIGHT = 0,
  PADDING_WIDTH = 0,
  INPUT_SIZE = BATCH * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL,
  FILTER_SIZE = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL * OUTPUT_CHANNEL,
  OUTPUT_SIZE = BATCH * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL,
  M = BATCH * OUTPUT_HEIGHT * OUTPUT_WIDTH,
  K = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL,
  IM2ROW_SIZE = M * K
};
#endif

#if defined(ENABLE_BENCHMARK)
#include "skl-test.h"
#endif

#if defined(ENABLE_TEST)
#include <math.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> // rand, aligned_alloc
#include <string.h>

#include "./conv2d_nhwc_f32.h"
#include "./conv2d_utils.h"

// Global test data arrays
__attribute__((aligned(ALIGN))) static float input[INPUT_SIZE];

__attribute__((aligned(ALIGN))) static float filter[FILTER_SIZE];

__attribute__((aligned(ALIGN))) static float output[OUTPUT_SIZE];

#if defined(ENABLE_TEST)
__attribute__((aligned(ALIGN))) static float ref_output[OUTPUT_SIZE];

static float get_random_f32(void) {
  const float min = 0;
  const float max = 1;
  float frac = (float)rand() / (float)RAND_MAX;
  return frac * (max - min) + min;
}
#else
/* To save simulation time, just use an increasing int value when tests are
 * disabled.
 */
static float get_random_f32(void) {
  static int64_t next = 0;
  return (float)next++;
}
#endif // ENABLE_TEST

static void init_random(float *arr, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = get_random_f32();
  }
}

#if defined(ENABLE_TEST)
static int check_error(const char *name, const float *res, const float *ref,
                       size_t len, float tolerance) {
  for (size_t i = 0; i < len; i++) {
    float err = fabsf(res[i] - ref[i]);
    if (err > tolerance) {
      printf("%15s: result [%zd] (%f) != reference (%f)\n", name, i, res[i],
             ref[i]);
      return 1;
    }
  }
  printf("%15s: pass\n", name);
  return 0;
}
#endif

int main(void) {
  int ret = 0; // return value
#if defined(ENABLE_BENCHMARK)
  uint64_t c0;
  uint64_t c1;
  uint64_t i0;
  uint64_t i1;
#endif

  printf("=== Conv2D GEMM Transformation Example ===\n");
  printf("This example demonstrates GEMM transformation for general "
         "convolutions.\n");

  printf(
      "Convolution: [%d, %d, %d, %d] x [%d, %d, %d, %d] -> [%d, %d, %d, %d]\n",
      BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, FILTER_HEIGHT,
      FILTER_WIDTH, INPUT_CHANNEL, OUTPUT_CHANNEL, BATCH, OUTPUT_HEIGHT,
      OUTPUT_WIDTH, OUTPUT_CHANNEL);

  init_random(input, INPUT_SIZE);
  init_random(filter, FILTER_SIZE);

#define CONV2D_GENERAL_ARGS                                                    \
  BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, FILTER_HEIGHT,              \
      FILTER_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,               \
      PADDING_WIDTH, PADDING_HEIGHT, STRIDE_WIDTH, STRIDE_HEIGHT,              \
      DILATION_WIDTH, DILATION_HEIGHT

#define DIRECT_CONV2D_KERNEL(FUNCTION, OUTPUT, ...)                            \
  FUNCTION(OUTPUT, input, filter, __VA_ARGS__);

#define IM2ROW_GEMM_CONV2D_KERNEL(FUNCTION, OUTPUT, ...)                       \
  FUNCTION(OUTPUT, im2row, input, filter, __VA_ARGS__);

#if defined(ENABLE_TEST)
  memset(ref_output, -1, sizeof ref_output);
  DIRECT_CONV2D_KERNEL(conv2d_io_nhwc_filter_hwio_f32_f32_f32_ref, ref_output,
                       CONV2D_GENERAL_ARGS)
#endif

#if defined(ENABLE_BENCHMARK)
#define MEASURE_PERF(RUN_SPEC, FUNCTION, NAME, ...)                            \
  /* Measure 2nd run after caches warmed */                                    \
  c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();                        \
  riscv_fence();                                                               \
  RUN_SPEC(FUNCTION, output, __VA_ARGS__);                                     \
  riscv_fence();                                                               \
  c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();                        \
  report_perf_epc(NAME, c1 - c0, i1 - i0, OUTPUT_SIZE);
#else
#define MEASURE_PERF(RUN_SPEC, FUNCTION, NAME, ...)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOLERANCE)                                \
  ret += check_error(NAME, output, ref_output, OUTPUT_SIZE, TOLERANCE);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOLERANCE)
#endif

#define RUN(RUN_SPEC, FUNCTION, NAME, TOLERANCE, ...)                          \
  memset(output, 0, sizeof output);                                            \
  RUN_SPEC(FUNCTION, output, __VA_ARGS__);                                     \
  MEASURE_PERF(RUN_SPEC, FUNCTION, NAME, __VA_ARGS__);                         \
  CHECK_RESULT(FUNCTION, NAME, TOLERANCE);

#if defined(__riscv_zve32f)

  if (is_conv2d_to_gemm_im2row_required(
          FILTER_HEIGHT, FILTER_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH,
          DILATION_HEIGHT, DILATION_WIDTH, PADDING_HEIGHT, PADDING_WIDTH)) {
    printf("\n=== Im2Row+GEMM Transformation Demonstration ===\n");
    printf("Transform: Conv2D with filter (%dx%d), stride (%d, %d), dilation "
           "(%d, %d), padding (%d, %d) -> Im2Row -> GEMM\n",
           FILTER_HEIGHT, FILTER_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH,
           DILATION_HEIGHT, DILATION_WIDTH, PADDING_HEIGHT, PADDING_WIDTH);
    printf("Matrix dimensions: M=%dx%dx%d=%d, N=%d, K=%dx%dx%d=%d\n", BATCH,
           OUTPUT_HEIGHT, OUTPUT_WIDTH, M, OUTPUT_CHANNEL, FILTER_HEIGHT,
           FILTER_WIDTH, INPUT_CHANNEL, K);

    float *im2row = aligned_alloc(ALIGN, IM2ROW_SIZE * sizeof(float));

    // Demonstrate Im2Row+GEMM transformation for general convolution
    RUN(IM2ROW_GEMM_CONV2D_KERNEL,
        conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390,
        "conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390", 0.f,
        CONV2D_GENERAL_ARGS);

    free(im2row);
  } else {
    printf("\n=== Direct GEMM Transformation Demonstration ===\n");
    printf("Transform: Conv2D -> GEMM\n");
    printf("Matrix dimensions: M=%dx%dx%d=%d, N=%d, K=%dx%dx%d=%d\n", BATCH,
           OUTPUT_HEIGHT, OUTPUT_WIDTH, M, OUTPUT_CHANNEL, FILTER_HEIGHT,
           FILTER_WIDTH, INPUT_CHANNEL, K);

    // Demonstrate Direct GEMM transformation for general convolution
    RUN(DIRECT_CONV2D_KERNEL, conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390,
        "conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390", 0.f,
        CONV2D_GENERAL_ARGS);
  }

#else
  printf(
      "RISC-V Vector extension not available. Skipping RVV implementations.\n");
#endif

  return ret > 0;
}
