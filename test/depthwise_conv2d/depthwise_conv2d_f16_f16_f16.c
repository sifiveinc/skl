#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

enum {
  INPUT_HEIGHT = 16,
  INPUT_WIDTH = 16,
  INPUT_CHANNEL = 512,
  FILTER_HEIGHT = 3,
  FILTER_WIDTH = 3,
  OUTPUT_HEIGHT = 14,
  OUTPUT_WIDTH = 14,
  OUTPUT_CHANNEL = 512,
  STRIDE_HEIGHT = 1,
  STRIDE_WIDTH = 1,
  DILATION_HEIGHT_FACTOR = 1,
  DILATION_WIDTH_FACTOR = 1,
  DEPTH_MULTIPLIER = 1
};

#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"

#if defined(ENABLE_TEST)
#include <math.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__riscv_zvfh) || !defined(__riscv_zfh)
#error This file requires the Zvfh and Zfh extension
#endif

enum { ALIGN = 64 };

__attribute__((
    aligned(ALIGN))) _Float16 input[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL];
__attribute__((aligned(ALIGN))) _Float16
    filter[FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL * DEPTH_MULTIPLIER];
__attribute__((aligned(
    ALIGN))) _Float16 output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL];
__attribute__((aligned(
    ALIGN))) _Float16 ref_output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL];

#if defined(ENABLE_TEST)
static int fp_eq(float result, float golden, float relative_error) {
  if ((isnan(result) && isnan(golden)) || (isinf(result) && isinf(golden))) {
    return 1;
  }
  // if near zero, do absolute error instead.
  float abs_error =
      relative_error *
      ((fabsf(result) > relative_error) ? fabsf(result) : relative_error);
  return (fabsf(golden - result) <= abs_error);
}

static int check_error(const char *name, const _Float16 *res,
                       const _Float16 *ref, size_t len,
                       _Float16 relative_error) {
  for (size_t i = 0; i < len; i++) {
    if (!fp_eq((float)res[i], (float)ref[i], (float)relative_error)) {
      printf("%15s: error at index %zu: %f != %f\n", name, i, (float)res[i],
             (float)ref[i]);
      return 1;
    }
  }
  return 0;
}
#endif

int main(void) {
  int ret = 0; // return value
  printf("Measuring depthwise_conv2d: [%d, %d, %d] x [%d, %d, %d, %d] -> "
         "[%d, %d, %d]\n",
         INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH,
         INPUT_CHANNEL, DEPTH_MULTIPLIER, OUTPUT_HEIGHT, OUTPUT_WIDTH,
         OUTPUT_CHANNEL);

  skl_test_init_f16(
      input, (size_t)INPUT_HEIGHT * (size_t)INPUT_WIDTH * (size_t)INPUT_CHANNEL,
      SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);
  skl_test_init_f16(filter,
                    (size_t)FILTER_HEIGHT * (size_t)FILTER_WIDTH *
                        (size_t)INPUT_CHANNEL * (size_t)DEPTH_MULTIPLIER,
                    SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);

  const size_t input_row_stride = (size_t)INPUT_WIDTH * (size_t)INPUT_CHANNEL;
  const size_t input_col_stride = (size_t)INPUT_CHANNEL;
  const size_t filter_row_stride =
      (size_t)FILTER_WIDTH * (size_t)INPUT_CHANNEL * (size_t)DEPTH_MULTIPLIER;
  const size_t filter_col_stride =
      (size_t)INPUT_CHANNEL * (size_t)DEPTH_MULTIPLIER;
  const size_t output_row_stride =
      (size_t)OUTPUT_WIDTH * (size_t)OUTPUT_CHANNEL;
  const size_t output_col_stride = (size_t)OUTPUT_CHANNEL;

#define DWCONV2D_KERNEL(FUNCTION, OUTPUT, ...)                                 \
  FUNCTION(OUTPUT, input, filter, __VA_ARGS__);

#define DWCONV2D_GENERAL_ARGS                                                  \
  INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH,       \
      OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL, DEPTH_MULTIPLIER,           \
      STRIDE_HEIGHT, STRIDE_WIDTH, DILATION_HEIGHT_FACTOR,                     \
      DILATION_WIDTH_FACTOR, input_row_stride, input_col_stride,               \
      filter_row_stride, filter_col_stride, output_row_stride,                 \
      output_col_stride

#define DWCONV2D_3X3_ARGS                                                      \
  INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,       \
      OUTPUT_CHANNEL, DEPTH_MULTIPLIER, STRIDE_HEIGHT, STRIDE_WIDTH,           \
      DILATION_HEIGHT_FACTOR, DILATION_WIDTH_FACTOR, input_row_stride,         \
      input_col_stride, filter_row_stride, filter_col_stride,                  \
      output_row_stride, output_col_stride

#if defined(ENABLE_TEST)
  memset(ref_output, -1, sizeof ref_output);
  DWCONV2D_KERNEL(skl_depthwise_conv2d_hwc_f16_f16_f16_scalar, ref_output,
                  DWCONV2D_GENERAL_ARGS);
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME)                                           \
  ret += check_error(NAME, output, ref_output,                                 \
                     (size_t)OUTPUT_HEIGHT * (size_t)OUTPUT_WIDTH *            \
                         (size_t)OUTPUT_CHANNEL,                               \
                     1e-3);
#else
#define CHECK_RESULT(FUNCTION, NAME)
#endif

#define RUN(FUNCTION, NAME, ...)                                               \
  memset(output, 0, sizeof output);                                            \
  SKL_BENCHMARK_RUN(                                                           \
      NAME,                                                                    \
      (size_t)OUTPUT_HEIGHT *(size_t)OUTPUT_WIDTH *(size_t)OUTPUT_CHANNEL,     \
      SKL_TEST_WARMUP, FUNCTION, output, input, filter, __VA_ARGS__);          \
  CHECK_RESULT(FUNCTION, NAME);

#if defined(__riscv_zvfh)
  RUN(skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f16_f16_f16_zvfh,
      "skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f16_f16_f16_zvfh",
      DWCONV2D_GENERAL_ARGS);

  RUN(skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f16_f16_f16_zvfh,
      "skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f16_f16_f16_zvfh",
      DWCONV2D_3X3_ARGS);
#endif

  return ret > 0;
}
