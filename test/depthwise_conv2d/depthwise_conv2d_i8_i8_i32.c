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

// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 64 };

__attribute__((
    aligned(ALIGN))) int8_t input[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNEL];
__attribute__((aligned(ALIGN))) int8_t
    filter[FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNEL * DEPTH_MULTIPLIER];
__attribute__((aligned(
    ALIGN))) int32_t output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL];
__attribute__((aligned(
    ALIGN))) int32_t ref_output[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL];

#if defined(ENABLE_TEST)
static int check_error(const char *name, const int32_t *res, const int32_t *ref,
                       size_t len) {
  uint32_t max = 0;
  for (size_t i = 0; i < len; i++) {
    uint32_t err = abs(res[i] - ref[i]);
    if (err > max) {
      max = err;
    }
  }
  printf("%15s: maximum error %" PRIu32 "\n", name, max);
  return max > 1;
}
#endif

int main(void) {
  int ret = 0; // return value
  printf("Measuring depthwise_conv2d: [%d, %d, %d] x [%d, %d, %d, %d] -> "
         "[%d, %d, %d]\n",
         INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH,
         INPUT_CHANNEL, DEPTH_MULTIPLIER, OUTPUT_HEIGHT, OUTPUT_WIDTH,
         OUTPUT_CHANNEL);

  skl_test_init_i8(
      input, (size_t)INPUT_HEIGHT * (size_t)INPUT_WIDTH * (size_t)INPUT_CHANNEL,
      SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(filter,
                   (size_t)FILTER_HEIGHT * (size_t)FILTER_WIDTH *
                       (size_t)INPUT_CHANNEL * (size_t)DEPTH_MULTIPLIER,
                   SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);

  const int32_t input_zero_point = 0;
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
      output_col_stride, input_zero_point

#define DWCONV2D_3X3_M1_ARGS                                                   \
  INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,       \
      OUTPUT_CHANNEL, STRIDE_HEIGHT, STRIDE_WIDTH, DILATION_HEIGHT_FACTOR,     \
      DILATION_WIDTH_FACTOR, input_row_stride, input_col_stride,               \
      filter_row_stride, filter_col_stride, output_row_stride,                 \
      output_col_stride, input_zero_point

#if defined(ENABLE_TEST)
  memset(ref_output, -1, sizeof ref_output);
  DWCONV2D_KERNEL(skl_depthwise_conv2d_hwc_i8_i8_i32_scalar, ref_output,
                  DWCONV2D_GENERAL_ARGS);
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME)                                           \
  ret += check_error(NAME, output, ref_output,                                 \
                     (size_t)OUTPUT_HEIGHT * (size_t)OUTPUT_WIDTH *            \
                         (size_t)OUTPUT_CHANNEL);
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

#if defined(__riscv_zve32x)
  RUN(skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_i8_i8_i32_zve32x,
      "skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_i8_i8_i32_zve32x",
      DWCONV2D_GENERAL_ARGS);

  RUN(skl_depthwise_conv2d_vc_f3x3_sn_dn_m1_in_hwc_i8_i8_i32_zve32x,
      "skl_depthwise_conv2d_vc_f3x3_sn_dn_m1_in_hwc_i8_i8_i32_zve32x",
      DWCONV2D_3X3_M1_ARGS);
#endif

  return ret > 0;
}
