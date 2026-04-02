// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfofp4min)
#error This file requires the Zvfofp4min extension
#endif

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 1024 // Default input length
#endif

// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 4096 };

__attribute__((aligned(ALIGN))) uint8_t in_ofp4x2[(NUM_ELEMS + 1) / 2];
__attribute__((aligned(ALIGN))) uint8_t out_ofp8[NUM_ELEMS];

#if defined(ENABLE_TEST)
__attribute__((aligned(ALIGN))) uint8_t ref_ofp8[NUM_ELEMS];

static void cvt_ofp4x2_f8e4m3(uint8_t in, uint8_t *out0, uint8_t *out1) {

  uint8_t in_0 = in & 0xFU;
  uint8_t in_1 = in >> 4U;
  *out0 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_0), false);
  *out1 = skl_cvt_f32_f8e4m3(skl_cvt_f4e2m1_f32(in_1), false);
}

static void golden_cvt_f4e2m1_f8e4m3(uint8_t *pDst, const uint8_t *pSrc,
                                     size_t n) {
  for (size_t i = 0; i < n / 2 * 2; i += 2) {
    cvt_ofp4x2_f8e4m3(pSrc[i / 2], pDst + i, pDst + i + 1);
  }
  if (n % 2 == 1) {
    uint8_t tmp;
    cvt_ofp4x2_f8e4m3(pSrc[n / 2], pDst + n - 1, &tmp);
  }
}

static int check_error_ofp8(const char *name, const uint8_t *res,
                            const uint8_t *ref) {
  for (size_t i = 0; i < NUM_ELEMS; i++) {
    if (res[i] != ref[i]) {
      printf("%-25s : result[%zu]: %x != ref[%zu]: %x\n", name, i, res[i], i,
             ref[i]);

      printf(" res ");
      print_float(skl_cvt_f8e4m3_f32(res[i]));
      printf(", ref ");
      print_float(skl_cvt_f8e4m3_f32(ref[i]));
      printf("\n");

      return 1;
    }
  }
  printf("  test: %-25s pass!\n", name);
  return 0;
}
#endif

#define TEST_LABEL(S) #S ":\n"
#define PRINT_TEST_NAME(S) printf(TEST_LABEL(S));

#if defined(ENABLE_TEST)
#define GENERATE_GOLDEN_WCVT(NAME, IN_TYPE, OUT_TYPE)                          \
  golden_##NAME(ref_##OUT_TYPE, in_##IN_TYPE, NUM_ELEMS);
#else
#define GENERATE_GOLDEN_WCVT(NAME, IN_TYPE, OUT_TYPE)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE)                                  \
  ret += check_error_##OUT_TYPE(#NAME, out_##OUT_TYPE, ref_##OUT_TYPE);
#else
#define CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE)
#endif

#define RUN_WCVT(FUNCTION, NAME, IN_TYPE, OUT_TYPE)                            \
  memset(out_##OUT_TYPE, 0, NUM_ELEMS * sizeof(*out_##OUT_TYPE));              \
  SKL_BENCHMARK_RUN(#NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION,               \
                    out_##OUT_TYPE, in_##IN_TYPE, NUM_ELEMS);                  \
  GENERATE_GOLDEN_WCVT(NAME, IN_TYPE, OUT_TYPE);                               \
  CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE);

int main(void) {

#if !defined(__riscv_zvfofp4min)
#error The test/benchmark requires the Zvfofp4min extension
#endif

  int ret = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);

  skl_test_init_i8((int8_t *)in_ofp4x2, (NUM_ELEMS + 1) / 2, SKL_TEST_MIN_I8,
                   SKL_TEST_MAX_I8); // input data

  RUN_WCVT(skl_cvt_f4e2m1_f8e4m3_zvfofp4min, cvt_f4e2m1_f8e4m3, ofp4x2, ofp8);

  return ret > 0;
}
