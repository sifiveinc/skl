// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfofp8min) || !defined(__riscv_zvfbfmin)
#error This file requires the Zvfofp8min and Zvfbfmin extensions
#endif

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !(defined(RUN_BF16_E4M3) || defined(RUN_BF16_E5M2))
#error No tests or benchmarks enabled!
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 1024 // Default input length
#endif

#if !defined(SCALING_TEST_MODE)
#define SCALING_TEST_MODE 3 // 1: no scaling, 2: scaling, 3: both
#endif

#if !defined(SATURATION_TEST_MODE)
#define SATURATION_TEST_MODE 3 // 1: no saturation, 2: saturation, 3: both
#endif

#if defined(ENABLE_TEST)
#include "skl-common.h"
#endif
#include "skl-test.h"

#include "skl.h"

#if defined(ENABLE_TEST)
#include <math.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 4096 };

__attribute__((aligned(ALIGN))) __bf16 in_bf16[NUM_ELEMS];
__attribute__((aligned(ALIGN))) uint8_t out_ofp8[NUM_ELEMS];

#if defined(ENABLE_TEST)
__attribute__((aligned(ALIGN))) uint8_t ref_ofp8[NUM_ELEMS];
#endif

static float scaling_factor = 1.0f;

#if defined(ENABLE_TEST)
static void golden_cvt_bf16_f8e4m3_scale(uint8_t *pDst, const __bf16 *pSrc,
                                         float scaling_factor, size_t n) {

  for (size_t i = 0; i < n; i++) {
    float in = (float)pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e4m3(in, false);
  }
}
static void golden_cvt_sat_bf16_f8e4m3_scale(uint8_t *pDst, const __bf16 *pSrc,
                                             float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float in = (float)pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e4m3(in, true);
  }
}

__attribute__((unused)) static void golden_cvt_bf16_f8e4m3(uint8_t *pDst,
                                                           const __bf16 *pSrc,
                                                           float scaling_factor,
                                                           size_t n) {
  golden_cvt_bf16_f8e4m3_scale(pDst, pSrc, scaling_factor, n);
}

__attribute__((unused)) static void
golden_cvt_sat_bf16_f8e4m3(uint8_t *pDst, const __bf16 *pSrc,
                           float scaling_factor, size_t n) {
  golden_cvt_sat_bf16_f8e4m3_scale(pDst, pSrc, scaling_factor, n);
}

static void golden_cvt_bf16_f8e5m2_scale(uint8_t *pDst, const __bf16 *pSrc,
                                         float scaling_factor, size_t n) {

  for (size_t i = 0; i < n; i++) {
    float in = (float)pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e5m2(in, false);
  }
}
static void golden_cvt_sat_bf16_f8e5m2_scale(uint8_t *pDst, const __bf16 *pSrc,
                                             float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float in = (float)pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e5m2(in, true);
  }
}

__attribute__((unused)) static void golden_cvt_bf16_f8e5m2(uint8_t *pDst,
                                                           const __bf16 *pSrc,
                                                           float scaling_factor,
                                                           size_t n) {
  golden_cvt_bf16_f8e5m2_scale(pDst, pSrc, scaling_factor, n);
}

__attribute__((unused)) static void
golden_cvt_sat_bf16_f8e5m2(uint8_t *pDst, const __bf16 *pSrc,
                           float scaling_factor, size_t n) {
  golden_cvt_sat_bf16_f8e5m2_scale(pDst, pSrc, scaling_factor, n);
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
#define GENERATE_GOLDEN_NCVT(NAME, IN_TYPE, OUT_TYPE)                          \
  golden_##NAME(ref_##OUT_TYPE, in_##IN_TYPE, scaling_factor, NUM_ELEMS);
#else
#define GENERATE_GOLDEN_NCVT(NAME, IN_TYPE, OUT_TYPE)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE)                                  \
  ret += check_error_##OUT_TYPE(#NAME, out_##OUT_TYPE, ref_##OUT_TYPE);
#else
#define CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE)
#endif

#define RUN_NCVT(FUNCTION, NAME, IN_TYPE, OUT_TYPE)                            \
  memset(out_##OUT_TYPE, 0, NUM_ELEMS * sizeof(*out_##OUT_TYPE));              \
  SKL_BENCHMARK_RUN(#NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION,               \
                    out_##OUT_TYPE, in_##IN_TYPE, scaling_factor, NUM_ELEMS);  \
  GENERATE_GOLDEN_NCVT(NAME, IN_TYPE, OUT_TYPE);                               \
  CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE);

#if defined(RUN_BF16_E4M3)
// Helper function to run BF16 to E4M3 conversion tests
static int run_bf16_f8e4m3_tests(void) {
  int ret = EXIT_SUCCESS;
  if (((unsigned)SATURATION_TEST_MODE & 1U) == 1U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
      RUN_NCVT(skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin, cvt_bf16_f8e4m3, bf16,
               ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin, cvt_bf16_f8e4m3_scale,
               bf16, ofp8);
    }
  }
  if (((unsigned)SATURATION_TEST_MODE & 2U) == 2U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
      RUN_NCVT(skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin, cvt_sat_bf16_f8e4m3,
               bf16, ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin,
               cvt_sat_bf16_f8e4m3_scale, bf16, ofp8);
    }
  }
  return ret;
}
#endif

#if defined(RUN_BF16_E5M2)
// Helper function to run BF16 to E5M2 conversion tests
static int run_bf16_f8e5m2_tests(void) {
  int ret = EXIT_SUCCESS;
  if (((unsigned)SATURATION_TEST_MODE & 1U) == 1U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
      RUN_NCVT(skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin, cvt_bf16_f8e5m2, bf16,
               ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin, cvt_bf16_f8e5m2_scale,
               bf16, ofp8);
    }
  }
  if (((unsigned)SATURATION_TEST_MODE & 2U) == 2U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
      RUN_NCVT(skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin, cvt_sat_bf16_f8e5m2,
               bf16, ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin,
               cvt_sat_bf16_f8e5m2_scale, bf16, ofp8);
    }
  }
  return ret;
}
#endif

int main(void) {

#if !defined(__riscv_zvfofp8min)
#error The test/benchmark requires the Zvfofp8min extension
#endif

  int ret = EXIT_SUCCESS;
  PRINT_TEST_NAME(SKL_TEST_NAME);

  skl_test_init_bf16(in_bf16, NUM_ELEMS, SKL_TEST_MIN_BF16,
                     SKL_TEST_MAX_BF16); // input data
#if defined(ENABLE_TEST)
  // Make sure we test special values
  in_bf16[0] = (__bf16)nanf("");
  in_bf16[1] = (__bf16)INFINITY;
  in_bf16[2] = (__bf16)-INFINITY;
#endif

#if defined(RUN_BF16_E4M3)
  ret += run_bf16_f8e4m3_tests();
#endif

#if defined(RUN_BF16_E5M2)
  ret += run_bf16_f8e5m2_tests();
#endif

  return ret > 0;
}
