// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !(defined(RUN_F32_E4M3) || defined(RUN_F32_E5M2) ||                        \
      defined(RUN_E4M3_BF16) || defined(RUN_E5M2_BF16))
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

// Use the range [-512, 512] for testing this extension
// NOTE: must define before including skl-test.h
#if defined(SKL_TEST_RAND_MIN_F32) && defined(SKL_TEST_RAND_MAX_F32)
#error SKL_TEST_RAND_MIN_F32 and SKL_TEST_RAND_MAX_F32 already defined
#endif
#define SKL_TEST_RAND_MIN_F32 (-512.0f)
#define SKL_TEST_RAND_MAX_F32 (512.0f)

#include "skl-test.h"
#include "skl.h"

#include <math.h> // NOLINT(misc-include-cleaner)
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 4096 };

#if defined(RUN_F32_E4M3) || defined(RUN_F32_E5M2)
__attribute__((aligned(ALIGN))) float in_f32[NUM_ELEMS];
__attribute__((aligned(ALIGN))) uint8_t out_ofp8[NUM_ELEMS];
#if defined(ENABLE_TEST)
__attribute__((aligned(ALIGN))) uint8_t ref_ofp8[NUM_ELEMS];
#endif
#endif

#if defined(RUN_E4M3_BF16) || defined(RUN_E5M2_BF16)
__attribute__((aligned(ALIGN))) uint8_t in_ofp8[NUM_ELEMS];
__attribute__((aligned(ALIGN))) __bf16 out_bf16[NUM_ELEMS];
#if defined(ENABLE_TEST)
__attribute__((aligned(ALIGN))) __bf16 ref_bf16[NUM_ELEMS];
#endif
#endif

__attribute__((unused)) static float scaling_factor = 1.0f;

#if defined(ENABLE_TEST)

#if defined(RUN_F32_E4M3)
__attribute__((unused)) static void
golden_cvt_f32_f8e4m3_scale(uint8_t *pDst, const float *pSrc,
                            float scaling_factor, size_t n) {

  for (size_t i = 0; i < n; i++) {
    float in = pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e4m3(in, false);
  }
}

__attribute__((unused)) static void
golden_cvt_sat_f32_f8e4m3_scale(uint8_t *pDst, const float *pSrc,
                                float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float in = pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e4m3(in, true);
  }
}
#endif

#if defined(RUN_F32_E5M2)
__attribute__((unused)) static void
golden_cvt_f32_f8e5m2_scale(uint8_t *pDst, const float *pSrc,
                            float scaling_factor, size_t n) {

  for (size_t i = 0; i < n; i++) {
    float in = pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e5m2(in, false);
  }
}

__attribute__((unused)) static void
golden_cvt_sat_f32_f8e5m2_scale(uint8_t *pDst, const float *pSrc,
                                float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float in = pSrc[i] * scaling_factor;
    pDst[i] = skl_cvt_f32_f8e5m2(in, true);
  }
}
#endif

#if defined(RUN_E4M3_BF16)
__attribute__((unused)) static void
golden_cvt_f8e4m3_bf16(__bf16 *pDst, const uint8_t *pSrc, size_t n) {
  for (size_t i = 0; i < n; i++) {
    pDst[i] = (__bf16)skl_cvt_f8e4m3_f32(pSrc[i]);
  }
}
#endif

#if defined(RUN_E5M2_BF16)
__attribute__((unused)) static void
golden_cvt_f8e5m2_bf16(__bf16 *pDst, const uint8_t *pSrc, size_t n) {
  for (size_t i = 0; i < n; i++) {
    pDst[i] = (__bf16)skl_cvt_f8e5m2_f32(pSrc[i]);
  }
}
#endif

#if defined(RUN_F32_E4M3) || defined(RUN_F32_E5M2)
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

#if defined(RUN_E4M3_BF16) || defined(RUN_E5M2_BF16)
__attribute__((unused)) static int
check_error_bf16(const char *name, const __bf16 *res, const __bf16 *ref) {
  for (size_t i = 0; i < NUM_ELEMS; i++) {

    uint16_t out_bits;
    uint16_t ref_bits;
    memcpy(&out_bits, &res[i], sizeof(uint16_t));
    memcpy(&ref_bits, &ref[i], sizeof(uint16_t));

    if (out_bits != ref_bits) {
      printf("%-25s : result[%zu]: ", name, i);
      print_float((float)res[i]);
      printf(" (0x%x) != ref[%zu]: ", out_bits, i);
      print_float((float)ref[i]);
      printf(" (0x%x)\n", ref_bits);
      return 1;
    }
  }
  printf("  test: %-25s pass!\n", name);
  return 0;
}
#endif
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

#define RUN_NCVT(FUNCTION, NAME, IN_TYPE, OUT_TYPE)                            \
  memset(out_##OUT_TYPE, 0, NUM_ELEMS * sizeof(*out_##OUT_TYPE));              \
  SKL_BENCHMARK_RUN(#NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION,               \
                    out_##OUT_TYPE, in_##IN_TYPE, scaling_factor, NUM_ELEMS);  \
  GENERATE_GOLDEN_NCVT(NAME, IN_TYPE, OUT_TYPE);                               \
  CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE);

#define RUN_WCVT(FUNCTION, NAME, IN_TYPE, OUT_TYPE)                            \
  memset(out_##OUT_TYPE, 0, NUM_ELEMS * sizeof(*out_##OUT_TYPE));              \
  SKL_BENCHMARK_RUN(#NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION,               \
                    out_##OUT_TYPE, in_##IN_TYPE, NUM_ELEMS);                  \
  GENERATE_GOLDEN_WCVT(NAME, IN_TYPE, OUT_TYPE);                               \
  CHECK_RESULT(NAME, IN_TYPE, OUT_TYPE);

#if defined(RUN_F32_E4M3)
// Helper function to run F32 to E4M3 conversion tests
static int run_f32_f8e4m3_tests(void) {
  int ret = EXIT_SUCCESS;
  if (((unsigned)SATURATION_TEST_MODE & 1U) == 1U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
#define golden_cvt_f32_f8e4m3 golden_cvt_f32_f8e4m3_scale
      RUN_NCVT(skl_cvt_f32_f8e4m3_zvfofp8min, cvt_f32_f8e4m3, f32, ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_f32_f8e4m3_zvfofp8min, cvt_f32_f8e4m3_scale, f32, ofp8);
    }
  }
  if (((unsigned)SATURATION_TEST_MODE & 2U) == 2U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
#define golden_cvt_sat_f32_f8e4m3 golden_cvt_sat_f32_f8e4m3_scale
      RUN_NCVT(skl_cvt_sat_f32_f8e4m3_zvfofp8min, cvt_sat_f32_f8e4m3, f32,
               ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_sat_f32_f8e4m3_zvfofp8min, cvt_sat_f32_f8e4m3_scale, f32,
               ofp8);
    }
  }
  return ret;
}
#endif

#if defined(RUN_F32_E5M2)
// Helper function to run F32 to E5M2 conversion tests
static int run_f32_f8e5m2_tests(void) {
  int ret = EXIT_SUCCESS;
  if (((unsigned)SATURATION_TEST_MODE & 1U) == 1U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
#define golden_cvt_f32_f8e5m2 golden_cvt_f32_f8e5m2_scale
      RUN_NCVT(skl_cvt_f32_f8e5m2_zvfofp8min, cvt_f32_f8e5m2, f32, ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_f32_f8e5m2_zvfofp8min, cvt_f32_f8e5m2_scale, f32, ofp8);
    }
  }
  if (((unsigned)SATURATION_TEST_MODE & 2U) == 2U) {
    if (((unsigned)SCALING_TEST_MODE & 1U) == 1U) {
      scaling_factor = 1.0f;
#define golden_cvt_sat_f32_f8e5m2 golden_cvt_sat_f32_f8e5m2_scale
      RUN_NCVT(skl_cvt_sat_f32_f8e5m2_zvfofp8min, cvt_sat_f32_f8e5m2, f32,
               ofp8);
    }
    if (((unsigned)SCALING_TEST_MODE & 2U) == 2U) {
      scaling_factor = 1.234f;
      RUN_NCVT(skl_cvt_sat_f32_f8e5m2_zvfofp8min, cvt_sat_f32_f8e5m2_scale, f32,
               ofp8);
    }
  }
  return ret;
}
#endif

int main(void) {

  int ret = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);

#if defined(RUN_F32_E4M3) || defined(RUN_F32_E5M2)
  skl_test_init_f32(in_f32, NUM_ELEMS, SKL_TEST_MIN_F32,
                    SKL_TEST_MAX_F32); // input data
  // Also set a few special values for testing
  in_f32[0] = nanf("");
  in_f32[1] = INFINITY;
  in_f32[2] = -INFINITY;
#endif
#if defined(RUN_E4M3_BF16) || defined(RUN_E5M2_BF16)
  skl_test_init_i8((int8_t *)in_ofp8, NUM_ELEMS, SKL_TEST_MIN_I8,
                   SKL_TEST_MAX_I8); // input data
#endif

#if defined(RUN_F32_E4M3)
  ret += run_f32_f8e4m3_tests();
#endif

#if defined(RUN_F32_E5M2)
  ret += run_f32_f8e5m2_tests();
#endif

#if defined(RUN_E4M3_BF16)
  RUN_WCVT(skl_cvt_f8e4m3_bf16_zvfofp8min, cvt_f8e4m3_bf16, ofp8, bf16);
#endif

#if defined(RUN_E5M2_BF16)
  RUN_WCVT(skl_cvt_f8e5m2_bf16_zvfofp8min, cvt_f8e5m2_bf16, ofp8, bf16);
#endif

  return ret > 0;
}
