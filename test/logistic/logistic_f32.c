// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 1024
#endif

#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#if defined(ENABLE_TEST)
#include <math.h>
#include <stdlib.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(RUN_ALL)
#define RUN_SCALAR 1
#define RUN_RVV 1
#define RUN_XSFVFEXPA 1
#define RUN_XSFVFEXP32E 1
#endif

#if defined(ENABLE_TEST)
static inline void scalar_logistic_f32(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = (float)(1 / (1 + exp(-(double)in[i])));
  }
}
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) float input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float ref_output[NUM_ELEMS];

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element logistic:\n", NUM_ELEMS);
  skl_test_init_f32(input, NUM_ELEMS, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_logistic_f32(ref_output, input,
                      NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += skl_check_error_ulp_f32(NAME, output, ref_output, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    NUM_ELEMS);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zve32f) && defined(RUN_RVV)
  RUN(skl_logistic_3u_f32_zve32f, "rvv", 3);
#endif

#if defined(__riscv_xsfvfexpa) && defined(RUN_XSFVFEXPA)
  RUN(skl_logistic_4u_f32_xsfvfexpa, "xsfvfexpa", 4);
#endif

#if defined(__riscv_xsfvfexp32e) && defined(RUN_XSFVFEXP32E)
  RUN(skl_logistic_5u_f32_xsfvfexp32e, "xsfvfexp32e", 5);
#endif

#if defined(RUN_SCALAR)
  RUN(skl_logistic_3u_f32_ref, "reference", 3)
#endif

#if !(defined(RUN_RVV) || defined(RUN_SCALAR) || defined(RUN_XSFVFEXPA) ||     \
      defined(RUN_XSFVFEXP32e))
#error No tests or benchmarks enabled!
#endif

  return ret;
}
