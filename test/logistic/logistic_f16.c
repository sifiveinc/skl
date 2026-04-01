// Copyright 2026 SiFive, Inc.
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
#define RUN_XSFVFEXP16E 1
#endif

#if defined(ENABLE_TEST)
static inline void scalar_logistic_f16(_Float16 *out, const _Float16 *in,
                                       size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float x = in[i];
    if (!isnan(x)) {
      if (x <= 0.0f) {
        float ex = expf(x);
        out[i] = (_Float16)(ex / (1.0f + ex));
      } else {
        /* For x > 0, compute logistic(x) = 1 - logistic(-x) */
        float ex = expf(-x);
        out[i] = (_Float16)(1.0f - ex / (1.0f + ex));
      }
    } else {
      out[i] = (_Float16)(x + x); // propagate NaN
    }
  }
}
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) _Float16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 ref_output[NUM_ELEMS];

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element logistic:\n", NUM_ELEMS);
  skl_test_init_f16(input, NUM_ELEMS, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_logistic_f16(ref_output, input,
                      NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += skl_check_error_ulp_f16(NAME, output, ref_output, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    NUM_ELEMS);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zvfh) && defined(RUN_RVV)
  RUN(skl_logistic_3u_f16_zvfh, "zvfh", 3);
#endif

#if defined(__riscv_xsfvfexp16e) && defined(RUN_XSFVFEXP16E)
  RUN(skl_logistic_5u_f16_xsfvfexp16e, "xsfvfexp16e", 5);
#endif

#if defined(RUN_SCALAR)
  RUN(skl_logistic_1u_f16_scalar, "scalar", 1)
#endif

#if !(defined(RUN_RVV) || defined(RUN_XSFVFEXP16E) || defined(RUN_SCALAR))
#error No tests or benchmarks enabled!
#endif

  return ret;
}
