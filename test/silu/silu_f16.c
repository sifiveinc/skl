// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

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
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(RUN_ALL)
#define RUN_ZVFH 1
#define RUN_SCALAR 1
#define RUN_XSFVFEXP16E 1
#define RUN_XSFVFEXPA 1
#endif

#if defined(ENABLE_TEST)
static inline void reference_silu_f16(_Float16 *out, const _Float16 *in,
                                      size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = (_Float16)(in[i] / (1 + exp(-(double)in[i])));
  }
}
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) _Float16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) int16_t output_bits[NUM_ELEMS];
__attribute__((aligned(ALIGN))) int16_t ref_output_bits[NUM_ELEMS];

static void init_random(_Float16 *arr, size_t len) {
#if defined(ENABLE_TEST)
  /* If testing, fill in an array with random floats in [-21, 9] */
  const float min = -21;
  const float max = 9;
  float frac;
  for (size_t i = 0; i < len; i++) {
    if (i == 0) {
      arr[i] = __builtin_nanf16("");
    } else {
      frac = (float)rand() / (float)RAND_MAX;
      arr[i] = (_Float16)(frac * (max - min) + min);
    }
  }
#else
  /* Otherwise, save simulation time using increasing ints */
  int32_t next = 0;
  for (size_t i = 0; i < len; i++) {
    arr[i] = (_Float16)(next++);
  }
#endif
}

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element silu:\n", NUM_ELEMS);

  _Float16 *output = (_Float16 *)output_bits;
  init_random(input, NUM_ELEMS); // input data
#if defined(ENABLE_TEST)
  _Float16 *ref_output = (_Float16 *)ref_output_bits;
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  reference_silu_f16(ref_output, input, NUM_ELEMS);
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

#if defined(__riscv_zvfh) && defined(RUN_ZVFH)
  RUN(skl_silu_9u_f16_zvfh, "zvfh", 9);
#endif

#if defined(__riscv_xsfvfexp16e) && defined(RUN_XSFVFEXP16E)
  RUN(skl_silu_31u_f16_xsfvfexp16e, "xsfvfexp16e", 31);
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh) &&                     \
    defined(RUN_XSFVFEXPA)
  RUN(skl_silu_9u_f16_xsfvfexpa_zvfh, "xsfvfexpa_zvfh", 9);
#endif

#if defined(RUN_SCALAR)
  RUN(skl_silu_1u_f16_ref, "reference", 1)
#endif

#if !(defined(RUN_RVV) || defined(RUN_SCALAR) || defined(RUN_XSFVFEXP16E) ||   \
      defined(RUN_XSFVFEXPA))
#error No tests or benchmarks enabled!
#endif

  return ret;
}
