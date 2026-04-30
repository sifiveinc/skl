// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK.
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 2048 // Input length
#endif

#if !defined(BETA)
#define BETA 1.0 // Exponential scaling factor
#endif

#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#if defined(ENABLE_TEST)
#include <math.h>
#include <stdlib.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(RUN_ALL)
#define RUN_SCALAR 1
#define RUN_ZVE32F 1
#define RUN_BFMIN 1
#define RUN_VFBFA 1
#define RUN_VFEXPA_BFMIN 1
#define RUN_VFEXPA_VFBFA 1
#define RUN_VFEXP_BFMIN 1
#define RUN_VFEXP_VFBFA 1
#define RUN_VFEXP32E_BFMIN 1
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) __bf16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) __bf16 output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) __bf16 ref_output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) double workspace[NUM_ELEMS];

#if defined(ENABLE_TEST)
/** Scalar softmax using FP64 intermediates for high accuracy */
static void reference_softmax_bf16(__bf16 *out, const __bf16 *in, __bf16 beta,
                                   double *workspace, size_t n) {
  if (n < 1)
    return;

  float max = in[0];
  for (size_t i = 1; i < n; i++) {
    max = fmaxf(in[i], max);
  }

  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    workspace[i] = exp(beta * ((double)in[i] - max));
    sum += workspace[i];
  }

  for (size_t i = 0; i < n; i++) {
    out[i] = (__bf16)(workspace[i] / sum);
  }
}
#endif // defined(ENABLE_TEST)

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element softmax:\n\n", NUM_ELEMS);
  skl_test_init_bf16(input, NUM_ELEMS, SKL_TEST_MIN_BF16, SKL_TEST_MAX_BF16);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  reference_softmax_bf16(ref_output, input, BETA, workspace, NUM_ELEMS);
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += skl_check_error_ulp_bf16(NAME, output, ref_output, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    BETA, NUM_ELEMS);                                          \
  CHECK_RESULT(FUNCTION, NAME, TOL);

  // Run subset of functions depending on ISA compatibility
#if defined(RUN_SCALAR)
  RUN(skl_softmax_bf16_ref, "reference", 1);
#endif
#if defined(__riscv_zve32f) && defined(RUN_ZVE32F)
  RUN(skl_softmax_bf16_zve32f, "zve32f", 1);
#endif
#if defined(__riscv_zvfbfmin) && defined(RUN_BFMIN)
  RUN(skl_softmax_bf16_zvfbfmin, "zvfbfmin", 1);
#endif
#if defined(__riscv_xsfvfbfa) && defined(RUN_VFBFA)
  RUN(skl_softmax_bf16_xsfvfbfa, "xsfvfbfa", 16);
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin) &&                 \
    defined(RUN_VFEXPA_BFMIN)
  RUN(skl_softmax_bf16_xsfvfexpa_zvfbfmin, "xsfvfexpa+zvfbfmin", 2);
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_xsfvfbfa) &&                 \
    defined(RUN_VFEXPA_VFBFA)
  RUN(skl_softmax_bf16_xsfvfexpa_xsfvfbfa, "xsfvfexpa+xsfvfbfa", 18);
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_zvfbfmin) &&             \
    defined(RUN_VFEXP_BFMIN)
  RUN(skl_softmax_bf16_xsfvfbfexp16e_zvfbfmin, "xsfvfbfexp16e+zvfbfmin", 16);
#endif
#if defined(__riscv_xsfvfbfexp16e) && defined(__riscv_xsfvfbfa) &&             \
    defined(RUN_VFEXP_VFBFA)
  RUN(skl_softmax_bf16_xsfvfbfexp16e_xsfvfbfa, "xsfvfbfexp16e+xsfvfbfa", 16);
#endif
#if defined(__riscv_xsfvfexp32e) && defined(__riscv_zvfbfmin) &&               \
    defined(RUN_VFEXP32E_BFMIN)
  RUN(skl_softmax_bf16_xsfvfexp32e_zvfbfmin, "xsfvfbfexp32e+zvfbfmin", 1);
#endif

#if !(defined(RUN_SCALAR) || defined(RUN_VFBFA) || defined(RUN_VFEXP_VFBFA) || \
      defined(RUN_VFEXP_BFMIN) || defined(RUN_VFEXPA_VFBFA) ||                 \
      defined(RUN_VFEXPA_BFMIN) || defined(RUN_ZVE32F) ||                      \
      defined(RUN_BFMIN) || defined(RUN_VFEXP32E_BFMIN))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
