// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 1024
#endif

#if defined(ENABLE_BENCHMARK)
#include "skl-test.h"
#endif

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
#define RUN_RVV 1
#define RUN_SCALAR 1
#define RUN_XSFVFEXP32E 1
#define RUN_XSFVFEXPA 1
#endif

#if defined(ENABLE_TEST)
static inline void scalar_silu_f32(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = (float)(in[i] / (1 + exp(-(double)in[i])));
  }
}
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) float input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) int32_t output_bits[NUM_ELEMS];
__attribute__((aligned(ALIGN))) int32_t ref_output_bits[NUM_ELEMS];

static void init_random(float *arr, size_t len) {
#if defined(ENABLE_TEST)
  /* If testing, fill in an array with random floats in [-100, 100] */
  const float min = -100;
  const float max = 100;
  float frac;
  for (size_t i = 0; i < len; i++) {
    if (i == 0) {
      arr[i] = nanf("");
    } else {
      frac = (float)rand() / (float)RAND_MAX;
      arr[i] = frac * (max - min) + min;
    }
  }
#else
  /* Otherwise, save simulation time using increasing ints */
  int32_t next = 0;
  for (size_t i = 0; i < len; i++) {
    arr[i] = (float)(next++);
  }
#endif
}

#if defined(ENABLE_TEST)
static int check_error(const char *name, const int32_t *res, const int32_t *ref,
                       uint32_t tol, size_t len) {
  uint32_t max = 0;
  for (size_t i = 0; i < len; i++) {
    uint32_t err = abs(res[i] - ref[i]);
    if (err > max) {
      max = err;
    }
  }
  printf("%15s : maximum error %" PRIu32 " ulp\n", name, max);
  return max > tol;
}
#endif

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element silu:\n", NUM_ELEMS);

  float *output = (float *)output_bits;
  init_random(input, NUM_ELEMS); // input data
#if defined(ENABLE_TEST)
  float *ref_output = (float *)ref_output_bits;
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_silu_f32(ref_output, input,
                  NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += check_error(NAME, output_bits, ref_output_bits, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    NUM_ELEMS);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zve32f) && defined(RUN_RVV)
  RUN(skl_silu_52u_f32_zve32f, "zve32f", 52);
#endif

#if defined(__riscv_xsfvfexpa) && defined(RUN_XSFVFEXPA)
  RUN(skl_silu_52u_f32_xsfvfexpa, "xsfvfexpa", 52);
#endif

#if defined(__riscv_xsfvfexp32e) && defined(RUN_XSFVFEXP32E)
  RUN(skl_silu_52u_f32_xsfvfexp32e, "xsfvfexp32e", 52);
#endif

#if defined(RUN_SCALAR)
  RUN(skl_silu_52u_f32_scalar, "scalar", 52)
#endif

#if !(defined(RUN_RVV) || defined(RUN_SCALAR) || defined(RUN_XSFVFEXPA) ||     \
      defined(RUN_XSFVFEXP32E))
#error No tests or benchmarks enabled!
#endif

  return ret;
}
