// Copyright (c) 2025 SiFive, Inc.
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
#define RUN_ZVFH 1
#define RUN_SCALAR 1
#define RUN_XSFVFEXP16E 1
#define RUN_XSFVFEXPA 1
#endif

#if defined(ENABLE_TEST)
static inline void scalar_silu_f16(_Float16 *out, const _Float16 *in,
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
  /* If testing, fill in an array with random floats in [-100, 100] */
  const float min = -100;
  const float max = 100;
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

#if defined(ENABLE_TEST)
static int check_error(const char *name, const int16_t *res, const int16_t *ref,
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
#if defined(ENABLE_BENCHMARK)
  // Cycle and instruction counts
  uint64_t c0, c1, i0, i1; // NOLINT(readability-isolate-declaration)
#endif
  printf("Measuring %d-element silu:\n", NUM_ELEMS);

  _Float16 *output = (_Float16 *)output_bits;
  init_random(input, NUM_ELEMS); // input data
#if defined(ENABLE_TEST)
  _Float16 *ref_output = (_Float16 *)ref_output_bits;
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_silu_f16(ref_output, input,
                  NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_BENCHMARK)
#define MEASURE_PERF(FUNCTION, NAME)                                           \
  /* Measure 2nd run after caches warmed */                                    \
  riscv_fence();                                                               \
  c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();                        \
  FUNCTION(output, input, NUM_ELEMS);                                          \
  riscv_fence();                                                               \
  c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();                        \
  report_perf_epc(NAME, c1 - c0, i1 - i0, NUM_ELEMS);
#else
#define MEASURE_PERF(FUNCTION, NAME)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += check_error(NAME, output_bits, ref_output_bits, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  FUNCTION(output, input, NUM_ELEMS);                                          \
  MEASURE_PERF(FUNCTION, NAME);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zvfh) && defined(RUN_ZVFH)
  RUN(skl_silu_9u_f16_zvfh, "zvfh", 9);
#endif

#if defined(__riscv_xsfvfexp16e) && defined(RUN_XSFVFEXP16E)
  RUN(skl_silu_30u_f16_xsfvfexp16e, "xsfvfexp16e", 30);
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh) &&                     \
    defined(RUN_XSFVFEXPA)
  RUN(skl_silu_9u_f16_xsfvfexpa_zvfh, "xsfvfexpa_zvfh", 9);
#endif

#if defined(RUN_SCALAR)
  RUN(skl_silu_1u_f16_scalar, "scalar", 1)
#endif

#if !(defined(RUN_RVV) || defined(RUN_SCALAR) || defined(RUN_XSFVFEXP16E) ||   \
      defined(RUN_XSFVFEXPA))
#error No tests or benchmarks enabled!
#endif

  return ret;
}
