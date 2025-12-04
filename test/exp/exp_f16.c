#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 2048 // Default input length
#endif

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void scalar_exp_f16(_Float16 *out, const _Float16 *in, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = (_Float16)expf(in[i]);
  }
}

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) _Float16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 ref_output[NUM_ELEMS];

int main(void) {
  int ret = 0; // return value
#if defined(ENABLE_BENCHMARK)
  // Cycle and instruction counts
  uint64_t c0, c1, i0, i1; // NOLINT(readability-isolate-declaration)
#endif
  printf("Measuring %d-element exponential:\n", NUM_ELEMS);
  skl_test_init_f16(input, NUM_ELEMS, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_exp_f16(ref_output, input,
                 NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_BENCHMARK)
#define MEASURE_PERF(FUNCTION, NAME)                                           \
  /* Measure 2nd run after caches warmed */                                    \
  c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();                        \
  FUNCTION(output, input, NUM_ELEMS);                                          \
  c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();                        \
  report_perf_epc(NAME, c1 - c0, i1 - i0, NUM_ELEMS);
#else
#define MEASURE_PERF(FUNCTION, NAME)
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += skl_check_error_ulp_f16(NAME, output, ref_output, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  FUNCTION(output, input, NUM_ELEMS);                                          \
  MEASURE_PERF(FUNCTION, NAME);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zvfh) && defined(RUN_ZVFH)
  RUN(skl_exp_1u_f16_zvfh, "zvfh", 1);
#endif

#if defined(__riscv_xsfvfexp16e) && defined(RUN_VFEXP)
  RUN(skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e, "xsfvfexp32e", 3);
  RUN(skl_exp_3p16u_f16_xsfvfexp16e, "gen xsfvfexp32e", 6);
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh) && defined(RUN_VFEXPA)
  RUN(skl_exp_1u_f16_xsfvfexpa_zvfh, "xsfvfexpa", 1);
  RUN(skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh, "weak xsfvfexpa", 1);
#endif

#if !(defined(RUN_ZVFH) || defined(RUN_VFEXP) || defined(RUN_VFEXPA))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
