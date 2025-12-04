#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK.
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 2048 // Input length
#endif

#if !defined(BETA)
#define BETA 1.0 // Exponential scaling factor
#endif

#if !defined(TOL_ULPS)
#define TOL_ULPS 64 // Error tolerance in ULPs
#endif

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
#define RUN_VFEXP 1
#define RUN_VFEXPA 1
#define RUN_ZVFH 1
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) _Float16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) _Float16 ref_output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) double workspace[NUM_ELEMS];

#if defined(ENABLE_TEST)
/** Scalar softmax using FP64 intermediates for high accuracy */
static void reference_softmax_f16(_Float16 *out, const _Float16 *in,
                                  _Float16 beta, double *workspace, size_t n) {
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
    out[i] = (_Float16)(workspace[i] / sum);
  }
}
#endif // defined(ENABLE_TEST)

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element softmax:\n\n", NUM_ELEMS);
  skl_test_init_f16(input, NUM_ELEMS, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);

#if defined(ENABLE_BENCHMARK)
  // cycle and instruction counters
  uint64_t c0, c1, i0, i1; // NOLINT(readability-isolate-declaration)
#define MEASURE_PERF(FUNCTION, NAME)                                           \
  /* Measure 2nd run after caches warmed */                                    \
  c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();                        \
  FUNCTION(output, input, BETA, NUM_ELEMS);                                    \
  c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();                        \
  report_perf_epc(NAME, c1 - c0, i1 - i0, NUM_ELEMS);
#else
#define MEASURE_PERF(FUNCTION, NAME)
#endif

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  reference_softmax_f16(ref_output, input, BETA, workspace, NUM_ELEMS);
#define CHECK_RESULT(FUNCTION, NAME)                                           \
  ret += skl_check_error_ulp_f16(NAME, output, ref_output, TOL_ULPS, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME)
#endif

#define RUN(FUNCTION, NAME)                                                    \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  FUNCTION(output, input, BETA, NUM_ELEMS);                                    \
  MEASURE_PERF(FUNCTION, NAME);                                                \
  CHECK_RESULT(FUNCTION, NAME);

  // Run subset of functions depending on ISA compatibility
#if defined(RUN_SCALAR)
  RUN(skl_softmax_f16_scalar, "scalar");
#endif
#if defined(__riscv_xsfvfexp16e) && defined(RUN_VFEXP)
  RUN(skl_softmax_f16_xsfvfexp16e, "xsfvfexp16e");
#endif
#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfh) && defined(RUN_VFEXPA)
  RUN(skl_softmax_f16_xsfvfexpa_zvfh, "xsfvfexpa+zvfh");
#endif
#if defined(__riscv_zvfh) && defined(RUN_ZVFH)
  RUN(skl_softmax_f16_zvfh, "zvfh");
#endif

#if !(defined(RUN_SCALAR) || defined(RUN_VFEXP) || defined(RUN_VFEXPA) ||      \
      defined(RUN_ZVFH))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
