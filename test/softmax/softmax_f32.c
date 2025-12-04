#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK.
#endif

#if !defined(NUM_ELEMS)
#define NUM_ELEMS 2048 // Input length
#endif

#if !defined(BETA)
#define BETA 1.0 // Exponential scaling factor
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
#define RUN_ZVE32F 1
#define RUN_VFEXPA 1
#define RUN_VFEXP 1
#endif

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) float input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float ref_output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) double workspace[NUM_ELEMS];

#if defined(ENABLE_TEST)
/** Scalar softmax using FP64 intermediates for high accuracy */
static void reference_softmax_f32(float *out, const float *in, float beta,
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
    out[i] = (float)(workspace[i] / sum);
  }
}
#endif // defined(ENABLE_TEST)

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element softmax:\n\n", NUM_ELEMS);
  skl_test_init_f32(input, NUM_ELEMS, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_BENCHMARK)
  // cycle and instruction counters
  uint64_t c0, c1, i0, i1; // NOLINT(readability-isolate-declaration)
#define MEASURE_PERF(FUNCTION, NAME)                                           \
  /* Measure 2nd run after caches warmed */                                    \
  c0 = riscv_read_mcycle(), i0 = riscv_read_minstret();                        \
  FUNCTION(output, input, BETA, NUM_ELEMS);                                    \
  riscv_fence();                                                               \
  c1 = riscv_read_mcycle(), i1 = riscv_read_minstret();                        \
  report_perf_epc(NAME, c1 - c0, i1 - i0, NUM_ELEMS);
#else
#define MEASURE_PERF(FUNCTION, NAME)
#endif

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  reference_softmax_f32(ref_output, input, BETA, workspace, NUM_ELEMS);
#define CHECK_RESULT(FUNCTION, NAME)                                           \
  ret += skl_check_error_ulp_f32(NAME, output, ref_output, 64, NUM_ELEMS);
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
  RUN(skl_softmax_f32_scalar, "scalar");
#endif
#if defined(__riscv_zve32f) && defined(RUN_ZVE32F)
  RUN(skl_softmax_f32_zve32f, "zve32f");
#endif
#if defined(__riscv_xsfvfexpa) && defined(RUN_VFEXPA)
  RUN(skl_softmax_f32_xsfvfexpa, "xsfvfexpa");
#endif
#if defined(__riscv_xsfvfexp32e) && defined(RUN_VFEXP)
  RUN(skl_softmax_f32_xsfvfexp32e, "xsfvfexp32e");
#endif

#if !(defined(RUN_SCALAR) || defined(RUN_ZVE32F) || defined(RUN_VFEXPA) ||     \
      defined(RUN_VFEXP))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
