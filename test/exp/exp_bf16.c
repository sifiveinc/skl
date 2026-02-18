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

void scalar_exp_bf16(__bf16 *out, const __bf16 *in, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = (__bf16)expf(in[i]);
  }
}

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) __bf16 input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) __bf16 output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) __bf16 ref_output[NUM_ELEMS];

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element exponential:\n", NUM_ELEMS);
  skl_test_init_bf16(input, NUM_ELEMS, SKL_TEST_MIN_BF16, SKL_TEST_MAX_BF16);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  scalar_exp_bf16(ref_output, input,
                  NUM_ELEMS); // Use scalar output as reference
#endif

#if defined(ENABLE_TEST)
#define CHECK_RESULT(FUNCTION, NAME, TOL)                                      \
  ret += skl_check_error_ulp_bf16(NAME, output, ref_output, TOL, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL)
#endif

#define RUN(FUNCTION, NAME, TOL)                                               \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    NUM_ELEMS);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL);

#if defined(__riscv_zve32f) && defined(RUN_ZVE32F)
  RUN(skl_exp_1u_bf16_zve32f, "zve32f", 1);
#endif

#if defined(__riscv_xsfvfbfa) && defined(RUN_VFBFA)
  RUN(skl_exp_1u_bf16_xsfvfbfa, "xsfvfbfa", 1);
#endif

#if defined(__riscv_xsfvfbfexp16e) && defined(RUN_VFEXP)
  RUN(skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e, "xsfvfbfexp16e", 3);
#endif

#if defined(__riscv_xsfvfexpa) && defined(__riscv_zvfbfmin) &&                 \
    defined(RUN_VFEXPA)
  RUN(skl_exp_1u_bf16_xsfvfexpa_zvfbfmin, "xsfvfexpa+zvfbfmin", 1);
#endif

#if !(defined(RUN_ZVE32F) || defined(RUN_VFBFA) || defined(RUN_VFEXP) ||       \
      defined(RUN_VFEXPA))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
