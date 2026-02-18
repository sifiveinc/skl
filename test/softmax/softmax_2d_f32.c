#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK.
#endif

#if !defined(M) // Input rows
#error Must define M
#endif

#if !defined(N) // Input columns
#error Must define N
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

#if !defined(RSA)
#define RSA N
#endif
#if !defined(RSS)
#define RSS N
#endif

#if RSA < N || RSS < N
#error RSA and RSS must be greater than or equal to N
#endif

enum {
  ALIGN = 4096,
  ALEN = M * RSA,
  SLEN = M * RSS,
};
__attribute__((aligned(ALIGN))) float input[ALEN];
__attribute__((aligned(ALIGN))) float output[SLEN];
__attribute__((aligned(ALIGN))) float ref_output[SLEN];
__attribute__((aligned(ALIGN))) double workspace[N];

#if defined(ENABLE_TEST)
/** Scalar softmax using FP64 intermediates for high accuracy */
static void reference_softmax_2d_f32(float *S, const size_t rss, const float *A,
                                     const size_t rsa, const float beta,
                                     const size_t m, const size_t n,
                                     double *workspace) {
  if (m == 0 || n == 1)
    return;

  for (size_t i = 0; i < m; i++) {
    float max = A[i * rsa];
    for (size_t j = 1; j < n; j++) {
      max = fmaxf(A[i * rsa + j], max);
    }

    double sum = 0;
    for (size_t j = 0; j < n; j++) {
      workspace[j] = exp(beta * ((double)A[i * rsa + j] - max));
      sum += workspace[j];
    }

    for (size_t j = 0; j < n; j++) {
      S[i * rss + j] = (float)(workspace[j] / sum);
    }
  }
}
#endif // defined(ENABLE_TEST)

int main(void) {
  int ret = 0; // return value
  printf("Measuring [%dx%d] softmax:\n\n", M, N);
  skl_test_init_f32(input, ALEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  memset(ref_output, 0, SLEN * sizeof(*ref_output));
  reference_softmax_2d_f32(ref_output, RSS, input, RSA, BETA, M, N, workspace);
  float max_err;
#define CHECK_RESULT(FUNCTION, NAME)                                           \
  max_err = 0.f;                                                               \
  for (size_t i = 0; i < M; i++) {                                             \
    max_err = fmaxf(max_err, skl_error_ulp_f32(output + i * RSS,               \
                                               ref_output + i * RSS, N));      \
  }                                                                            \
  skl_print_max_error(NAME, max_err);                                          \
  ret += (max_err > 64);
#else
#define CHECK_RESULT(FUNCTION, NAME)
#endif

#define RUN(FUNCTION, NAME)                                                    \
  memset(output, 0, SLEN * sizeof(*output));                                   \
  SKL_BENCHMARK_RUN(NAME, (M * N), SKL_TEST_WARMUP, FUNCTION, output, RSS,     \
                    input, RSA, BETA, M, N);                                   \
  CHECK_RESULT(FUNCTION, NAME);

  // Run subset of functions depending on ISA compatibility
#if defined(RUN_SCALAR)
  RUN(skl_softmax_2d_f32_scalar, "scalar");
#endif
#if defined(__riscv_zve32f) && defined(RUN_ZVE32F)
  RUN(skl_softmax_2d_f32_zve32f, "zve32f");
#endif
#if defined(__riscv_xsfvfexpa) && defined(RUN_VFEXPA)
  RUN(skl_softmax_2d_f32_xsfvfexpa, "xsfvfexpa");
#endif
#if defined(__riscv_xsfvfexp32e) && defined(RUN_VFEXP)
  RUN(skl_softmax_2d_f32_xsfvfexp32e, "xsfvfexp32e");
#endif

#if !(defined(RUN_SCALAR) || defined(RUN_ZVE32F) || defined(RUN_VFEXPA) ||     \
      defined(RUN_VFEXP))
#error No tests or benchmarks enabled!
#endif

  return ret > 0;
}
