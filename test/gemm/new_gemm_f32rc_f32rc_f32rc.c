#include "skl-test-driver.h"
#include "skl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// All members are configurable by SKL test parameters of the same name
// in skl_test_init.
static struct {
  size_t M;
  size_t N;
  size_t K;
  size_t RSA;
  size_t CSA;
  size_t RSB;
  size_t CSB;
  size_t RSC;
  size_t CSC;
  float ALPHA;
  float BETA;
  const char *FUNC;
} gemm = {
    .M = 128,
    .N = 128,
    .K = 128,
    .ALPHA = 1.0f,
    .BETA = 0.0f,
    .FUNC = "skl_gemm_f32_f32_f32_zve32f_x390_wrapper",
};

// All FUNCs need to have the same signature, so define wrappers for each
// variant that we want to test.

#if defined(__riscv_zve32f)
int skl_gemm_f32_f32_f32_zve32f_x390_wrapper(size_t m, size_t n, size_t k,
                                             float alpha, const float *a,
                                             size_t rsa, size_t csa,
                                             const float *b, size_t rsb,
                                             size_t csb, float beta, float *c,
                                             size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, csa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  if (status) {
    return status;
  }

  skl_gemm_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                   rsc);
  return 0;
}

int skl_gemm_f32_f32_f32_zve32f_x390_clp_wrapper(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    size_t csa, const float *b, size_t rsb, size_t csb, float beta, float *c,
    size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, csa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  if (status) {
    return status;
  }
  skl_gemm_f32_f32_f32_zve32f_x390_clp(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                       rsc);
  return 0;
}
#endif

SKL_TEST_FUNCS(SKL_TEST_FUNC(skl_gemm_f32_f32_f32_zve32f_x390_wrapper),
               SKL_TEST_FUNC(skl_gemm_f32_f32_f32_zve32f_x390_clp_wrapper))

// The buffers to use for the test.
static float *a, *b, *c, *c_ref;
static double *a_wide, *b_wide;
static double *bound;

int skl_test_init(skl_test_param_t *params, size_t num_params) {
  // Parse the configurable parameters.
  SKL_TEST_PARAMS(params, num_params, SKL_TEST_PARAM_SZ("M", gemm.M),
                  SKL_TEST_PARAM_SZ("N", gemm.N),
                  SKL_TEST_PARAM_SZ("K", gemm.K),
                  SKL_TEST_PARAM_F32("ALPHA", gemm.ALPHA),
                  SKL_TEST_PARAM_F32("BETA", gemm.BETA));

  // Set the derived parameters.
  gemm.RSA = gemm.M;
  gemm.CSA = 1;
  gemm.RSB = gemm.N;
  gemm.CSB = 1;
  gemm.RSC = gemm.N;
  gemm.CSC = 1;

  // Allocate and initialize driver-managed buffers.
  SKL_TEST_BUFFER(&a, float, gemm.M *gemm.K);
  SKL_TEST_BUFFER(&b, float, gemm.K *gemm.N);
  SKL_TEST_BUFFER(&c, float, gemm.M *gemm.N);

  // Buffers the driver does not manage (initialized in SKL_TEST_VERIFY).
  c_ref = (float *)malloc(gemm.M * gemm.N * sizeof(float));
  a_wide = (double *)malloc(gemm.M * gemm.K * sizeof(double));
  b_wide = (double *)malloc(gemm.K * gemm.N * sizeof(double));
  bound = (double *)malloc(gemm.M * gemm.N * sizeof(double));

  return 0;
}

int skl_test_execute(void) {
  // Execute the test.
  return skl_test_config.FUNC(gemm.M, gemm.N, gemm.K, gemm.ALPHA, a, gemm.RSA,
                              gemm.CSA, b, gemm.RSB, gemm.CSB, gemm.BETA, c,
                              gemm.RSC, gemm.CSC);
}

int skl_test_report(uint64_t cycles, uint64_t insts) {
  size_t maccs = gemm.M * gemm.N * gemm.K;
  float mpc = (float)maccs / (float)cycles;
  SKL_TEST_RESULT("MACCS", "%lu", maccs);
  SKL_TEST_RESULT("MACCS/CYCLE", "%f", mpc);
  return 0;
}

int skl_test_verify(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_f32rc_f32rc_f32rc_scalar(gemm.M, gemm.N, gemm.K, gemm.ALPHA, a,
                                    gemm.RSA, gemm.CSA, b, gemm.RSB, gemm.CSB,
                                    gemm.BETA, c_ref, gemm.RSC, gemm.CSC);
  size_t ALEN = gemm.M * gemm.K;
  size_t BLEN = gemm.K * gemm.N;
  size_t CLEN = gemm.M * gemm.N;

  //
  // Compute the error bound array for comparing test vs reference results.
  //
  // For any matrix M, define |M| as the matrix of absolute values where:
  //     |M|(i,j) = |M(i,j)|
  //
  // Let u = 2^-P be the maximum relative roundoff error for a floating-point
  // type with P-1 mantissa bits.
  //
  // For GEMM operations, the error between computed and exact results is
  // bounded by:
  //     ((1 + u)^(K + 2) - 1) * (|alpha| * |A| * |B| + |beta| * |C|)
  //
  // Since both test and reference results have roundoff errors, we double this
  // bound using the triangle inequality to get the final comparison threshold.
  //
  for (size_t i = 0; i < ALEN; ++i) {
    a_wide[i] = fabsf(a[i]);
  }
  for (size_t i = 0; i < BLEN; ++i) {
    b_wide[i] = fabsf(b[i]);
  }
  for (size_t i = 0; i < CLEN; ++i) {
    bound[i] = fabs((double)c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1(((double)gemm.K + 2) * log1p(u));
  skl_gemm_f64rc_f64rc_f64rc_scalar(
      gemm.M, gemm.N, gemm.K, roundoff_scaling * fabs((double)gemm.ALPHA),
      a_wide, gemm.RSA, gemm.CSA, b_wide, gemm.RSB, gemm.CSB,
      roundoff_scaling * fabs((double)gemm.BETA), bound, gemm.RSC, gemm.CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < gemm.M; ++i) {
    for (size_t j = 0; j < gemm.N; ++j) {
      size_t idx = i * gemm.RSC + j * gemm.CSC;
      if (fabs((double)c[idx] - (double)c_ref[idx]) > bound[idx]) {
        printf("result [%zu, %zu] (%f) != reference (%f)\n", i, j, c[idx],
               c_ref[idx]);
        return 1;
      }
    }
  }

  return 0;
}

int skl_test_finish(void) {
  // Free the buffers.
  SKL_TEST_FREE(a);
  SKL_TEST_FREE(b);
  SKL_TEST_FREE(c);
  SKL_TEST_FREE(c_ref);
  SKL_TEST_FREE(bound);
  free(c_ref);
  free(bound);
  free(a_wide);
  free(b_wide);
  return 0;
}
