#include "skl-test-driver.h"
#include "skl.h"
#include <math.h>

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
} params = {
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
void skl_gemm_f32_f32_f32_zve32f_x390_wrapper(size_t m, size_t n, size_t k,
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
    exit(status);
  }
  skl_gemm_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                   rsc);
}

void skl_gemm_f32_f32_f32_zve32f_x390_clp_wrapper(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    size_t csa, const float *b, size_t rsb, size_t csb, float beta, float *c,
    size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, csa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_f32_f32_f32_zve32f_x390_clp(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                       rsc);
}
#endif

SKL_TEST_FUNCS(SKL_TEST_FUNC(skl_gemm_f32_f32_f32_zve32f_x390_wrapper),
               SKL_TEST_FUNC(skl_gemm_f32_f32_f32_zve32f_x390_clp_wrapper));

// The buffers to use for the test.
static float *a, *b, *c, *c_ref;
static double *a_wide, *b_wide;
static double *bound;

int skl_test_init(skl_test_param_t *params, size_t num_params) {
  // Parse the configurable parameters.
  SKL_TEST_PARAMS(params, num_params, SKL_TEST_PARAM_SZ("M", params.M),
                  SKL_TEST_PARAM_SZ("N", params.N),
                  SKL_TEST_PARAM_SZ("K", params.K),
                  SKL_TEST_PARAM_F32("ALPHA", params.ALPHA),
                  SKL_TEST_PARAM_F32("BETA", params.BETA));

  // Set the derived parameters.
  params.RSA = params.M;
  params.CSA = 1;
  params.RSB = params.N;
  params.CSB = 1;
  params.RSC = params.N;
  params.CSC = 1;

  // Allocate and initialize driver-managed buffers.
  SKL_TEST_BUFFER(&a, float, params.M *params.K);
  SKL_TEST_BUFFER(&b, float, params.K *params.N);
  SKL_TEST_BUFFER(&c, float, params.M *params.N);

  // Buffers the driver does not manage (initialized in SKL_TEST_VERIFY).
  c_ref = (float)malloc(params.M * params.N * sizeof(float));
  a_wide = (double)malloc(params.M * params.K * sizeof(double));
  b_wide = (double)malloc(params.K * params.N * sizeof(double));
  bound = (double)malloc(params.M * params.N * sizeof(double));

  return 0;
}

int skl_test_execute(void) {
  // Execute the test.
  return params.FUNC(params.M, params.N, params.K, params.ALPHA, a, params.RSA,
                     params.CSA, b, params.RSB, params.CSB, params.BETA, c,
                     params.RSC, params.CSC);
}

int skl_test_report(uint64_t cycles, uint64_t insts) {
  size_t maccs = params.M * params.N * params.K;
  float mpc = (float)maccs / (float)cycles;
  SKL_TEST_RESULT("MACCS", "%lu", maccs);
  SKL_TEST_RESULT("MACCS/CYCLE", "%f", mpc);
  return 0;
}

int skl_test_verify(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_f32rc_f32rc_f32rc_scalar(
      params.M, params.N, params.K, params.ALPHA, a, params.RSA, params.CSA, b,
      params.RSB, params.CSB, params.BETA, c_ref, params.RSC, params.CSC);
  size_t ALEN = params.M * params.K;
  size_t BLEN = params.K * params.N;
  size_t CLEN = params.M * params.N;

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
  const double roundoff_scaling = 2 * expm1((params.K + 2) * log1p(u));
  skl_gemm_f64rc_f64rc_f64rc_scalar(
      params.M, params.N, params.K,
      roundoff_scaling * fabs((double)params.ALPHA), a_wide, params.RSA,
      params.CSA, b_wide, params.RSB, params.CSB,
      roundoff_scaling * fabs((double)params.BETA), bound, params.RSC,
      params.CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < params.M; ++i) {
    for (size_t j = 0; j < params.N; ++j) {
      size_t idx = i * params.RSC + j * params.CSC;
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
