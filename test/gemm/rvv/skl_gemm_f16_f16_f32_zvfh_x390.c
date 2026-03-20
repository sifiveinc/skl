#include "gemm/gemm_f16rc_f16rc_f32rc.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension.
#endif

/**
 * @brief Test cases for skl_gemm_f16_f16_f32_zvfh_x390.
 *
 * This test uses the gemm_f32rc_f32rc_f32rc harness with the following
 * restrictions on the input parameters:
 *  - All matrices are row-major (csa == 1, csb == 1, csc == 1)
 */

static int execute(skl_test_t *t);

// Uses defaults from gemm_f16rc_f16rc_f32rc.h
#define TEST GEMM_F16RC_F16RC_F32RC_DEFAULTS, .warmup = false, .verify = true
#define BENCH GEMM_F16RC_F16RC_F32RC_DEFAULTS, .warmup = true, .verify = false

// clang-format off
gemm_f16rc_f16rc_f32rc_t tests[] = {
    // Benchmark tests
    {BENCH, .m =  64, .n = 128, .k = 128},

    // Verifcation tests - comprehensive coverage for RVV GEMM
    /* Edge cases: minimal dimensions */
    {TEST, .m =   1, .n =   1, .k =   0},
    {TEST, .m =   1, .n =   1, .k =   1},
    /* Small odd dimensions for remainder handling */
    {TEST, .m =   7, .n =   7, .k =   7},
    {TEST, .m =  17, .n =  17, .k =  17},
    /* Skinny matrices (one dimension = 1) */
    {TEST, .m =   1, .n =  33, .k =  31},
    {TEST, .m =  33, .n =   1, .k =  31},
    /* k=0 edge case (C = beta*C, no A*B contribution) */
    {TEST, .m =  33, .n =  33, .k =   0},
    {TEST, .m =  16, .n =  16, .k =   0},
    /* Vector length boundary tests (multiples of 4, 8, 16, 32) */
    {TEST, .m =  16, .n =  16, .k =  16},
    {TEST, .m =  32, .n =  32, .k =  32},
    {TEST, .m =  32, .n = 128, .k =  32},
    /* Near-boundary tests (±1 from vector length multiples) */
    {TEST, .m =  15, .n =  17, .k =  31},
    {TEST, .m =  31, .n =  33, .k =  15},
    /* Wide and tall matrices */
    {TEST, .m =  33, .n = 129, .k =  32},
    {TEST, .m = 129, .n =  33, .k =  32},
    {TEST, .m =  31, .n = 133, .k =  32},
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_gemm_f16_f16_f32_zvfh_x390",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(gemm_f16rc_f16rc_f32rc_t),
    .tests = tests,
    .init = gemm_f16rc_f16rc_f32rc_init,
    .warmup = gemm_f16rc_f16rc_f32rc_warmup,
    .execute = execute,
    .verify = gemm_f16rc_f16rc_f32rc_verify,
    .report = gemm_f16rc_f16rc_f32rc_report,
    .cleanup = gemm_f16rc_f16rc_f32rc_cleanup,
};

static int execute(skl_test_t *t) {
  const gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;

  SKL_TEST_REQUIRE(t, h->csa == 1);
  SKL_TEST_REQUIRE(t, h->csb == 1);
  SKL_TEST_REQUIRE(t, h->csc == 1);
  skl_gemm_f16_f16_f32_zvfh_x390(h->m, h->n, h->k, h->alpha, h->a.data, h->rsa,
                                 h->b.data, h->rsb, h->beta, h->c.data, h->rsc);

  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

int main(void) {

  for (size_t i = 0; i < suite.num_tests; i++) {
    // Set default row stride to row length
    tests[i].rsa = tests[i].rsa ? tests[i].rsa : tests[i].k;
    tests[i].rsb = tests[i].rsb ? tests[i].rsb : tests[i].n;
    tests[i].rsc = tests[i].rsc ? tests[i].rsc : tests[i].n;
    // Make sure we still generate an error if a column stride is set
    // accidentally
    tests[i].csa = tests[i].csa ? tests[i].csa : 1;
    tests[i].csb = tests[i].csb ? tests[i].csb : 1;
    tests[i].csc = tests[i].csc ? tests[i].csc : 1;
  }

  return skl_test_driver_run_suite(&suite);
}
