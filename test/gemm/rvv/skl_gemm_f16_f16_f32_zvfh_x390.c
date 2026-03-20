#include "gemm/gemm_f16rc_f16rc_f32rc.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension.
#endif

static int execute(skl_test_t *t);

gemm_f16rc_f16rc_f32rc_t tests[] = {
    {.warmup = false,
     .verify = true,
     .alpha = (_Float16)2.0f,
     .beta = (_Float16)3.0f,
     .a = {.min = (_Float16)-1.0f,
           .max = (_Float16)1.0f,
           .mode = SKL_TEST_RANDOM},
     .b = {.min = (_Float16)-1.0f,
           .max = (_Float16)1.0f,
           .mode = SKL_TEST_RANDOM},
     .c = {.min = (_Float16)-1.0f,
           .max = (_Float16)1.0f,
           .mode = SKL_TEST_RANDOM},
     .m = 32,
     .n = 32,
     .k = 32},
};

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
