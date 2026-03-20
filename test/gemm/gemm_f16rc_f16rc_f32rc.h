

#include "skl-test-driver.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  bool warmup;
  bool verify;

  size_t m, n, k;
  _Float16 alpha;
  SKL_TEST_BUFFER(_Float16) a;
  size_t rsa, csa;
  SKL_TEST_BUFFER(_Float16) b;
  size_t rsb, csb;
  _Float16 beta;
  SKL_TEST_BUFFER(float) c;
  size_t rsc, csc;

  struct {
    double *a_wide, *b_wide;
    float *ref_c;
    double *bound;
  } ctx;
} gemm_f16rc_f16rc_f32rc_t;

int gemm_f16rc_f16rc_f32rc_init(skl_test_t *t);
int gemm_f16rc_f16rc_f32rc_warmup(skl_test_t *t);
int gemm_f16rc_f16rc_f32rc_verify(skl_test_t *t);
int gemm_f16rc_f16rc_f32rc_report(skl_test_t *t);
int gemm_f16rc_f16rc_f32rc_cleanup(skl_test_t *t);
