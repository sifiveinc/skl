// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Test and benchmark for GEMM: C = alpha * A * B + beta * C.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Matrix dimensions M, N, and K
 *  - Scalar coefficients ALPHA and BETA
 *  - Row and column strides (RSA, CSA, RSB, CSB, RSC, CSC)
 *  - The GEMM kernel function to test
 *
 * Matrix layouts:
 *  - A is M x K with row stride RSA and column stride CSA
 *  - B is K x N with row stride RSB and column stride CSB
 *  - C is M x N with row stride RSC and column stride CSC
 */
#pragma once

#include "skl-test-driver.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test configuration
  bool warmup;
  bool verify;

  // Kernel parameters
  size_t m, n, k;
  _Float16 alpha;
  SKL_TEST_BUFFER(_Float16) a;
  size_t rsa, csa;
  SKL_TEST_BUFFER(_Float16) b;
  size_t rsb, csb;
  _Float16 beta;
  SKL_TEST_BUFFER(float) c;
  size_t rsc, csc;

  // Derived parameters & buffers (private to the test harness)
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

#define GEMM_F16RC_F16RC_F32RC_DEFAULTS                                        \
  .a = {.min = (_Float16)-1.0f,                                                \
        .max = (_Float16)1.0f,                                                 \
        .mode = SKL_TEST_RANDOM},                                              \
  .b = {.min = (_Float16)-1.0f,                                                \
        .max = (_Float16)1.0f,                                                 \
        .mode = SKL_TEST_RANDOM},                                              \
  .c = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},                   \
  .alpha = (_Float16)2.f, .beta = (_Float16)3.f
