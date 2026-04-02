// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#include "skl-test-driver.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to GEMM function)
  size_t m, n, k;
  float alpha;
  size_t rsa, csa;
  size_t rsb, csb;
  float beta;
  size_t rsc, csc;

  // Buffer generation settings for A, B, C
  SKL_TEST_BUFFER(float) a, b, c;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *a_wide, *b_wide;
    float *ref_c;
    double *bound;
  } ctx;
} gemm_f32rc_f32rc_f32rc_t;

void gemm_f32rc_f32rc_f32rc_init(skl_test_t *t);
void gemm_f32rc_f32rc_f32rc_verify(skl_test_t *t);
void gemm_f32rc_f32rc_f32rc_report(skl_test_t *t);
void gemm_f32rc_f32rc_f32rc_cleanup(skl_test_t *t);

#define GEMM_F32RC_F32RC_F32RC_DEFAULTS                                        \
  .a = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},                   \
  .b = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},                   \
  .c = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM}, .alpha = 1.0f
