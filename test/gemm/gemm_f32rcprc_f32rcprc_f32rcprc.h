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
  size_t m0, n0, k0;
  size_t m1, n1, k1;
  float alpha;
  size_t rsa0, csa0, rsa1, csa1;
  size_t rsb0, csb0, rsb1, csb1;
  float beta;
  size_t rsc0, csc0, rsc1, csc1;

  // Buffer generation settings for A, B, C
  SKL_TEST_BUFFER(float) a_pack, b_pack, c_pack;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *a_wide, *b_wide;
    float *ref_c;
    double *bound;
  } ctx;
} gemm_f32rcprc_f32rcprc_f32rcprc_t;

void gemm_f32rcprc_f32rcprc_f32rcprc_init(skl_test_t *t);
void gemm_f32rcprc_f32rcprc_f32rcprc_verify(skl_test_t *t);
void gemm_f32rcprc_f32rcprc_f32rcprc_report(skl_test_t *t);
void gemm_f32rcprc_f32rcprc_f32rcprc_cleanup(skl_test_t *t);

#define GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS                               \
  .a_pack = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},              \
  .b_pack = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},              \
  .c_pack = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM}
