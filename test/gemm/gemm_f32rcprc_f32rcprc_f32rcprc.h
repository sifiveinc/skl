// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @brief Test and benchmark for packed GEMM: C_pack = alpha * A_pack * B_pack +
 * beta * C_pack.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Block dimensions M0, N0, and K0
 *  - Packed matrix dimensions M1, N1, and K1
 *  - Scalar coefficients ALPHA and BETA
 *  - Row and column strides (RSA0, CSA0, RSA1, CSA1, RSB0, CSB0, RSB1, CSB1,
 *    RSC0, CSC0, RSC1, CSC1)
 *  - The GEMM kernel function to test
 *
 * Matrix layouts:
 *  - A_pack is packed M1 x K1 with block size M0 x K0 and strides RSA0, CSA0,
 *    RSA1, CSA1
 *  - B_pack is packed K1 x N1 with block size K0 x N0 and strides RSB0, CSB0,
 *    RSB1, CSB1
 *  - C_pack is packed M1 x N1 with block size M0 x N0 and strides RSC0, CSC0,
 *    RSC1, CSC1
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

  // Buffer generation settings for A_pack, B_pack, C_pack
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
