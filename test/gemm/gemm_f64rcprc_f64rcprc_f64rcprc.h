// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for packed GEMM: C_pack = alpha * A_pack * B_pack +
 * beta * C_pack.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Block dimensions m0, n0, and k0
 *  - Packed matrix dimensions m1, n1, and k1
 *  - Scalar coefficients alpha and beta
 *  - Row and column strides (rsa0, csa0, rsa1, csa1, rsb0, csb0, rsb1, csb1,
 *    rsc0, csc0, rsc1, csc1)
 *  - The GEMM kernel function to test
 *
 * Matrix layouts:
 *  - A_pack is packed m1 x k1 with block size m0 x k0 and strides rsa0, csa0,
 *    rsa1, csa1
 *  - B_pack is packed k1 x n1 with block size k0 x n0 and strides rsb0, csb0,
 *    rsb1, csb1
 *  - C_pack is packed m1 x n1 with block size m0 x n0 and strides rsc0, csc0,
 *    rsc1, csc1
 */

#pragma once

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
  double alpha;
  size_t rsa0, csa0, rsa1, csa1;
  size_t rsb0, csb0, rsb1, csb1;
  double beta;
  size_t rsc0, csc0, rsc1, csc1;

  // Buffer generation settings for A_pack, B_pack, C_pack
  SKL_TEST_BUFFER(double) a_pack, b_pack, c_pack;

  // Derived parameters & buffers (private to the test harness)
  struct {
    long double *a_wide, *b_wide;
    double *ref_c;
    long double *bound;
  } ctx;
} gemm_f64rcprc_f64rcprc_f64rcprc_t;

void gemm_f64rcprc_f64rcprc_f64rcprc_init(skl_test_t *t);
void gemm_f64rcprc_f64rcprc_f64rcprc_verify(skl_test_t *t);
void gemm_f64rcprc_f64rcprc_f64rcprc_test_report(skl_test_t *t);
void gemm_f64rcprc_f64rcprc_f64rcprc_benchmark_report(skl_test_t *t);
void gemm_f64rcprc_f64rcprc_f64rcprc_cleanup(skl_test_t *t);

#define GEMM_F64RCPRC_F64RCPRC_F64RCPRC_DEFAULTS                               \
  .a_pack = {.min = -1.0, .max = 1.0, .mode = SKL_TEST_RANDOM},                \
  .b_pack = {.min = -1.0, .max = 1.0, .mode = SKL_TEST_RANDOM},                \
  .c_pack = {.min = -1.0, .max = 1.0, .mode = SKL_TEST_RANDOM}
