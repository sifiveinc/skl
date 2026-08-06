// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for packed GEMM: C = alpha * A * B + beta * C.
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
 *  - A is packed m1 x k1 with block size m0 x k0 and strides rsa0, csa0,
 *    rsa1, csa1
 *  - B is packed k1 x n1 with block size k0 x n0 and strides rsb0, csb0,
 *    rsb1, csb1
 *  - C is packed m1 x n1 with block size m0 x n0 and strides rsc0, csc0,
 *    rsc1, csc1
 */

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef enum { F8E4M3, F8E5M2 } gemm_ofp8_type_t;

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

  // OFP8 type for A and B
  gemm_ofp8_type_t input_type;

  // Buffer generation settings for A, B, C
  SKL_TEST_BUFFER(uint8_t) a;
  SKL_TEST_BUFFER(uint8_t) b;
  SKL_TEST_BUFFER(float) c;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *a_wide, *b_wide;
    float *ref_c;
    double *bound;
  } ctx;
} gemm_f8rcprc_f8rcprc_f32rcprc_t;

void gemm_f8rcprc_f8rcprc_f32rcprc_init(skl_test_t *t);
void gemm_f8rcprc_f8rcprc_f32rcprc_verify(skl_test_t *t);
void gemm_f8rcprc_f8rcprc_f32rcprc_test_report(skl_test_t *t);
void gemm_f8rcprc_f8rcprc_f32rcprc_benchmark_report(skl_test_t *t);
void gemm_f8rcprc_f8rcprc_f32rcprc_cleanup(skl_test_t *t);

// The min value for E4M3 is 2^-6, represented by 0.0001.000.
// The max value for E4M3 is 1.75 * 2^8, represented by 0.1111.110.
#define GEMM_F8E4M3RCPRC_F8E4M3RCPRC_F32RCPRC_DEFAULTS                         \
  .input_type = F8E4M3, .a = {.min = 8, .max = 126, .mode = SKL_TEST_RANDOM},  \
  .b = {.min = 8, .max = 126, .mode = SKL_TEST_RANDOM},                        \
  .c = {.min = 8, .max = 126, .mode = SKL_TEST_RANDOM}

// The min value for E5M2 is 2^-14, represented by 0.00001.00.
// The max value for E5M2 is 1.75 * 2^15, represented by 0.11110.11.
#define GEMM_F8E5M2RCPRC_F8E5M2RCPRC_F32RCPRC_DEFAULTS                         \
  .input_type = F8E5M2, .a = {.min = 4, .max = 123, .mode = SKL_TEST_RANDOM},  \
  .b = {.min = 4, .max = 123, .mode = SKL_TEST_RANDOM},                        \
  .c = {.min = 4, .max = 123, .mode = SKL_TEST_RANDOM}
