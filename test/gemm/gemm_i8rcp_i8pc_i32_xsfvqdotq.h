// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for Xsfvqdotq GEMM: C = alpha * A_pack * B_pack +
 * beta * C.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Matrix dimensions m, n, and k
 *  - Scalar coefficients alpha and beta
 *  - Row and column strides (rsa1, csa1, rsb1, rsc)
 *  - The GEMM kernel function to test
 *
 * Matrix layouts:
 *  - A_pack is packed m x k1, where k1 = (k + 3) / 4, with 1 x 4 row-major
 *    blocks and strides rsa1 and csa1. The blocks in the last block-column of
 *    A_pack have (k - 1) % 4 + 1 elements.
 *  - B_pack is packed k1 x n with 4 x 1 column-major blocks and stride rsb1
 *  - C is m x n row-major with row stride rsc
 */

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to GEMM function)
  size_t m, n, k;
  int32_t alpha;
  size_t rsa1, csa1;
  size_t rsb1;
  int32_t beta;
  size_t rsc;

  // Buffer generation settings for A_pack, B_pack, C
  SKL_TEST_BUFFER(int8_t) a_pack, b_pack;
  SKL_TEST_BUFFER(int32_t) c;

  struct {
    int32_t *ref_c;
  } ctx;
} gemm_i8rcp_i8pc_i32_xsfvqdotq_t;

void gemm_i8rcp_i8pc_i32_xsfvqdotq_init(skl_test_t *t);
void gemm_i8rcp_i8pc_i32_xsfvqdotq_verify(skl_test_t *t);
void gemm_i8rcp_i8pc_i32_xsfvqdotq_test_report(skl_test_t *t);
void gemm_i8rcp_i8pc_i32_xsfvqdotq_benchmark_report(skl_test_t *t);
void gemm_i8rcp_i8pc_i32_xsfvqdotq_cleanup(skl_test_t *t);

#define GEMM_I8RCP_I8PC_I32_XSFVQDOTQ_DEFAULTS                                 \
  .a_pack = {.min = -128, .max = 127, .mode = SKL_TEST_RANDOM},                \
  .b_pack = {.min = -128, .max = 127, .mode = SKL_TEST_RANDOM},                \
  .c = {.min = -128, .max = 127, .mode = SKL_TEST_RANDOM}
