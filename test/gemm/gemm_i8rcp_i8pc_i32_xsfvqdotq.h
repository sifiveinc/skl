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

  // Buffer generation settings for A, B, C
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
