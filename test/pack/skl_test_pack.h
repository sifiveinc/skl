// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @file skl_test_pack.h
 * @brief Functions shared among the Pack test harnesses.
 */

#pragma once

#include "skl-test-driver.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Print out the matrix dimensions and strides for a pack test.
 *
 * @param t - Test context.
 * @param m - Number of rows in the matrix.
 * @param n - Number of columns in the matrix.
 * @param rs - Row stride in the matrix.
 * @param cs - Column stride in the matrix.
 * @param m0 - Number of rows in each block of the packed matrix.
 * @param n0 - Number of columns in each block of the packed matrix.
 * @param rs0 - Row stride within each block of the packed matrix in elements.
 * @param cs0 - Column stride within each block of the packed matrix in
 * elements.
 * @param rs1 - Row stride between blocks of the packed matrix in elements.
 * @param cs1 - Column stride between blocks of the packed matrix in elements.
 */
static inline void pack_rc_rcprc_report_param(skl_test_t *t, size_t m, size_t n,
                                              size_t rs, size_t cs, size_t m0,
                                              size_t n0, size_t rs0, size_t cs0,
                                              size_t rs1, size_t cs1) {

  size_t m1 = (m + m0 - 1) / m0;
  size_t n1 = (n + n0 - 1) / n0;
#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("M: %zu, N: %zu\n", m, n);
  INFO("M0: %zu, N0: %zu\n", m0, n0);
  INFO("M1: %zu, N1: %zu\n", m1, n1);
  INFO("RS: %zu, CS: %zu\n", rs, cs);
  INFO("RS0: %zu, CS0: %zu\n", rs0, cs0);
  INFO("RS1: %zu, CS1: %zu\n", rs1, cs1);
  if (rs == 1) {
    if (rs0 == 1) {
      INFO("%s", "=> COPY case!\n");
    } else if (cs0 == 1) {
      INFO("%s", "=> TRANSPOSE case!\n");
    } else {
      INFO("%s", "=> GENERAL case!\n");
    }
  } else if (cs == 1) {
    if (rs0 == 1) {
      INFO("%s", "=> TRANSPOSE case!\n");
    } else if (cs0 == 1) {
      INFO("%s", "=> COPY case!\n");
    } else {
      INFO("%s", "=> GENERAL case!\n");
    }
  } else {
    INFO("%s", "=> GENERAL case!\n");
  }
#undef INFO
}

/**
 * @brief Print out performance information for a pack test.
 *
 * @param t - Test context.
 * @param warmup - Pointer to the test harness's warmup function.
 * @param output_elements - Number of elements in the output matrix.
 */
static inline void pack_rc_rcprc_report_perf(skl_test_t *t,
                                             void (*warmup)(skl_test_t *),
                                             size_t output_elements) {
  uint64_t cycles = t->counters.cycles;
  float epc = (float)output_elements / (float)cycles;
#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("Warmup: %s\n", warmup ? "yes" : "no");
  INFO("Cycles: %" PRIu64 "\n", t->counters.cycles);
  INFO("Instructions: %" PRIu64 "\n", t->counters.instret);
  INFO("Elements/Cycle: %f\n", epc);
#undef INFO
}
