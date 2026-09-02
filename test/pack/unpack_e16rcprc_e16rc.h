// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for Unpack
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Input block dimensions m0, n0
 *  - Input block strides rs0, cs0 (within block), rs1, cs1 (between blocks)
 *  - Output matrix dimensions m, n
 *  - Output matrix strides rs, cs
 */

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Input block dimensions
  size_t m0, n0;
  // Input block strides
  size_t rs0, cs0; // within block
  size_t rs1, cs1; // between blocks
  // Output matrix dimensions
  size_t m, n;
  // Output matrix strides
  size_t rs, cs;

  // Buffer generation settings
  SKL_TEST_BUFFER(uint16_t) src, dst;

  // Derived parameters & buffers (private to the test harness)
  struct {
    uint16_t *ref_dst;
  } ctx;
} unpack_e16rcprc_e16rc_t;

#define UNPACK_E16RCPRC_E16RC_DEFAULTS                                         \
  .src = {.min = 0, .max = 0XFFFF, .mode = SKL_TEST_RANDOM},                   \
  .dst = {.min = 0, .max = 0XFFFF, .mode = SKL_TEST_RANDOM}

void unpack_e16rcprc_e16rc_init(skl_test_t *t);
void unpack_e16rcprc_e16rc_verify(skl_test_t *t);
void unpack_e16rcprc_e16rc_test_report(skl_test_t *t);
void unpack_e16rcprc_e16rc_benchmark_report(skl_test_t *t);
void unpack_e16rcprc_e16rc_cleanup(skl_test_t *t);
