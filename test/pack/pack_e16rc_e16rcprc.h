// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Test and benchmark for Pack
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Input matrix dimensions m, n
 *  - Input matrix strides rs, cs
 *  - Block dimensions m0, n0
 *  - Output block strides rs0, cs0 (within block), rs1, cs1 (between blocks)
 *  - Padding value
 */

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Input matrix dimensions
  size_t m, n;
  // Input matrix strides
  size_t rs, cs;
  // Block dimensions
  size_t m0, n0;
  // Output block strides
  size_t rs0, cs0; // within block
  size_t rs1, cs1; // between blocks

  uint16_t pad;
  // Buffer generation settings
  SKL_TEST_BUFFER(uint16_t) src, dst;

  // Derived parameters & buffers (private to the test harness)
  struct {
    uint16_t *ref_dst;
  } ctx;
} pack_e16rc_e16rcprc_t;

#define PACK_E16RC_E16RCPRC_DEFAULTS                                           \
  .src = {.min = 0, .max = 255, .mode = SKL_TEST_RANDOM},                      \
  .dst = {.min = 0, .max = 255, .mode = SKL_TEST_RANDOM}, .pad = 0xBEEF

void pack_e16rc_e16rcprc_init(skl_test_t *t);
void pack_e16rc_e16rcprc_verify(skl_test_t *t);
void pack_e16rc_e16rcprc_test_report(skl_test_t *t);
void pack_e16rc_e16rcprc_benchmark_report(skl_test_t *t);
void pack_e16rc_e16rcprc_cleanup(skl_test_t *t);
