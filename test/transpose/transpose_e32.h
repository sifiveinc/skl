// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  size_t m, n;
  size_t rsa, rsat;

  // Buffer generation settings for a, at
  SKL_TEST_BUFFER(uint32_t) a, at;

  // Derived parameters & buffers (private to the test harness)
  struct {
    uint32_t *ref_at;
  } ctx;
} transpose_e32_t;

#define TRANSPOSE_E32_DEFAULTS                                                 \
  .a = {.min = 0, .max = UINT32_MAX, .mode = SKL_TEST_RANDOM},                 \
  .at = {.min = 0, .max = UINT32_MAX, .mode = SKL_TEST_RANDOM}

void transpose_e32_init(skl_test_t *t);
void transpose_e32_verify(skl_test_t *t);
void transpose_e32_report_test(skl_test_t *t);
void transpose_e32_report_benchmark(skl_test_t *t);
void transpose_e32_cleanup(skl_test_t *t);
