// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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
  SKL_TEST_BUFFER(uint16_t) a, at;

  // Derived parameters & buffers (private to the test harness)
  struct {
    uint16_t *ref_at;
  } ctx;
} transpose_e16_t;

#define TRANSPOSE_E16_DEFAULTS                                                 \
  .a = {.min = 0, .max = UINT16_MAX, .mode = SKL_TEST_RANDOM},                 \
  .at = {.min = 0, .max = UINT16_MAX, .mode = SKL_TEST_RANDOM}

void transpose_e16_init(skl_test_t *t);
void transpose_e16_verify(skl_test_t *t);
void transpose_e16_report(skl_test_t *t);
void transpose_e16_cleanup(skl_test_t *t);
