// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @brief Test and benchmark for ofp4 to ofp8 conversion.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Length of the input/output vector
 *  - Type of conversion
 *
 * The test will generate a random input vector and convert it to the desired
 * output format. The output is then compared to the reference implementation.
 */

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to ofp conversion function)
  size_t len;

  // Buffer pointers for in,out
  SKL_TEST_BUFFER(uint8_t) in;
  SKL_TEST_BUFFER(uint8_t) out;

  // Reference output
  uint8_t *ref;
} cvt_f4_f8_t;

void cvt_f4_f8_init(skl_test_t *t);
void cvt_f4_f8_verify(skl_test_t *t);
void cvt_f4_f8_report(skl_test_t *t);
void cvt_f4_f8_cleanup(skl_test_t *t);

#define CVT_F4_F8_DEFAULTS                                                     \
  .in = {.min = 0, .max = 255, .mode = SKL_TEST_SEQ},                          \
  .out = {.min = 0, .max = 255, .mode = SKL_TEST_SEQ}
