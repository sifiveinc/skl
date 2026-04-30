// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

/**
 * @brief Test and benchmark for f32 to ofp8 conversion.
 *
 * This test uses a table-driven approach where test configurations are defined
 * in the `tests` array. Each test specifies:
 *  - Length of the input/output vector
 *  - Saturation flag
 *  - Scaling factor
 *  - Type of conversion
 *
 * The test will generate a random input vector and convert it to the desired
 * output format. The output is then compared to the reference implementation.
 */

#include "skl-test-driver.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { F8E4M3, F8E5M2 } cvt_ofp8_type_t;

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to ofp conversion function)
  size_t len;
  float scale;
  bool saturation;

  // Buffer pointers for in,out
  SKL_TEST_BUFFER(float) in;
  SKL_TEST_BUFFER(uint8_t) out;

  // OFP8 type for out
  cvt_ofp8_type_t out_type;

  // Reference output
  uint8_t *ref;
} cvt_f32_f8_t;

void cvt_f32_f8_init(skl_test_t *t);
void cvt_f32_f8_verify(skl_test_t *t);
void cvt_f32_f8_report(skl_test_t *t);
void cvt_f32_f8_cleanup(skl_test_t *t);

#define CVT_F32_F8_DEFAULTS                                                    \
  .in = {.min = -512.0f, .max = 512.0f, .mode = SKL_TEST_RANDOM},              \
  .out = {.min = 0, .max = 255, .mode = SKL_TEST_SEQ}
