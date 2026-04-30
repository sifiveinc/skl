// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

/**
 * @brief Test and benchmark for ofp8 to bf16 conversion.
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

typedef enum { F8E4M3, F8E5M2 } cvt_ofp8_type_t;

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to ofp conversion function)
  size_t len;

  // Buffer pointers for in,out
  SKL_TEST_BUFFER(uint8_t) in;
  SKL_TEST_BUFFER(__bf16) out;

  // OFP8 type for in
  cvt_ofp8_type_t in_type;

  // Reference output
  __bf16 *ref;
} cvt_f8_bf16_t;

void cvt_f8_bf16_init(skl_test_t *t);
void cvt_f8_bf16_verify(skl_test_t *t);
void cvt_f8_bf16_report(skl_test_t *t);
void cvt_f8_bf16_cleanup(skl_test_t *t);

#define CVT_F8_BF16_DEFAULTS                                                   \
  .in = {.min = 0, .max = 255, .mode = SKL_TEST_SEQ},                          \
  .out = {.min = (__bf16)0.0f, .max = (__bf16)1.0f, .mode = SKL_TEST_SEQ}
