// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Test and benchmark for ofp conversion.
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

typedef enum { F4E2M1, F8E4M3, F8E5M2, BF16, F32 } cvt_ofp_type_t;

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to ofp conversion function)
  size_t len;
  float scale;
  bool saturation;

  // Buffer type for in,out
  cvt_ofp_type_t in_type;
  cvt_ofp_type_t out_type;

  // Memory region for in,out
  size_t in_region;
  size_t out_region;

  // Buffer pointers for in,out
  void *in;
  void *out;

  // Reference output
  float *ref;
} cvt_ofp_t;

void cvt_ofp_init(skl_test_t *t);
void cvt_ofp_verify(skl_test_t *t);
void cvt_ofp_report(skl_test_t *t);
void cvt_ofp_cleanup(skl_test_t *t);
