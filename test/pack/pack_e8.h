// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#ifndef SKL_TEST_PACK_PACK_E8_H_
#define SKL_TEST_PACK_PACK_E8_H_

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Input matrix dimensions
  size_t m, n;
  // Block dimensions
  size_t m0, n0;
  // Input matrix strides
  size_t rs, cs;
  // Output block strides
  size_t rs0, cs0; // within block
  size_t rs1, cs1; // between blocks

  uint8_t pad;
  // Buffer generation settings
  SKL_TEST_BUFFER(uint8_t) src, dst;

  // Derived parameters & buffers (private to the test harness)
  struct {
    uint8_t *ref_dst;
    size_t dst_size;
  } ctx;
} pack_e8_t;

#define PACK_E8_DEFAULTS                                                       \
  .src = {.min = 0, .max = 255, .mode = SKL_TEST_RANDOM},                      \
  .dst = {.min = 0, .max = 255, .mode = SKL_TEST_RANDOM}, .pad = 255

void pack_e8_init(skl_test_t *t);
void pack_e8_verify(skl_test_t *t);
void pack_e8_param_report(skl_test_t *t);
void pack_e8_bench_report(skl_test_t *t);
void pack_e8_cleanup(skl_test_t *t);

#endif // SKL_TEST_PACK_PACK_E8_H_
