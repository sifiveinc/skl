// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-test-driver.h"
#include "skl.h"
#include "softmax/softmax_bf16.h"

/**
 * @brief Test cases for 1D BF16 Softmax with Zve32f extension
 */

// clang-format off
softmax_bf16_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  VARIANT_BENCHMARKS(zve32f),
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_TESTS(zve32f),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_softmax_bf16_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_bf16_t),
                                 .tests = tests};

int main(void) {
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
