// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#include "skl-test-driver.h"
#include "skl.h"
#include "softmax/softmax_bf16.h"

/**
 * @brief Test cases for 1D Softmax with Xsfvfbfa extension
 */

// clang-format off
softmax_bf16_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  VARIANT_BENCHMARKS(xsfvfbfa),
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_TESTS(xsfvfbfa),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_softmax_bf16_xsfvfbfa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_bf16_t),
                                 .tests = tests};

int main(void) {
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
