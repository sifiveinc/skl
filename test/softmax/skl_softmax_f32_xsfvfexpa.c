// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include "skl-test-driver.h"
#include "skl.h"
#include "softmax/softmax_f32.h"

/**
 * @brief Test cases for 1D Softmax with Xsfvfexpa extension
 */

// clang-format off
softmax_f32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  VARIANT_BENCHMARKS(xsfvfexpa),
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_TESTS(xsfvfexpa),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_softmax_f32_xsfvfexpa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_f32_t),
                                 .tests = tests};

int main(void) {
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
