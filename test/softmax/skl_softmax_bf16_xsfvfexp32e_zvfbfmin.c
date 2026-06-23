// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfexp32e and Zvfbfmin extensions
#endif

#include "skl-test-driver.h"
#include "skl.h"
#include "softmax/softmax_bf16.h"

/**
 * @brief Test cases for 1D BF16 Softmax with Xsfvfexp32e and Zvfbfmin
 * extensions
 */

// clang-format off
softmax_bf16_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  VARIANT_BENCHMARKS(xsfvfexp32e_zvfbfmin),
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_TESTS(xsfvfexp32e_zvfbfmin),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name =
                                     "skl_softmax_bf16_xsfvfexp32e_zvfbfmin",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_bf16_t),
                                 .tests = tests};

int main(void) {
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
