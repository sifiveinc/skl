// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-test-driver.h"
#include "skl.h"
#include "softmax/softmax_f32.h"

/**
 * @brief Test cases for 2D Softmax with Zve32f extension
 */

// clang-format off
softmax_f32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  VARIANT_2D_BENCHMARKS(zve32f),
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_2D_TESTS(zve32f),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_softmax_f32r_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_f32_t),
                                 .tests = tests};

int main(void) {
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
