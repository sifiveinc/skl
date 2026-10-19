// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve64d)
#error This file requires the Zve64d extension
#endif

/**
 * @brief Test cases for SQRT with Zve64d extension.
 */

#include "elementwise/unary_f64.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <float.h>

// clang-format off
#define SQRT_TESTS(FUNC, ERR) \
  FUNCTION_TESTS(FUNC, skl_sqrt_f64_ref, 0, DBL_MIN, ERR), \
  FUNCTION_TESTS(FUNC, skl_sqrt_f64_ref, 1, 4, ERR),       \
  FUNCTION_TESTS(FUNC, skl_sqrt_f64_ref, 0x1p1020, DBL_MAX, ERR)

unary_f64_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_sqrt_f64_zve64d, 1, 4),
    FUNCTION_BENCHMARKS(skl_sqrt_1u_f64_zve64d, 1, 4),
#endif
#if defined(SKL_ENABLE_TESTS)
    SQRT_TESTS(skl_sqrt_f64_zve64d, 0),
    SQRT_TESTS(skl_sqrt_1u_f64_zve64d, 1),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_sqrt_f64_zve64d",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f64_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
