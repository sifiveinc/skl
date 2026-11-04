// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

/**
 * @brief Test cases for SQRT with Zvfh extension.
 */

#include "elementwise/unary_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

// clang-format off
#define SQRT_TESTS(FUNC, ERR) \
  FUNCTION_TESTS(FUNC, skl_sqrt_f16_ref, 0, 0x1p-14, ERR), \
  FUNCTION_TESTS(FUNC, skl_sqrt_f16_ref, 1, 4, ERR),       \
  FUNCTION_TESTS(FUNC, skl_sqrt_f16_ref, 0x1p13, 0x1.ffcp15, ERR)

unary_f16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_sqrt_f16_zvfh, 1, 4),
    FUNCTION_BENCHMARKS(skl_sqrt_1u_f16_zvfh, 1, 4),
#endif
#if defined(SKL_ENABLE_TESTS)
    SQRT_TESTS(skl_sqrt_f16_zvfh, 0),
    SQRT_TESTS(skl_sqrt_1u_f16_zvfh, 1),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_sqrt_f16_zvfh",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
