// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

/**
 * @brief Test cases for SQRT with Zve32f extension.
 */

#include "elementwise/unary_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

// clang-format off
#define SQRT_TESTS(FUNC, ERR) \
  FUNCTION_TESTS(FUNC, skl_sqrt_bf16_ref, 0, 0x1p-118, ERR), \
  FUNCTION_TESTS(FUNC, skl_sqrt_bf16_ref, 1, 0x1p15, ERR),   \
  FUNCTION_TESTS(FUNC, skl_sqrt_bf16_ref, 0x1p112, 0x1.fep127, ERR)

unary_bf16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_sqrt_bf16_zve32f, 1, 4),
    FUNCTION_BENCHMARKS(skl_sqrt_2u_bf16_zve32f, 1, 4),
#endif
#if defined(SKL_ENABLE_TESTS)
    SQRT_TESTS(skl_sqrt_bf16_zve32f, 0),
    SQRT_TESTS(skl_sqrt_2u_bf16_zve32f, 2),
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_sqrt_bf16_zve32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_bf16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
