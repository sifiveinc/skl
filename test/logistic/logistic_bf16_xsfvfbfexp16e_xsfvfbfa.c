// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfexp16e) || !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfexp16e and Xsfvfbfa extensions
#endif

/**
 * @brief Test cases for BF16 Exponential with Xsfvfbfexp16e and Xsfvfbfa
 * extensions.
 */

#include "elementwise/unary_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-0x1.74p6f)
#define MAX (+0x1.90p2f)

unary_bf16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_logistic_5u_bf16_xsfvfbfexp16e_xsfvfbfa, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_logistic_5u_bf16_xsfvfbfexp16e_xsfvfbfa,
                   skl_logistic_1u_bf16_ref, MIN, MAX, 5.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_logistic_bf16_xsfvfbfexp16e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_bf16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
