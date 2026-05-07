// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

/**
 * @brief Test cases for FP16 Exponential with Xsfvfexp16e extension.
 *
 * This test uses the unary_f16 harness.
 */

#include "elementwise/unary_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-18.f)
#define MAX (+12.f)

unary_f16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e, -1, 1),
    FUNCTION_BENCHMARKS(skl_exp_3p16u_f16_xsfvfexp16e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e, skl_exp_f16_ref,
                   -0x1.ffcp2f, 0x1.ffcp2f, 2.0f),
    FUNCTION_TESTS(skl_exp_3p16u_f16_xsfvfexp16e, skl_exp_f16_ref, MIN, MAX,
                   4.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f16_xsfvfexp16e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
