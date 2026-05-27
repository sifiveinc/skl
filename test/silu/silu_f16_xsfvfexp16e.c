// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

/**
 * @brief Test cases for FP16 SiLU with Xsfvfexp16e extension.
 *
 * This test uses the unary_f16 harness.
 */

#include "elementwise/unary_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-21.f)
#define MAX (+9.f)

unary_f16_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_silu_31u_f16_xsfvfexp16e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_silu_31u_f16_xsfvfexp16e, skl_silu_1u_f16_ref, MIN, MAX, 31.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_silu_f16_xsfvfexp16e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
