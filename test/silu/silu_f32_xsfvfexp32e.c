// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

/**
 * @brief Test cases for FP32 SiLU with Xsfvfexp32e extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "elementwise/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-110.f)
#define MAX (+18.f)

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_silu_52u_f32_xsfvfexp32e, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_silu_52u_f32_xsfvfexp32e, skl_silu_1u_f32_ref, MIN, MAX, 52.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_silu_f32_xsfvfexp32e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
