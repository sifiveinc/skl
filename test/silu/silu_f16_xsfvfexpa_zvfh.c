// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

/**
 * @brief Test cases for FP16 SiLU with Xsfvfexpa and Zvfh extensions.
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
    FUNCTION_BENCHMARKS(skl_silu_9u_f16_xsfvfexpa_zvfh, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_silu_9u_f16_xsfvfexpa_zvfh, skl_silu_1u_f16_ref, MIN, MAX, 9.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_silu_f16_xsfvfexpa_zvfh",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
