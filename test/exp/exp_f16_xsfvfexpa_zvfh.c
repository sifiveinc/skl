// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

/**
 * @brief Test cases for FP16 Exponential with Xsfvfexpa and Zvfh extensions.
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
    FUNCTION_BENCHMARKS(skl_exp_1u_f16_xsfvfexpa_zvfh, -1, 1),
    FUNCTION_BENCHMARKS(skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh, -1, 1),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u_f16_xsfvfexpa_zvfh, skl_exp_f16_ref, MIN, MAX,
                   1.0f),
    FUNCTION_TESTS(skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh, skl_exp_f16_ref,
                   -0x1.33p3f, MAX, 2.0f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f16_xsfvfexpa_zvfh",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f16_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
