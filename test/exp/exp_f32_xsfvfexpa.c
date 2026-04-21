// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

/**
 * @brief Test cases for Exponential with Xsfvfexpa extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "math/unary_f32.h"
#include "skl-ref.h" // NOLINT(misc-include-cleaner)
#include "skl-test-driver.h"
#include "skl.h"

#define MIN (-104.f)
#define MAX (+89.f)

#define FUNCTION_TESTS(FUN, MIN, MAX, ULP)                                     \
  {TEST,           .ref_func = skl_exp_f32_ref, .func = (FUN), .a.min = (MIN), \
   .a.max = (MAX), .ctx.max_err = (ULP)}

#define FUNCTION_BENCHMARKS(FUN, MIN, MAX)                                     \
  {BENCH, .func = (FUN), .a.min = (MIN), .a.max = (MAX)}

unary_f32_t tests[] = {
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX),
    FUNCTION_BENCHMARKS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa,
                        -0x1.5d639ep6f, +89.f),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX, 1.0f),
    FUNCTION_TESTS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa, -0x1.5d639ep6f,
                   +89.f, 1.0002f),
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f32_xsfvfexpa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
