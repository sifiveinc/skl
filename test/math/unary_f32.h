// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Test and benchmark for FP32 unary functions f(x) = y.
 *
 * This test uses a table-driven approach where test configurations
 * are defined in the `tests` array.  Each test specifies:
 *  - Array dimension
 *  - Expected maximum numerical error of the result
 *  - The kernel function to test
 */

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

// Type of SKL unary functions
typedef void (*unary_func_t)(float *, const float *, size_t);

// Type of SKL unary function harnesses
typedef struct {
  // Test function pointers for various steps
  skl_test_steps_t steps;

  // Configurable parameters
  unary_func_t ref_func; // The reference function
  unary_func_t func;     // The function to test

  // Buffer generation settings
  SKL_TEST_BUFFER(float) a;

  // Derived parameters and buffers (private to the test harness)
  struct {
    float *b;      // result buffer
    float *ref;    // reference buffer
    float max_err; // ulp tolerance
  } ctx;
} unary_f32_t;

void unary_f32_init(skl_test_t *t);
void unary_f32_execute(skl_test_t *t);
void unary_f32_verify(skl_test_t *t);
void unary_f32_report(skl_test_t *t);
void unary_f32_cleanup(skl_test_t *t);

#define UNARY_F32_TEST_DEFAULTS .a = {.len = 65536, .mode = SKL_TEST_RANDOM}
#define UNARY_F32_BENCH_DEFAULTS .a = {.len = 1024, .mode = SKL_TEST_SEQ}
#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = unary_f32_init,                                                  \
      .execute = unary_f32_execute,                                                      \
      .cleanup = unary_f32_cleanup,                                            \
      .report = unary_f32_report,                                              \
  }

#define TEST                                                                   \
  UNARY_F32_TEST_DEFAULTS, BASIC_STEPS, .steps.warmup = NULL,                  \
                                        .steps.verify = unary_f32_verify

#define BENCH                                                                  \
  UNARY_F32_BENCH_DEFAULTS, BASIC_STEPS, .steps.warmup = unary_f32_execute,              \
                                         .steps.verify = NULL
