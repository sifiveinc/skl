// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>

typedef struct {
  // Test function pointers for various steps
  skl_test_steps_t steps;

  // Configurable parameters (RMSNorm function and arguments)
  void *func;     // polymorphic pointer to function
  void *ref_func; // polymorphic pointer to reference function
  char *name;     // name of the function under test
  size_t n;       // number of elements
  float epsilon;  // small value to avoid division by zero
  size_t rsc;     // row stride of input and output

  // Buffer for input data
  SKL_TEST_BUFFER(float) src;
  SKL_TEST_BUFFER(float) weight;

  // Derived parameters & buffers (private to the test harness)
  struct {
    float *ref_dst; // reference result
    float *dst;      // result
    float max_err;
  } ctx;
} rmsnorm_f32_t;

void rmsnorm_f32_init(skl_test_t *);
void rmsnorm_f32_execute(skl_test_t *);
void rmsnorm_f32_verify(skl_test_t *);
void rmsnorm_f32_report(skl_test_t *);
void rmsnorm_f32_cleanup(skl_test_t *);

#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = rmsnorm_f32_init,                                                \
      .report = rmsnorm_f32_report,                                            \
      .cleanup = rmsnorm_f32_cleanup,                                          \
  }

#define TEST                                                                   \
  .src.mode = SKL_TEST_RANDOM,                                                 \
  .weight.mode = SKL_TEST_RANDOM,                                              \
  BASIC_STEPS,                                                                 \
  .steps.verify = rmsnorm_f32_verify

#define BENCHMARK                                                              \
  .src.mode = SKL_TEST_SEQ,                                                    \
  .weight.mode = SKL_TEST_SEQ,                                                 \
  BASIC_STEPS,                                                                 \
  .steps.verify = NULL

// clang-format off
#define BASE_PARAMS(VAR)                                                       \
  .func = (void *)skl_rmsnorm_f32_##VAR,                                       \
  .ref_func = (void *)skl_rmsnorm_f32_ref,                                        \
  .name = "skl_rmsnorm_f32_" #VAR,                                             \
  .steps.execute = rmsnorm_f32_execute

#define VARIANT_BENCHMARKS(VAR)                                                \
  {BENCHMARK, BASE_PARAMS(VAR), .n = 65536, .src.min = -1, .src.max = 1., .weight.min = -1, .weight.max = 1., .rsc = 4096, .epsilon = 1e-09, .ctx.max_err = 100.f}                       \

#define VARIANT_TESTS(VAR)                                                     \
  {TEST, BASE_PARAMS(VAR), .n = 1024, .src.min = -1, .src.max = 1., .weight.min = -1, .weight.max = 1., .rsc = 128, .epsilon = 1e-09, .ctx.max_err = 100.f},                        \
  {TEST, BASE_PARAMS(VAR), .n = 2048, .src.min = -1, .src.max = 1., .weight.min = -1, .weight.max = 1., .rsc = 256, .epsilon = 1e-09, .ctx.max_err = 100.f},                        \
  {TEST, BASE_PARAMS(VAR), .n = 4096, .src.min = -1, .src.max = 1., .weight.min = -1, .weight.max = 1., .rsc = 512, .epsilon = 1e-09, .ctx.max_err = 100.f},                        \
  {TEST, BASE_PARAMS(VAR), .n = 8192, .src.min = -1, .src.max = 1., .weight.min = -1, .weight.max = 1., .rsc = 1024, .epsilon = 1e-09, .ctx.max_err =100.f}


// clang-format on
