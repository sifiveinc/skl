// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameter (Softmax function and arguments)
  void *func;  // polymorphic pointer to function
  char *name;  // name of the function under test
  size_t m, n; // row and column count
  float beta;
  size_t rsa, rss; // row strides of input and output

  // Buffer for input data
  SKL_TEST_BUFFER(float) a;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *S; // reference result
    float *s;  // result
    float max_err;
  } ctx;
} softmax_f32_t;

void softmax_f32_init(skl_test_t *);
void softmax_f32_verify(skl_test_t *);
void softmax_f32_report(skl_test_t *);
void softmax_f32_cleanup(skl_test_t *);

#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = softmax_f32_init,                                                \
      .report = softmax_f32_report,                                            \
      .cleanup = softmax_f32_cleanup,                                          \
  }

#define TEST                                                                   \
  .a.mode = SKL_TEST_RANDOM, BASIC_STEPS, .steps.verify = softmax_f32_verify

#define BENCHMARK .a.mode = SKL_TEST_SEQ, BASIC_STEPS, .steps.verify = NULL
