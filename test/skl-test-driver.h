// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @file skl-test-driver.h
 * @brief SKL test driver framework for benchmarking and verification
 *
 * This header defines the test driver framework used by SKL kernel tests.
 * The framework provides a standardized interface for test harnesses to
 * initialize, execute, verify, and report on kernel performance.
 *
 * The driver handles memory allocation, test data initialization, performance
 * measurement (cycles and instructions), and result reporting. Test harnesses
 * implement the required callback functions to integrate with the driver:
 * - skl_test_init
 * - skl_test_execute
 * - skl_test_verify
 * - skl_test_cleanup
 * - skl_test_report
 *
 * A test suite is a collection of related tests that share a common harness.
 * The harness defines a test configuration structure, and a suite is an array
 * of test configurations.
 *
 * This interface is intended to allow consumers of SKL to adapt the test suite
 * to their own environments by replacing the short driver program with one that
 * uses the appropriate I/O, data generation, and measurement methods for their
 * environment.
 */

#pragma once

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//----- Test definitions -----

// Forward declaration to resolve circular dependency
typedef struct skl_test_t skl_test_t;

/**
 * @brief Test suite metadata.
 *
 * Describes a collection of related tests.
 */
typedef struct {
  const char *name; /**< Name of the test suite */
  size_t num_tests; /**< Number of tests in the suite */
  size_t test_size; /**< Size of the test-specific data structure */
  void *tests;      /**< Array of test-specific data structures */
} skl_test_suite_t;

/**
 *  @brief Function pointers determining each step ran by the driver.
 */
typedef struct {
  /**
   * @brief Test initialization callback.
   *
   * @param t - Test context.
   *
   * This step should prepare the test for execution, including allocating and
   * initializing all appropriate buffers.
   *
   * This function should set the init_status member of a skl_test_step_status_t
   * to indicate its pass/fail result.
   *
   * This step is optional and the callback can be set to NULL to skip its call.
   */
  void (*init)(skl_test_t *t);

  /**
   * @brief Test warmup callback.
   *
   * @param t - Test context.
   *
   * This step can be used as a warmup before the execute step, such as for
   * making certain data cache-resident.
   *
   * This function should set the warmup_status member of a
   * skl_test_step_status_t to indicate its pass/fail result.
   *
   * This step is optional and the callback can be set to NULL to skip its call.
   */
  void (*warmup)(skl_test_t *t);

  /**
   * @brief Test execution callback.
   *
   * @param t - Test context.
   *
   * This step should perform the actual test execution, which is typically
   * calling into the underlying kernel with appropriate values.
   *
   * This function should set the execute_status member of a
   * skl_test_step_status_t to indicate its pass/fail result.
   *
   * This step is mandatory and the callback cannot be NULL.
   */
  void (*execute)(skl_test_t *t);

  /**
   * @brief Test verification callback.
   *
   * @param t - Test context.
   *
   * This step should verify that execution performed as expected, such as for
   * correctness testing.
   *
   * This function should set the verify_status member of a
   * skl_test_step_status_t to indicate its pass/fail result.
   *
   * This step is optional and the callback can be set to NULL to skip its call.
   */
  void (*verify)(skl_test_t *t);

  /**
   * @brief Test reporting callback.
   *
   * @param t - Test context.
   *
   * This step should log any desired information using the skl_test_driver_log
   * function, such as test parameters or performance results.
   *
   * This function should set the report_status member of a
   * skl_test_step_status_t to indicate its pass/fail result.
   *
   * This step is optional and the callback can be set to NULL to skip its call.
   */
  void (*report)(skl_test_t *t);

  /**
   * @brief Test cleanup callback.
   *
   * @param t - Test context.
   *
   * This step should perform any cleanup required to end the test, such as
   * freeing any buffers allocated in a previous step.
   *
   * This function should set the cleanup_status member of a
   * skl_test_step_status_t to indicate its pass/fail result.
   *
   * This step is optional and the callback can be set to NULL to skip its call.
   */
  void (*cleanup)(skl_test_t *t);

} skl_test_steps_t;

/**
 * @brief Test status values.
 */
typedef enum {
  SKL_TEST_PASS = 0, /**< Test passed */
  SKL_TEST_FAIL = 1, /**< Test failed */
} skl_test_status_t;

/**
 * @brief Status values for each step of skl_test_steps_t.
 */
typedef struct {
  skl_test_status_t init_status;
  skl_test_status_t warmup_status;
  skl_test_status_t execute_status;
  skl_test_status_t verify_status;
  skl_test_status_t report_status;
  skl_test_status_t cleanup_status;
} skl_test_step_status_t;

/**
 * @brief Performance counter values.
 *
 * Contains cycle and instruction counts measured during test execution.
 */
typedef struct {
  uint64_t cycles;  /**< Number of CPU cycles elapsed */
  uint64_t instret; /**< Number of instructions retired */
} skl_test_counters_t;

/**
 * @brief Test context structure.
 *
 * Contains all state for a single test execution, including driver
 * configuration and harness-specific data. The driver initializes this
 * structure and passes it to all harness callback functions.
 */
struct skl_test_t {
  /**
   * @brief Harness-specific data.
   *
   * Pointer set by driver, with data accessible and modifiable by harness. The
   * first member of the underlying struct must be of type `skl_test_steps_t`.
   */
  void *harness;

  /**
   * @brief Name of the suite this test is part of.
   *
   * Set by driver before init.
   */
  const char *suite_name;

  /**
   * @brief Identifier for this test within a suite.
   *
   * Set by driver before init.
   */
  unsigned int id;

  /**
   * @brief Logging verbosity level.
   *
   *  Set by driver before init.
   */
  int log_level;

  /**
   * @brief Performance counters.
   *
   * Set by driver after execute.
   */
  skl_test_counters_t counters;

  /**
   * @brief Status of each step of the test.
   *
   * Set by harness and checked by driver between each step.
   */
  skl_test_step_status_t status;
};

//----- Functions provided by the driver -----

/**
 * @brief Run a test suite.
 *
 * @param suite - Test suite to run
 * @return 0 on success, non-zero on error
 */
int skl_test_driver_run_suite(skl_test_suite_t *suite);

/**
 * @brief Allocate memory for a test buffer.
 *
 * @param region - Memory region to allocate from (defined by driver, must
 * support 0)
 * @param size - Size of memory to allocate in bytes
 * @return Pointer to allocated memory
 */
void *skl_test_driver_alloc(size_t region, size_t size);

/**
 * @brief Free memory for a test buffer allocated with skl_test_driver_alloc.
 *
 * @param region - Memory region to free from (defined by driver, must support
 * 0)
 * @param ptr - Pointer to memory to free
 */
void skl_test_driver_free(size_t region, void *ptr);

/**
 * @brief Update performance counters.
 *
 * @param counters - Performance counter structure to update
 *
 * Reads the current values of the cycle and instruction counters and
 * updates counters with the delta since the last update.
 */
void skl_test_driver_update_counters(skl_test_counters_t *counters);

/**
 * @brief Log an informational message.
 *
 * @param t - Test context
 * @param stream - Output stream to log the message to.
 * @param fmt - Printf-style format string
 * @param ... - Format arguments
 *
 * Outputs a log message to stdout with test ID prefix.
 * Supports printf-style formatting.
 */
void skl_test_driver_log(skl_test_t *t, FILE *stream, const char *fmt, ...);

/**
 * @brief Returns the current status of the overall test.
 *
 * @param t - Test context
 * @return Returns SKL_TEST_PASS if all values in t->status are SKL_TEST_PASS,
 * and SKL_TEST_FAIL otherwise.
 */
skl_test_status_t skl_test_driver_status(skl_test_t *t);

//----- Utility macros for use in tests -----

/**
 * @brief Free a test buffer.
 *
 * @param T - Test context pointer
 * @param BUF - Buffer to free
 *
 * Frees a buffer allocated with SKL_TEST_BUF_CREATE using the configured
 * memory allocator. Logs the deallocation at log level SKL_TEST_LOG_DEBUG.
 */
#define SKL_TEST_BUF_FREE(T, BUF)                                              \
  do {                                                                         \
    skl_test_driver_free((BUF)->region, (BUF)->data);                          \
    SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG, "freed %s at %p\n", #BUF,            \
                 (BUF)->data);                                                 \
  } while (0)

// Helper macros for generating random values (used internally by
// SKL_TEST_BUF_CREATE)

#define SKL_TEST_BUF_RANDOM_FLOAT_(TYPE, MIN, MAX, LEN)                        \
  ((TYPE)(TYPE)((MIN) + ((float)rand() / (float)RAND_MAX) * ((MAX) - (MIN))))

#define SKL_TEST_BUF_RANDOM___bf16(TYPE, MIN, MAX, LEN)                        \
  SKL_TEST_BUF_RANDOM_FLOAT_(TYPE, MIN, MAX, LEN)

#define SKL_TEST_BUF_RANDOM__Float16(TYPE, MIN, MAX, LEN)                      \
  SKL_TEST_BUF_RANDOM_FLOAT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_float(TYPE, MIN, MAX, LEN)                         \
  SKL_TEST_BUF_RANDOM_FLOAT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_double(TYPE, MIN, MAX, LEN)                        \
  SKL_TEST_BUF_RANDOM_FLOAT_(TYPE, MIN, MAX, LEN)

#define SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)                          \
  ((TYPE)(rand() % ((MAX) - (MIN) + 1) + (MIN)))

#define SKL_TEST_BUF_RANDOM_int8_t(TYPE, MIN, MAX, LEN)                        \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_uint8_t(TYPE, MIN, MAX, LEN)                       \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_int16_t(TYPE, MIN, MAX, LEN)                       \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_uint16_t(TYPE, MIN, MAX, LEN)                      \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_int32_t(TYPE, MIN, MAX, LEN)                       \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_uint32_t(TYPE, MIN, MAX, LEN)                      \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_int64_t(TYPE, MIN, MAX, LEN)                       \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)
#define SKL_TEST_BUF_RANDOM_uint64_t(TYPE, MIN, MAX, LEN)                      \
  SKL_TEST_BUF_RANDOM_INT_(TYPE, MIN, MAX, LEN)

/**
 * @brief Test buffer descriptor.
 *
 * Describes a buffer of test data with initialization parameters.
 * Test harnesses use SKL_TEST_BUFFER() to define buffer members in their
 * test configuration structures. These can then be allocated and initialized by
 * the driver using SKL_TEST_BUF_CREATE.
 */
#define SKL_TEST_BUFFER(TYPE)                                                  \
  struct {                                                                     \
    TYPE *data;                                                                \
    size_t len;                                                                \
    skl_test_init_mode_t mode;                                                 \
    TYPE min, max;                                                             \
    const TYPE *static_data;                                                   \
    size_t static_data_len;                                                    \
    size_t region;                                                             \
  }

/**
 * @brief Test data initialization modes.
 *
 * Specifies how test buffers should be initialized by SKL_TEST_BUF_CREATE.
 */
typedef enum {
  SKL_TEST_RANDOM, /**< Initialize with random values in [min, max] */
  SKL_TEST_STATIC, /**< Initialize from static_data array */
  SKL_TEST_SEQ,    /**< Initialize with sequential values from min to max */
} skl_test_init_mode_t;

/**
 * @brief Allocate and initialize a test buffer.
 *
 * @param T - Test context pointer
 * @param TYPE - Element type (e.g., float, int32_t)
 * @param BUF - SKL_TEST_BUFFER() to receive the allocated buffer
 *
 * Allocates a buffer and initializes it according to the parameters in BUF.
 * The initialization mode determines how the buffer is filled:
 * - SKL_TEST_RANDOM: Fill with random values in [min, max]
 * - SKL_TEST_STATIC: Fill from static_data array (cycling if needed)
 * - SKL_TEST_SEQ: Fill with sequential values from min to max
 *
 * Logs allocation at log level SKL_TEST_LOG_DEBUG.
 */
#define SKL_TEST_BUF_CREATE(T, TYPE, BUF)                                      \
  do {                                                                         \
    (BUF)->data = (TYPE *)skl_test_driver_alloc((BUF)->region,                 \
                                                (BUF)->len * sizeof(TYPE));    \
    SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG, "allocated %s of size %zd at %p \n", \
                 #BUF, (BUF)->len, (BUF)->data);                               \
    switch ((BUF)->mode) {                                                     \
    case SKL_TEST_RANDOM:                                                      \
      SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG, "random values in [%f, %f]\n",     \
                   (double)(BUF)->min, (double)(BUF)->max);                    \
      for (size_t i = 0; i < (BUF)->len; ++i) {                                \
        (BUF)->data[i] = SKL_TEST_BUF_RANDOM_##TYPE(TYPE, (BUF)->min,          \
                                                    (BUF)->max, (BUF)->len);   \
      }                                                                        \
      break;                                                                   \
    case SKL_TEST_STATIC:                                                      \
      SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG,                                    \
                   "static data from %p (len = %zd)\n", (BUF)->static_data,    \
                   (BUF)->static_data_len);                                    \
      if ((BUF)->static_data) {                                                \
        for (size_t i = 0; i < (BUF)->len; ++i) {                              \
          (BUF)->data[i] = (BUF)->static_data[i % (BUF)->static_data_len];     \
        }                                                                      \
      } else {                                                                 \
        SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG, "no static data\n");             \
      }                                                                        \
      break;                                                                   \
    case SKL_TEST_SEQ: {                                                       \
      TYPE step = (BUF)->len ? ((BUF)->max - (BUF)->min) / (BUF)->len : 0;     \
      SKL_TEST_LOG((T), SKL_TEST_LOG_DEBUG,                                    \
                   "sequential values in [%f, %f] by %f\n",                    \
                   (double)(BUF)->min, (double)(BUF)->max, (double)step);      \
      for (size_t i = 0; i < (BUF)->len; ++i) {                                \
        (BUF)->data[i] = (BUF)->min + step * i;                                \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    }                                                                          \
  } while (0)

//----- Utility functions for use in tests -----

/**
 * @brief Calculate absolute error.
 *
 * @param x The test value
 * @param r The reference value
 * @param emin One greater than the smallest normal exponent.
 * @param p Bits of precision, including the implicit leading one.
 * @returns The distance between x and r in units of `ulp(r)`.
 *
 * @note This function assumes `x` and `r` are of the same floating-point type.
 */
static inline float skl_abs_error_ulp(float x, float r, int emin, int p) {
  if (x == r || (isnan(x) && isnan(r)))
    return 0.f;
  if (isinf(r) || isnan(x) || isnan(r))
    return INFINITY;

  int e;
  float d = fabsf(x - r);
  frexpf(r, &e);
  e = (e < emin || r == 0.f) ? emin - p : e - p;
  return ldexpf(d, -e);
}

/**
 * @brief Calculate absolute error for float data.
 *
 * @param x The test value
 * @param r The reference value
 * @returns The distance between x and r in units of `ulp(r)`.
 *
 * This is a specialization of skl_abs_error_ulp.
 */
static inline float skl_abs_error_ulp_f32(float x, float r) {
  return skl_abs_error_ulp(x, r, FLT_MIN_EXP, FLT_MANT_DIG);
}

/**
 * @brief Calculate absolute error for _Float16 data.
 * @details @copydetails skl_abs_error_ulp_f32
 */
static inline float skl_abs_error_ulp_f16(_Float16 x, _Float16 r) {
  return skl_abs_error_ulp(x, r, -13, 11);
}

/**
 * @brief Calculate absolute error for __bf16 data.
 * @details @copydetails skl_abs_error_ulp_f32
 */
static inline float skl_abs_error_ulp_bf16(__bf16 x, __bf16 r) {
  return skl_abs_error_ulp(x, r, FLT_MIN_EXP, 8);
}

/**
 * @brief Determine maximum error for float data
 *
 * @param res Array of test result values
 * @param ref Array of test reference values
 * @param len Length of result and reference arrays
 * @returns The maximum error found
 *
 * This function calculates and returns the absolute error in ulps
 * between res[i] and ref[i] for i in [0, len).
 */
static inline float skl_error_ulp_f32(const float *res, const float *ref,
                                      size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_f32(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  return max;
}

/**
 * @brief Determine maximum error for _Float16 data
 * @details @copydetails skl_check_ulp_f32
 */
static inline float skl_error_ulp_f16(const _Float16 *res, const _Float16 *ref,
                                      size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_f16(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  return max;
}

/**
 * @brief Determine maximum error for __bf16 data
 * @details @copydetails skl_check_error_ulp_f32
 */
static inline float skl_error_ulp_bf16(const __bf16 *res, const __bf16 *ref,
                                       size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_bf16(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  return max;
}

//----- Logging macros -----

/**
 * @brief Logging verbosity levels.
 */
enum {
  SKL_TEST_LOG_ERROR = 0, /**< Error messages only */
  SKL_TEST_LOG_INFO,      /**< Error and info messages */
  SKL_TEST_LOG_DEBUG,     /**< All messages */
};

/**
 * @brief Conditionally log a message.
 *
 * @param T - Test context pointer
 * @param LEVEL - Minimum log level required for this message
 * @param ... - Printf-style format string and arguments
 *
 * Logs a message only if T->log_level >= LEVEL.
 * Higher log levels provide more verbose output.
 */
#define SKL_TEST_LOG(T, LEVEL, ...)                                            \
  if ((T)->log_level >= (LEVEL)) {                                             \
    FILE *stream = (LEVEL) == SKL_TEST_LOG_ERROR ? stderr : stdout;            \
    skl_test_driver_log((T), stream, __VA_ARGS__);                             \
  }

/**
 * @brief Assert a condition and set error status.
 *
 * @param T - Test context pointer.
 * @param STATUS - The member of skl_test_step_status_t to set upon failure.
 * @param COND - Condition to check.
 *
 * If COND is false, logs an error message with the test name, condition,
 * file, and line number, then sets T->status to SKL_TEST_FAIL. Does not
 * return early, allowing multiple requirements to be checked.
 */
#define SKL_TEST_REQUIRE(T, STATUS, COND)                                      \
  do {                                                                         \
    if (!(COND)) {                                                             \
      skl_test_driver_log((T), stderr, "%s: %s\n", (T)->suite_name, #COND);    \
      skl_test_driver_log((T), stderr, "    %s:%d\n", __FILE__, __LINE__);     \
      (T)->status.STATUS = SKL_TEST_FAIL;                                      \
    }                                                                          \
  } while (0)

/**
 * @brief Check packed matrix dimensions and strides
 *
 * @param t - Test context.
 * @param m0 - Number of rows in each block of the matrix.
 * @param n0 - Number of columns in each block of the matrix.
 * @param m1 - Number of block-rows in the matrix.
 * @param n1 - Number of block-columns in the matrix.
 * @param rs0 - Row stride within each block of the matrix in elements.
 * @param cs0 - Column stride within each block of the matrix in elements.
 * @param rs1 - Row stride between blocks of the matrix in elements.
 * @param cs1 - Column stride between blocks of the matrix in elements.
 *
 * This function performs some basic checks on the dimensions and strides of a
 * packed matrix and updates the init_status of t.
 */
static inline void skl_test_check_matrix_params_rcprc(skl_test_t *t, size_t m0,
                                                      size_t n0, size_t m1,
                                                      size_t n1, size_t rs0,
                                                      size_t cs0, size_t rs1,
                                                      size_t cs1) {
  SKL_TEST_REQUIRE(t, init_status, m0 > 0);
  SKL_TEST_REQUIRE(t, init_status, n0 > 0);

  if (m0 > 1) {
    SKL_TEST_REQUIRE(t, init_status, rs0 > 0);
  }
  if (n0 > 1) {
    SKL_TEST_REQUIRE(t, init_status, cs0 > 0);
  }
  if (m0 > 1 && n0 > 1) {
    if (rs0 >= cs0) {
      SKL_TEST_REQUIRE(t, init_status, rs0 >= (n0 - 1) * cs0 + 1);
    } else {
      SKL_TEST_REQUIRE(t, init_status, cs0 >= (m0 - 1) * rs0 + 1);
    }
  }

  size_t block_min_len = (m0 - 1) * rs0 + (n0 - 1) * cs0 + 1;
  if (m1 > 1 && n1 > 0) {
    SKL_TEST_REQUIRE(t, init_status, rs1 >= block_min_len);
  }
  if (m1 > 0 && n1 > 1) {
    SKL_TEST_REQUIRE(t, init_status, cs1 >= block_min_len);
  }
  if (m1 > 1 && n1 > 1) {
    if (rs1 >= cs1) {
      SKL_TEST_REQUIRE(t, init_status, rs1 >= (n1 - 1) * cs1 + block_min_len);
    } else {
      SKL_TEST_REQUIRE(t, init_status, cs1 >= (m1 - 1) * rs1 + block_min_len);
    }
  }
}
