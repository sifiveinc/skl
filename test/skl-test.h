// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if __riscv_xlen == 32
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint32_t lo;                                                               \
    uint32_t hi0, hi1;                                                         \
    /* guard against overflow between reads of lo/hi counter halves */         \
    do {                                                                       \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi0));                      \
      __asm__ volatile("rd" #COUNTER " %0" : "=r"(lo));                        \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi1));                      \
    } while (hi0 != hi1);                                                      \
    return (uint64_t)lo + ((uint64_t)hi1 << 32);                               \
  }
#elif __riscv_xlen == 64
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint64_t res;                                                              \
    __asm__ volatile("rd" #COUNTER " %0" : "=r"(res));                         \
    return res;                                                                \
  }
#endif

RISCV_READ_COUNTER_FUNC(cycle)   // riscv_read_mcycle()
RISCV_READ_COUNTER_FUNC(instret) // riscv_read_minstret()

static inline void riscv_fence(void) {
  __asm__ volatile("fence" : : : "memory");
}

// No floating-point support in some bare-metal libc printf, so emulate
static inline void print_float(float x) {
  x += 0.00005f;
  int i = (int)x;
  int f = (x - i) * 10000;
  printf("%d.%04d", i, f);
}

/**
 * @brief Output performance results in throughput and latency.
 *
 * @param name The name of the function.
 * @param cycles The number of cycles of execution.
 * @param insts The number of instructions executed.
 * @param num_elems The number of elements (application-specific)
 *
 * Writes a performance report to stdout in the following format:
 * ```
 * SIFIVE <name> latency: <cycles> cycles
 * SIFIVE <name> throughput: <epc> elements/cycle
 * ```
 */
static inline void report_perf_epc(const char *name, uint64_t cycles,
                                   uint64_t insts, size_t num_elems) {
  float epc = (float)num_elems / (float)cycles;
  printf("\n%15s : ", name);
  print_float(epc);
  printf(" elements / cycle (%" PRIu64 " cycles)\n", cycles);
  float ipe = (float)insts / (float)num_elems;
  printf("%15s : ", name);
  print_float(ipe);
  printf(" insts / element  (%" PRIu64 " insts)\n", insts);
}

/**
 * @brief Run a benchmark and report performance.
 *
 * @param name - The name of the benchmark.
 * @param num_elems - The number of elements (application-specific).
 * @param warmup - Whether to run a warmup iteration.
 * @param func - The function to benchmark (will be applied to __VA_ARGS__).
 * @param ... - The arguments to pass to the function.
 *
 * This macro is a wrapper around riscv_read_mcycle() and
 * riscv_read_minstret() to measure the performance of a function.
 *
 * If warmup is true, run the function once before timing to warm up caches.
 *
 * If ENABLE_BENCHMARK is not defined, this macro does nothing.
 */
#if defined(ENABLE_BENCHMARK)
#define SKL_BENCHMARK_RUN(name, num_elems, warmup, func, ...)                  \
  do {                                                                         \
    printf("SKL Benchmark %s (%zu elements):\n", name, (size_t)num_elems);     \
    if (warmup) {                                                              \
      func(__VA_ARGS__);                                                       \
    }                                                                          \
    riscv_fence();                                                             \
    uint64_t c0 = riscv_read_mcycle();                                         \
    uint64_t i0 = riscv_read_minstret();                                       \
    func(__VA_ARGS__);                                                         \
    riscv_fence();                                                             \
    uint64_t c1 = riscv_read_mcycle();                                         \
    uint64_t i1 = riscv_read_minstret();                                       \
    uint64_t cycles = c1 - c0;                                                 \
    uint64_t insts = i1 - i0;                                                  \
    printf("SIFIVE %s latency: %" PRIu64 " cycles\n", name, cycles);           \
    printf("SIFIVE %s instructions: %" PRIu64 " instructions\n", name, insts); \
    printf("SIFIVE %s throughput: ", name);                                    \
    print_float((float)num_elems / (float)cycles);                             \
    printf(" elements/cycle\n");                                               \
  } while (0)
#else
#define SKL_BENCHMARK_RUN(name, num_elems, warmup, func, ...) ((void)0)
#endif

/**
 * @brief Macro to check a requirement and set status to 1 if not met.
 *
 * @param status The status variable to set to 1 if the requirement is not met.
 * @param requirement The requirement to check.
 *
 * This helps to sanity-check parameters supplied to a test harness.
 * It is especially useful when a test is a wrapper around a kernel that has
 * more restrictive input requirements than the general API it implements.
 *
 * @note Callers should set status to 0 before calling this macro, and
 * `exit(status)` afterwards if any requirements were not met.
 */
#define SKL_TEST_REQUIRE(status, requirement)                                  \
  if (!(requirement)) {                                                        \
    printf("Test %s requires " #requirement "\n", __func__);                   \
    status = 1;                                                                \
  }

#ifdef __riscv_xsfmmbase
size_t skl_get_te_xsfmmbase(void) {
  size_t te = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e8, w1" : "=r"(te) : : "vtype", "vl");
  return te;
}
#endif

// Higher precision GEMM for computing bounds for GEMMs with float64
// accumulators
static inline void skl_gemm_f128rc_f128rc_f128rc_scalar(
    size_t m, size_t n, size_t k, long double alpha, const long double *a,
    size_t rsa, size_t csa, const long double *b, size_t rsb, size_t csb,
    long double beta, long double *c, size_t rsc, size_t csc) {
  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj++) {
      long double acc = 0;
      for (size_t kk = 0; kk < k; kk++) {
        acc += a[ii * rsa + kk * csa] * b[kk * rsb + jj * csb];
      }
      c[ii * rsc + jj * csc] = beta * c[ii * rsc + jj * csc] + alpha * acc;
    }
  }
}

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

static inline void skl_print_max_error(const char *name, float err) {
  printf("%15s : maximum error ", name);
  print_float(err);
  printf(" ulp\n");
}

/**
 * @brief Check maximum error for float data
 *
 * @param name Name of the test
 * @param res Array of test result values
 * @param ref Array of test reference values
 * @param tol Maximum error tolerance, in ULPs
 * @param len Length of result and reference arrays
 * @returns non-zero integer if maximum error is greater than tol
 *
 * This function calculates the absolute error in ulps between res[i]
 * and ref[i] for i in [0, len), prints the maximum such error found,
 * then returns 0 if the maximum is less than or equal to the given
 * tolerance or non-zero if greater.
 */
static inline int skl_check_error_ulp_f32(const char *name, const float *res,
                                          const float *ref, float tol,
                                          size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_f32(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  skl_print_max_error(name, max);
  return max > tol;
}

/**
 * @brief Check maximum error for _Float16 data
 * @details @copydetails skl_check_error_ulp_f32
 */
static inline int skl_check_error_ulp_f16(const char *name, const _Float16 *res,
                                          const _Float16 *ref, float tol,
                                          size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_f16(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  skl_print_max_error(name, max);
  return max > tol;
}

/**
 * @brief Check maximum error for __bf16 data
 * @details @copydetails skl_check_error_ulp_f32
 */
static inline int skl_check_error_ulp_bf16(const char *name, const __bf16 *res,
                                           const __bf16 *ref, float tol,
                                           size_t len) {
  float max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = skl_abs_error_ulp_bf16(res[i], ref[i]);
    if (err > max) {
      max = err;
    }
  }
  skl_print_max_error(name, max);
  return max > tol;
}

/**
 * @brief Test/Benchmark Initialization Modes
 *
 * Determines the behavior of skl_test_init functions in either test or
 * benchmark mode via the TEST_INIT_MODE macro.
 *
 * Benchmarks will typically use SEQ to reduce overhead from random number
 * generation. Tests will typically use RANDOM though some will likely use SEQ
 * to test specific ranges of interest. More modes may be added in the future.
 */
enum {
  SEQ,   ///< Initialize with equally-spaced values from min to max
  RANDOM ///< Initialize with random values
};

/**
 * @defgroup test_ranges Test/Benchmark Value Range Configuration
 * @brief Macros defining min/max ranges for input test data generation
 *
 * These macros can be overridden by defining them before including skl-test.h,
 * and individual tests will likely do this in CMakeLists.txt.
 * @{
 */

/**
 * @brief Template documentation for random range macros.
 * Used by testers as defaults for skl_test_init functions.
 * Can be overridden by defining before including skl-test.h.
 * @internal
 */
#define SKL_TEST_INIT_TEMPLATE

#if !defined(SKL_TEST_MIN_F16)
/** @brief Minimum value for random float16 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MIN_F16 0.0f
#endif

#if !defined(SKL_TEST_MAX_F16)
/** @brief Maximum value for random float16 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MAX_F16 1.0f
#endif

#if !defined(SKL_TEST_MIN_BF16)
/** @brief Minimum value for random bfloat16 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MIN_BF16 0.0f
#endif

#if !defined(SKL_TEST_MAX_BF16)
/** @brief Maximum value for random bfloat16 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MAX_BF16 1.0f
#endif

#if !defined(SKL_TEST_MIN_F32)
/** @brief Minimum value for random float inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MIN_F32 0.0f
#endif

#if !defined(SKL_TEST_MAX_F32)
/** @brief Maximum value for random float inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MAX_F32 1.0f
#endif

#if !defined(SKL_TEST_MIN_F64)
/** @brief Minimum value for random double inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MIN_F64 0.0
#endif

#if !defined(SKL_TEST_MAX_F64)
/** @brief Maximum value for random double inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MAX_F64 1.0
#endif

#if !defined(SKL_TEST_MIN_I8)
/** @brief Minimum value for random int8 inputs. @copydoc SKL_TEST_INIT_TEMPLATE
 */
#define SKL_TEST_MIN_I8 -128
#endif

#if !defined(SKL_TEST_MAX_I8)
/** @brief Maximum value for random int8 inputs. @copydoc SKL_TEST_INIT_TEMPLATE
 * @note This is an inclusive maximum, so the actual range is [MIN_I8, MAX_I8 +
 * 1]. */
#define SKL_TEST_MAX_I8 127
#endif

#if !defined(SKL_TEST_MIN_I32)
/** @brief Minimum value for random int32 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE */
#define SKL_TEST_MIN_I32 -128
#endif

#if !defined(SKL_TEST_MAX_I32)
/** @brief Maximum value for random int32 inputs. @copydoc
 * SKL_TEST_INIT_TEMPLATE
 * @note This is an inclusive maximum, so the actual range is [MIN_I32, MAX_I32
 * + 1]. */
#define SKL_TEST_MAX_I32 127
#endif

/** @} */ // end of test_random_ranges group

/**
 * @brief Macro to generate skl_test_init functions for different data types
 *
 * @param TYPE - The data type (e.g., _Float16, float, int8_t)
 * @param SUFFIX - The function suffix (e.g., f16, f32, i8)
 * @param IMPL_TYPE - Either FLOAT or INT to select implementation
 * @param FRAC_TYPE - The fractional type for floating point (float or double)
 *
 * Generated functions have the signature:
 * ```c
 * static inline void skl_test_init_<SUFFIX>(TYPE *buf, size_t len, TYPE min,
 * TYPE max);
 * ```
 *
 */
#define SKL_TEST_INIT_FUNC(TYPE, SUFFIX, IMPL_TYPE, FRAC_TYPE)                 \
  static inline void skl_test_init_##SUFFIX(TYPE *buf, size_t len, TYPE min,   \
                                            TYPE max) {                        \
    const TYPE step = (max - min) / len;                                       \
    for (size_t i = 0; i < len; i++) {                                         \
      if (TEST_INIT_MODE == RANDOM) {                                          \
        SKL_TEST_INIT_RANDOM_IMPL_##IMPL_TYPE(TYPE, FRAC_TYPE);                \
      } else {                                                                 \
        buf[i] = (TYPE)(min + step * i);                                       \
      }                                                                        \
    }                                                                          \
  }

#define SKL_TEST_INIT_RANDOM_IMPL_FLOAT(TYPE, FRAC_TYPE)                       \
  FRAC_TYPE frac = (FRAC_TYPE)rand() / (FRAC_TYPE)RAND_MAX;                    \
  buf[i] = (TYPE)(frac * (max - min) + min);

#define SKL_TEST_INIT_RANDOM_IMPL_INT(TYPE, UNUSED)                            \
  buf[i] = (TYPE)rand() % (max - min + 1) + min;

/**
 * @brief Buffer initialization modes
 * @details Determines the behavior of skl_test_init functions when not in
 * test mode.
 */
#if !defined(TEST_INIT_MODE)
/** @brief Default test/benchmark initialization mode. @copydoc TEST_INIT_MODE
 */
#define TEST_INIT_MODE RANDOM
#endif

/**
 * @defgroup test_init Test/Benchmark Data Initialization Functions
 * @brief Functions for initializing buffers with random or sequential data
 *
 * All functions in this group share a common interface and behavior:
 * - @param buf Output buffer to fill with values
 * - @param len Number of elements to generate
 * - Fill a buffer with values appropriate for testing
 * - In test mode: generate random values within configured ranges
 * - In benchmark mode: generate sequential values starting from 0
 * @{
 */

/**
 * @brief Initialize buffer with _Float16 values
 */
SKL_TEST_INIT_FUNC(_Float16, f16, FLOAT, float)

/**
 * @brief Initialize buffer with __bf16 values
 */
SKL_TEST_INIT_FUNC(__bf16, bf16, FLOAT, float)

/**
 * @brief Initialize buffer with float values
 */
SKL_TEST_INIT_FUNC(float, f32, FLOAT, float)

/**
 * @brief Initialize buffer with double values
 */
SKL_TEST_INIT_FUNC(double, f64, FLOAT, double)

/**
 * @brief Initialize buffer with int8_t values
 */
SKL_TEST_INIT_FUNC(int8_t, i8, INT, unused)

/**
 * @brief Initialize buffer with int32_t values
 */
SKL_TEST_INIT_FUNC(int32_t, i32, INT, unused)

/** @} */ // end of test_init group

#undef SKL_TEST_INIT_FUNC
#undef SKL_TEST_INIT_RANDOM_IMPL_FLOAT
#undef SKL_TEST_INIT_RANDOM_IMPL_INT
