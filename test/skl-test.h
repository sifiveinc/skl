// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Extract the test name from SKL_TEST_NAME as a string. */
#define SKL_TEST_NAME_STR(NAME) #NAME
#define DECL_SKL_TEST_NAME(NAME)                                               \
  static const char *skl_test_name = SKL_TEST_NAME_STR(NAME)
DECL_SKL_TEST_NAME(SKL_TEST_NAME);

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
 * @brief Output performance results in throughput as elements/cycle. (Default)
 *
 * @param name The name of the function being benchmarked.
 * @param cycles The number of cycles of execution.
 * @param insts The number of instructions executed.
 * @param num_elems The number of elements used in the benchmark
 * (application-specific).
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
 * @brief Output performance results in throughput as MACCs/cycle.
 *
 * @param name The name of the function being benchmarked.
 * @param cycles The number of cycles of execution.
 * @param insts The number of instructions executed.
 * @param num_maccs The number of multiply-accumulate ops used in the benchmark
 * (application-specific).
 */
static inline void report_perf_mpc(const char *name, uint64_t cycles,
                                   uint64_t insts, size_t num_maccs) {
  float mpc = (float)num_maccs / (float)cycles;
  printf("\n%15s : ", name);
  print_float(mpc);
  printf(" MACCs / cycle (%" PRIu64 " cycles)\n", cycles);
  printf("%15s : %" PRIu64 " insts\n", name, insts);
}

/** @brief Determine whether to execute a warmup iteration.
 *
 * Used by most SKL benchmarks as a default. Can be overridden by defining
 * before including skl-test.h.
 */
#if !defined(SKL_TEST_WARMUP)
#define SKL_TEST_WARMUP 1
#endif

/**
 * @brief Set benchmark performance reporting function.
 *
 * Must be a function with the following signature:
 *
 * ```c
 * void report_perf(const char *name, uint64_t cycles, uint64_t insts, size_t
 * num_elems);
 * ```
 *
 * Used by most SKL benchmarks as a default. Can be overridden by defining
 * before including skl-test.h.
 */
#if !defined(SKL_TEST_PERF_REPORT)
#define SKL_TEST_PERF_REPORT report_perf_epc
#else
void SKL_TEST_PERF_REPORT(const char *name, uint64_t cycles, uint64_t insts,
                          size_t num_elems);
#endif

/**
 * @brief Run a benchmark and report performance.
 *
 * @param NAME - The name of the benchmark.
 * @param NUM_ELEMS - The number of elements (application-specific).
 * @param WARMUP - Whether to run a warmup iteration.
 * @param FUNC - The function to benchmark (will be applied to __VA_ARGS__).
 * @param ... - The arguments to pass to the function.
 *
 * This macro is a wrapper around riscv_read_mcycle() and
 * riscv_read_minstret() to measure the performance of a function.
 *
 * If WARMUP is true, run the function once before timing to warm up caches.
 * Usually WARMUP is set to SKL_TEST_WARMUP, which is 1 by default.
 *
 * If ENABLE_BENCHMARK is not defined, but ENABLE_TEST is defined, this macro
 * calls FUNC(__VA_ARGS__).  Otherwise, this macro does nothing.
 */
#if defined(ENABLE_BENCHMARK)
#define SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, WARMUP, FUNC, ...)                  \
  do {                                                                         \
    if (WARMUP) {                                                              \
      FUNC(__VA_ARGS__);                                                       \
    }                                                                          \
    riscv_fence();                                                             \
    uint64_t i0 = riscv_read_minstret();                                       \
    uint64_t c0 = riscv_read_mcycle();                                         \
    FUNC(__VA_ARGS__);                                                         \
    riscv_fence();                                                             \
    uint64_t c1 = riscv_read_mcycle();                                         \
    uint64_t i1 = riscv_read_minstret();                                       \
    uint64_t cycles = c1 - c0;                                                 \
    uint64_t insts = i1 - i0;                                                  \
    SKL_TEST_PERF_REPORT(NAME, cycles, insts, (size_t)(NUM_ELEMS));            \
  } while (0)
#elif defined(ENABLE_TEST)
#define SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, WARMUP, FUNC, ...)                  \
  do {                                                                         \
    FUNC(__VA_ARGS__);                                                         \
  } while (0)
#else
#define SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, WARMUP, FUNC, ...) ((void)0)
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

/** @brief Reinterprets an unsigned 32-bit integer as an IEEE FP32. */
static inline float skl_u32_as_float(uint32_t x) {
  float y;
  memcpy(&y, &x, sizeof(float));
  return y;
}

/** @brief Function to convert OFP8 E4M3 to IEEE FP32.
 *
 * @param in - An OFP8 8-bit floating point number in E4M3 format, type-punned
 * as an 8-bit unsigned integer.
 */
static inline float skl_cvt_f8e4m3_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 7)
  uint32_t sign = (in & 0x80) << 24; // Shift to FP32 sign position

  // E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
  uint32_t exponent = (in >> 3) & 0x0F; // Bits 6-3
  uint32_t mantissa = in & 0x07;        // Bits 2-0

  if (exponent == 0) {
    // Subnormal numbers
    if (mantissa != 0) {
      // Subnormal E4M3: value = 2^(-6) * (mantissa / 2^3)
      int lz = __builtin_clz(mantissa << 29);
      uint32_t msk = ~(uint32_t)0 << (2 - lz);
      uint32_t man = (mantissa & ~msk) << (20 + lz + 1);
      uint32_t exp = (1 - (lz + 1) - 7 + 127) << 23;
      result = sign | exp | man;
    } else {
      // Zero
      result = sign;
    }
  } else if (exponent == 0x0F && mantissa == 0x7) {
    // RISC-V qNaN
    result = (0xFF << 23) | (0x1 << 22);
  } else {
    // Normal numbers
    // Bias: E4M3 uses 7, FP32 uses 127
    int32_t unbiased_exp = exponent - 7 + 127;
    result = sign | (unbiased_exp << 23) | (mantissa << 20);
  }

  return skl_u32_as_float(result);
}

/** @brief Function to convert OFP8 E5M2 to IEEE FP32.
 *
 * @param in - An OFP8 8-bit floating point number in E5M2 format, type-punned
 * as an 8-bit unsigned integer.
 */
static inline float skl_cvt_f8e5m2_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 7)
  uint32_t sign = (in & 0x80) << 24; // Shift to FP32 sign position

  // E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
  uint32_t exponent = (in >> 2) & 0x1F; // Bits 6-2
  uint32_t mantissa = in & 0x03;        // Bits 1-0

  if (exponent == 0) {
    // Subnormal numbers
    if (mantissa != 0) {
      // Subnormal E5M2: value = 2^(-14) * (mantissa / 2^2)
      int lz = __builtin_clz(mantissa << 30);
      uint32_t msk = ~(uint32_t)0 << (1 - lz);
      uint32_t man = (mantissa & ~msk) << (20 + lz + 2);
      uint32_t exp = (1 - (lz + 1) - 15 + 127) << 23;
      result = sign | exp | man;

    } else {
      // Zero
      result = sign;
    }
  } else if (exponent == 0x1F) {
    // Infinity or NaN
    if (mantissa == 0) {
      // Infinity
      result = sign | (0xFF << 23); // FP32 infinity
    } else {
      // RISC-V qNaN
      result = (0xFF << 23) | (0x1 << 22);
    }
  } else {
    // Normal numbers
    // Bias: E5M2 uses 15, FP32 uses 127
    int32_t unbiased_exp = exponent - 15 + 127;
    result = sign | (unbiased_exp << 23) | (mantissa << 21);
  }

  return skl_u32_as_float(result);
}

/** @brief Function to convert OCP FP4 E2M1 to IEEE FP32.
 *
 * @param in - The lowest 4 bits of `in` contain the OCP FP4 4-bit floating
 * point number in E2M1 format.
 */
static inline float skl_cvt_f4e2m1_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 3)
  uint32_t sign = (in & 0x08) << 28; // Shift to FP32 sign position

  // E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bits
  uint32_t exponent = (in >> 1) & 0x03; // Bits 2-1
  uint32_t mantissa = in & 0x01;        // Bit 0

  if (exponent == 0) {
    // Subnormal E2M1: value = mantissa / 2^1
    if (mantissa != 0) {
      int lz = __builtin_clz(mantissa << 31);
      uint32_t msk = ~(uint32_t)0 << (1 - (lz + 1));
      uint32_t man = (mantissa & ~msk) << (23 - 1 + lz + 1);
      uint32_t exp = (1 - (lz + 1) - 1 + 127) << 23;
      result = sign | exp | man;
    } else {
      // Zero
      result = sign;
    }
  } else {
    // Normal numbers
    // Bias: E2M1 uses 1, FP32 uses 127
    int32_t unbiased_exp = exponent - 1 + 127;
    result = sign | (unbiased_exp << 23) | (mantissa << 22);
  }

  return skl_u32_as_float(result);
}

/** @brief Function to convert IEEE FP32 to OFP8 E4M3.
 *
 * @param in - An OFP8 8-bit floating point number in E4M3 format, type-punned
 * as an 8-bit unsigned integer.
 * @param is_sat - Whether to saturate the output to the maximum representable
 * value of the same sign when the input is out of range.
 */
static inline uint8_t skl_cvt_f32_f8e4m3(float in, bool is_sat) {
  const uint8_t nan_expr = 0x7F;
  const uint8_t max_bits = 0x7E;

  uint32_t in_bits;
  memcpy(&in_bits, &in, sizeof(in));
  uint8_t sign_bits = in_bits & 0x80000000 ? 0x80 : 0x00;

  // When saturating, the INF result would be replaced by the max representable
  // value of the same sign.
  uint8_t inf_expr = is_sat ? sign_bits | max_bits : nan_expr;

  if (isinf(in)) {
    return inf_expr;
  }
  if (isnan(in)) {
    return nan_expr;
  }

  float abs_in = fabsf(in);
  int32_t exp = (int32_t)floorf(log2f(abs_in));

  if (exp < -6) {
    uint8_t mantissa = (uint8_t)nearbyintf(abs_in * powf(2.0f, 6 + 3));
    return sign_bits | mantissa;
  }
  if (exp > 8) {
    return inf_expr;
  }

  uint8_t mantissa =
      (uint8_t)nearbyintf((abs_in * powf(2.0f, (float)(-exp)) - 1.0f) * 8);
  uint8_t mag_bits = ((uint8_t)(exp + 7) << 3U) + mantissa;
  if (mag_bits > max_bits) {
    return inf_expr;
  }
  return sign_bits | mag_bits;
}

/** @brief Function to convert IEEE FP32 to OFP8 E5M2.
 *
 * @param in - An OFP8 8-bit floating point number in E5M2 format, type-punned
 * as an 8-bit unsigned integer.
 * @param is_sat - Whether to saturate the output to the maximum representable
 * value of the same sign when the input is out of range.
 */
static inline uint8_t skl_cvt_f32_f8e5m2(float in, bool is_sat) {
  const uint8_t nan_expr = 0x7F;
  const uint8_t inf_bits = 0x7C;
  const uint8_t max_bits = 0x7B;

  uint32_t in_bits;
  memcpy(&in_bits, &in, sizeof(in));
  uint8_t sign_bits = in_bits & 0x80000000 ? 0x80 : 0x00;

  // When saturating, the INF result would be replaced by the max representable
  // value of the same sign.
  uint8_t inf_expr = is_sat ? sign_bits | max_bits : sign_bits | inf_bits;

  if (isinf(in)) {
    return inf_expr;
  }
  if (isnan(in)) {
    return nan_expr;
  }
  float abs_in = fabsf(in);
  int32_t exp = (int32_t)floorf(log2f(abs_in));

  if (exp < -14) {
    uint8_t mantissa = (uint8_t)nearbyintf(abs_in * powf(2.0f, 14 + 2));
    return sign_bits | mantissa;
  }
  if (exp > 15) {
    return inf_expr;
  }
  uint8_t mantissa =
      (uint8_t)nearbyintf((abs_in * powf(2.0f, (float)(-exp)) - 1.0f) * 4);
  uint8_t mag_bits = ((uint8_t)(exp + 15) << 2U) + mantissa;
  if (mag_bits > max_bits) {
    return inf_expr;
  }
  return sign_bits | mag_bits;
}

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

/**
 * @brief Print a short error message.
 *
 * @param name Name of the test
 * @param err The maximum error measured for this test
 */
static inline void skl_print_max_error(const char *name, float err) {
  printf("%15s : maximum error ", name);
  print_float(err);
  printf(" ulp\n");
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
static inline float skl_check_error_ulp_f32(const char *name, const float *res,
                                            const float *ref, float tol,
                                            size_t len) {
  float err = skl_error_ulp_f32(res, ref, len);
  skl_print_max_error(name, err);
  return err > tol;
}

/**
 * @brief Check maximum error for _Float16 data
 * @details @copydetails skl_check_error_ulp_f32
 */
static inline float skl_check_error_ulp_f16(const char *name,
                                            const _Float16 *res,
                                            const _Float16 *ref, float tol,
                                            size_t len) {
  float err = skl_error_ulp_f16(res, ref, len);
  skl_print_max_error(name, err);
  return err > tol;
}

/**
 * @brief Check maximum error for __bf16 data
 * @details @copydetails skl_check_error_ulp_f32
 */
static inline int skl_check_error_ulp_bf16(const char *name, const __bf16 *res,
                                           const __bf16 *ref, float tol,
                                           size_t len) {
  float err = skl_error_ulp_bf16(res, ref, len);
  skl_print_max_error(name, err);
  return err > tol;
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
 *
 * In STATIC mode, the input data is initialized with static values from
 * pre-defined arrays in SKL_TEST_DATA_HEADER.
 */
enum {
  SEQ,    ///< Initialize with equally-spaced values from min to max
  RANDOM, ///< Initialize with random values
  STATIC  ///< Initialize with static values from SKL_TEST_DATA_HEADER
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
    assert(TEST_INIT_MODE == RANDOM || TEST_INIT_MODE == SEQ);                 \
    const TYPE step = (max - min) / len;                                       \
    for (size_t i = 0; i < len; i++) {                                         \
      if (TEST_INIT_MODE == RANDOM) {                                          \
        SKL_TEST_INIT_RANDOM_IMPL_##IMPL_TYPE(TYPE, FRAC_TYPE);                \
      } else {                                                                 \
        buf[i] = (TYPE)(min + step * i);                                       \
      }                                                                        \
    }                                                                          \
  }

#if defined(TEST_INIT_MODE) && TEST_INIT_MODE == STATIC
#define SKL_TEST_STATIC_DATA(NAME) NAME##_data
#define SKL_TEST_STATIC_DATA_LEN(NAME) NAME##_len
#else
#define SKL_TEST_STATIC_DATA(NAME) NULL
#define SKL_TEST_STATIC_DATA_LEN(NAME) 0
#endif

#define SKL_TEST_INIT_RANDOM_IMPL_FLOAT(TYPE, FRAC_TYPE)                       \
  FRAC_TYPE frac = (FRAC_TYPE)rand() / (FRAC_TYPE)RAND_MAX;                    \
  buf[i] = (TYPE)(frac * (max - min) + min);

#define SKL_TEST_INIT_RANDOM_IMPL_INT(TYPE, UNUSED)                            \
  buf[i] = (TYPE)rand() % (max - min + 1) + min;

/**
 * @brief Macro to initialize a buffer in a SKL test/benchmark.
 *
 * Behavior is determined by TEST_INIT_MODE macro:
 * - RANDOM: Initialize with random values between min and max
 * - STATIC: Copy values from existing array NAME_data to buf
 * - SEQ: Initialize with equally-spaced values from min to max
 *
 * @note For STATIC mode, the array NAME_data must be defined in the header file
 * specified by SKL_TEST_DATA_HEADER, but need not have the same length as the
 * buffer being initialized. The data will be repeated as necessary to fill the
 * buffer.
 *
 * @param BUF - The buffer to initialize
 * @param LEN - The length of the buffer
 * @param TYPE - The data type of the buffer
 * @param MIN - The minimum value to use for initialization
 * @param MAX - The maximum value to use for initialization
 * @param IMPL_TYPE - Either FLOAT or INT to select implementation
 * @param FRAC_TYPE - Fractional type for floating point intermediates(float or
 * double)
 */
#if defined(TEST_INIT_MODE) && TEST_INIT_MODE == STATIC
#define SKL_TEST_INIT(BUF, LEN, TYPE, MIN, MAX, IMPL_TYPE, FRAC_TYPE)          \
  {                                                                            \
    TYPE *buf = (BUF);                                                         \
    const size_t len = (LEN);                                                  \
    size_t avl = len;                                                          \
    const size_t buf_len = SKL_TEST_STATIC_DATA_LEN(BUF);                      \
    while (avl > 0) {                                                          \
      size_t vl = avl > buf_len ? buf_len : avl;                               \
      memcpy(buf, SKL_TEST_STATIC_DATA(BUF), vl * sizeof(TYPE));               \
      avl -= vl;                                                               \
      buf += vl;                                                               \
    }                                                                          \
  }
#else
#define SKL_TEST_INIT(BUF, LEN, TYPE, MIN, MAX, IMPL_TYPE, FRAC_TYPE)          \
  {                                                                            \
    TYPE *buf = (BUF);                                                         \
    const size_t len = (LEN);                                                  \
    const TYPE min = (MIN);                                                    \
    const TYPE max = (MAX);                                                    \
    const TYPE step = (max - min) / len;                                       \
    for (size_t i = 0; i < len; i++) {                                         \
      if (TEST_INIT_MODE == RANDOM) {                                          \
        SKL_TEST_INIT_RANDOM_IMPL_##IMPL_TYPE(TYPE, FRAC_TYPE);                \
      } else {                                                                 \
        buf[i] = (TYPE)(min + step * i);                                       \
      }                                                                        \
    }                                                                          \
  }
#endif

/**
 * @defgroup test_init_macros Test/Benchmark Data Initialization Macros
 *
 * These macros are intended to be the standard way to initialize data buffers
 * in SKL tests and benchmarks. They will displace the direct use of
 * skl_test_init functions (deprecated).
 *
 * All macros in this group share a common interface and behavior:
 * - @param BUF Output buffer to fill with values
 * - @param LEN Number of elements to generate
 * All macros are wrappers around SKL_TEST_INIT, which has more detailed
 * documentation.
 * @{
 */
#define SKL_TEST_INIT_F16(BUF, LEN)                                            \
  SKL_TEST_INIT(BUF, LEN, _Float16, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16, FLOAT, \
                float)

#define SKL_TEST_INIT_F32(BUF, LEN)                                            \
  SKL_TEST_INIT(BUF, LEN, float, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32, FLOAT,    \
                float)

#define SKL_TEST_INIT_F64(BUF, LEN)                                            \
  SKL_TEST_INIT(BUF, LEN, double, SKL_TEST_MIN_F64, SKL_TEST_MAX_F64, FLOAT,   \
                double)

/** @} */ // end of test_init_macros group

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

/**
 * @brief Custom memory allocation function
 *
 * If defined, benchmarks should use this function to allocate buffers.
 * If not defined, benchmarks should declare and initialize buffers as static
 * arrays.
 *
 * @note The prototype is declared by this macro so that
 * it can name an arbitrary symbol in a linked translation unit without
 * modification to the benchmark source.
 */
#if defined(SKL_TEST_MALLOC)
/**
 * @brief Aligned memory allocation function.
 *
 * @param alignment Alignment of memory to allocate in bytes
 * @param size Size of memory to allocate in bytes
 * @return Pointer to allocated memory
 */
void *SKL_TEST_MALLOC(size_t alignment, size_t size);
#endif

/**
 * @brief Custom memory free function
 *
 * If defined, this function will be used to free memory allocated
 * by SKL_TEST_MALLOC.
 *
 * @note The prototype is declared by this macro so that
 * it can name an arbitrary symbol in a linked translation unit without
 * modification to the benchmark source.
 */
#if defined(SKL_TEST_FREE) && defined(SKL_TEST_MALLOC)
/**
 * @brief Custom memory free function.
 *
 * @param ptr Pointer to memory to free
 */
void SKL_TEST_FREE(void *ptr);
#endif

#if TEST_INIT_MODE == STATIC
#if !defined(SKL_TEST_DATA_HEADER)
#error "SKL_TEST_DATA_HEADER must be defined when TEST_INIT_MODE == STATIC"
#else
#include SKL_TEST_DATA_HEADER
#endif
#endif
