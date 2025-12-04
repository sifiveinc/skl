// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/**
 * @file skl-common.h
 * @brief Definitions shared by all SKL kernels.
 *
 * This is the only file on which any SKL source may depend.
 * To extract a function from the SKL project, simply copy its source file,
 * its declaration header, and `skl-common.h`.
 */

#if !defined(SKL_FUNC)
/** Declaration decorator for all public SKL functions.
 *
 * By default, this macro is defined to nothing, but when SKL is built as a
 * header, it can be defined to `static inline` to prevent redefinitions and
 * linker symbol conflicts.
 *
 * @note
 * Private functions (helper functions not exposed by the SKL API) are to be
 * declared with `SKL_FUNC_PRIVATE`.
 */
#define SKL_FUNC
#endif

#if !defined(SKL_FUNC_PRIVATE)
/** Declaration decorator for private functions within a SKL kernel.
 *
 * By default, this is macro is defined to `static`, but when SKL is built
 * as a header, it can be defined to `static inline` to prevent redefinitions
 * and linker symbol conflicts.
 *
 * @note
 * Private functions (helper functions not exposed by the SKL API) are to be
 * declared with `SKL_FUNC_PRIVATE`.
 */
#define SKL_FUNC_PRIVATE static
#endif

/** Declaration decorator for shared SKL utility functions.
 *
 * For utility functions shared between SKL kernels. Since not all functions
 * will be used by every kernel, definitions in `skl-common.h` are marked static
 * inline to avoid warnings about unused functions.
 *
 * These functions are considered private to SKL, and not part of the public
 * API.
 */
#define SKL_FUNC_UTIL static inline

/*
 * Xsfmm ABI macros:
 *
 * Placeholders for function attributes that indicate how Xsfmm matrix tile
 * state is used. These will be replaced with appropriate function attributes
 * once they are provided by the compiler.
 */

/** Xsfmm state shared with caller. Function reads state but does not modify. */
#define SKL_XSFMM_IN

/**
 * Xsfmm state shared with caller. Function ignores incoming state and
 * overwrites it.
 */
#define SKL_XSFMM_OUT

/** Xsfmm state shared with caller. Function reads and modifies state. */
#define SKL_XSFMM_INOUT

/** Function creates a new scope for Xsfmm state. */
#define SKL_XSFMM_NEW

/** Portable restrict pointer qualifier for C and C++ */
#if !defined(__cplusplus)
#define SKL_RESTRICT restrict
#else
#if defined(__GNUC__) || defined(__clang__)
#define SKL_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SKL_RESTRICT __restrict
#else
#define SKL_RESTRICT
#endif
#endif

/**
 * @brief An instruction scheduling barrier.
 *
 * This function acts as a barrier to instruction scheduling, by pretending to
 * have arbitrary side-effects. This is a necessary hack in some cases to
 * override pessimal scheduling decisions by the compiler when writing RVV
 * intrinsics.
 *
 * @note This is very much a hack, and should be used with caution. It is not
 *       guaranteed to prevent reordering of instructions in all cases, and
 *       in particular only works when the instructions in question access
 *       memory.
 */
SKL_FUNC_UTIL void skl_instruction_schedule_barrier(void) {
  __asm__ volatile("" ::: "memory");
}

/** @brief Reinterprets an unsigned 32-bit integer as an IEEE FP32. */
SKL_FUNC_UTIL float skl_u32_as_float(uint32_t x) {
  float y;
  memcpy(&y, &x, sizeof(float));
  return y;
}

/** @brief Function to convert OFP8 E4M3 to IEEE FP32.
 *
 * @param in - An OFP8 8-bit floating point number in E4M3 format, type-punned
 * as an 8-bit unsigned integer.
 */
SKL_FUNC_UTIL float skl_cvt_f8e4m3_f32(uint8_t in) {
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
SKL_FUNC_UTIL float skl_cvt_f8e5m2_f32(uint8_t in) {
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
SKL_FUNC_UTIL float skl_cvt_f4e2m1_f32(uint8_t in) {
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
SKL_FUNC_UTIL uint8_t skl_cvt_f32_f8e4m3(float in, bool is_sat) {
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
SKL_FUNC_UTIL uint8_t skl_cvt_f32_f8e5m2(float in, bool is_sat) {
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
