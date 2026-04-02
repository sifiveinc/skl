// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// NOLINTBEGIN(*)
// TODO(fix linting errors)
SKL_FUNC float skl_cvt_f8e4m3_f32(uint8_t in) {
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

SKL_FUNC float skl_cvt_f8e5m2_f32(uint8_t in) {
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

SKL_FUNC uint8_t skl_cvt_f32_f8e4m3(float in, bool is_sat) {
  const uint8_t nan_expr = 0x7F;
  const uint8_t max_bits = 0x7E;

  uint32_t in_bits;
  __builtin_memcpy(&in_bits, &in, sizeof(in));
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

SKL_FUNC uint8_t skl_cvt_f32_f8e5m2(float in, bool is_sat) {
  const uint8_t nan_expr = 0x7F;
  const uint8_t inf_bits = 0x7C;
  const uint8_t max_bits = 0x7B;

  uint32_t in_bits;
  __builtin_memcpy(&in_bits, &in, sizeof(in));
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
// NOLINTEND(*)
