// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/** @brief Reinterprets an unsigned 32-bit integer as an IEEE FP32. */
SKL_FUNC_PRIVATE float skl_u32_as_float(uint32_t x) {
  float y;
  __builtin_memcpy(&y, &x, sizeof(float));
  return y;
}

SKL_FUNC float skl_cvt_f8e4m3_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 7)
  uint32_t sign = ((uint32_t)in & 0x80U) << 24U; // Shift to FP32 sign position

  // E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
  uint32_t exponent = ((uint32_t)in >> 3U) & 0x0FU; // Bits 6-3
  uint32_t mantissa = (uint32_t)in & 0x07U;         // Bits 2-0

  if (exponent == 0U) {
    // Subnormal numbers
    if (mantissa != 0U) {
      // Subnormal E4M3: value = 2^(-6) * (mantissa / 2^3)
      int lz = __builtin_clz(mantissa << 29U);
      uint32_t msk = ~0U << (uint32_t)(2 - lz);
      uint32_t man = (mantissa & ~msk) << (uint32_t)(20 + lz + 1);
      uint32_t exp = (uint32_t)(1 - (lz + 1) - 7 + 127) << 23U;
      result = sign | exp | man;
    } else {
      // Zero
      result = sign;
    }
  } else if (exponent == 0x0FU && mantissa == 0x7U) {
    // RISC-V qNaN
    result = (0xFFU << 23U) | (0x1U << 22U);
  } else {
    // Normal numbers
    // Bias: E4M3 uses 7, FP32 uses 127
    uint32_t unbiased_exp = exponent - 7U + 127U;
    result = sign | (unbiased_exp << 23U) | (mantissa << 20U);
  }

  return skl_u32_as_float(result);
}

SKL_FUNC float skl_cvt_f8e5m2_f32(uint8_t in) {
  uint32_t result = 0; // Initialize to zero

  // Extract sign bit (bit 7)
  uint32_t sign = ((uint32_t)in & 0x80U) << 24U; // Shift to FP32 sign position

  // E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
  uint32_t exponent = ((uint32_t)in >> 2U) & 0x1FU; // Bits 6-2
  uint32_t mantissa = (uint32_t)in & 0x03U;         // Bits 1-0

  if (exponent == 0U) {
    // Subnormal numbers
    if (mantissa != 0U) {
      // Subnormal E5M2: value = 2^(-14) * (mantissa / 2^2)
      int lz = __builtin_clz(mantissa << 30U);
      uint32_t msk = ~0U << (uint32_t)(1 - lz);
      uint32_t man = (mantissa & ~msk) << (uint32_t)(20 + lz + 2);
      uint32_t exp = (uint32_t)(1 - (lz + 1) - 15 + 127) << 23U;
      result = sign | exp | man;

    } else {
      // Zero
      result = sign;
    }
  } else if (exponent == 0x1FU) {
    // Infinity or NaN
    if (mantissa == 0U) {
      // Infinity
      result = sign | (0xFFU << 23U); // FP32 infinity
    } else {
      // RISC-V qNaN
      result = (0xFFU << 23U) | (0x1U << 22U);
    }
  } else {
    // Normal numbers
    // Bias: E5M2 uses 15, FP32 uses 127
    uint32_t unbiased_exp = exponent - 15U + 127U;
    result = sign | (unbiased_exp << 23U) | (mantissa << 21U);
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

SKL_FUNC void skl_cvt_f32_f8e4m3_ref(uint8_t *pDst, const float *pSrc,
                                     float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e4m3(pSrc[i] * scaling_factor, false);
  }
}

SKL_FUNC void skl_cvt_sat_f32_f8e4m3_ref(uint8_t *pDst, const float *pSrc,
                                         float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e4m3(pSrc[i] * scaling_factor, true);
  }
}

SKL_FUNC void skl_cvt_f32_f8e5m2_ref(uint8_t *pDst, const float *pSrc,
                                     float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e5m2(pSrc[i] * scaling_factor, false);
  }
}

SKL_FUNC void skl_cvt_sat_f32_f8e5m2_ref(uint8_t *pDst, const float *pSrc,
                                         float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e5m2(pSrc[i] * scaling_factor, true);
  }
}

SKL_FUNC void skl_cvt_bf16_f8e4m3_ref(uint8_t *pDst, const __bf16 *pSrc,
                                      float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e4m3((float)pSrc[i] * scaling_factor, false);
  }
}

SKL_FUNC void skl_cvt_sat_bf16_f8e4m3_ref(uint8_t *pDst, const __bf16 *pSrc,
                                          float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e4m3((float)pSrc[i] * scaling_factor, true);
  }
}

SKL_FUNC void skl_cvt_bf16_f8e5m2_ref(uint8_t *pDst, const __bf16 *pSrc,
                                      float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e5m2((float)pSrc[i] * scaling_factor, false);
  }
}

SKL_FUNC void skl_cvt_sat_bf16_f8e5m2_ref(uint8_t *pDst, const __bf16 *pSrc,
                                          float scaling_factor, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = skl_cvt_f32_f8e5m2((float)pSrc[i] * scaling_factor, true);
  }
}

SKL_FUNC void skl_cvt_f8e4m3_bf16_ref(__bf16 *pDst, const uint8_t *pSrc,
                                      size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = (__bf16)skl_cvt_f8e4m3_f32(pSrc[i]);
  }
}

SKL_FUNC void skl_cvt_f8e5m2_bf16_ref(__bf16 *pDst, const uint8_t *pSrc,
                                      size_t n) {
  for (size_t i = 0; i < n; ++i) {
    pDst[i] = (__bf16)skl_cvt_f8e5m2_f32(pSrc[i]);
  }
}
