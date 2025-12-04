// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

/**
 * @file cvt_zvfofp8min.h
 * @brief OFP8 Conversion Functions
 *
 * This header provides vectorized conversion functions between various
 * floating-point formats and 8-bit OFP8 formats (E4M3 and E5M2) using the
 * RISC-V Zvfofp8min extension.
 *
 * ## Function Variants:
 *
 * ### Narrowing Conversions (wider → OFP8):
 * - **Standard**: `skl_cvt_{input}_{output}_zvfofp8min()`
 *   - E4M3: Infinite values are converted to the canonical NaN (E4M3 cannot
 * represent infinity)
 *   - E5M2: Infinite values are preserved as infinity in the target format
 * - **Saturating**: `skl_cvt_sat_{input}_{output}_zvfofp8min()`
 *   - Infinite results are clamped to the maximum-magnitude finite value of the
 * same sign
 *
 * ### Widening Conversions (OFP8 → wider):
 * - **Standard**: `skl_cvt_{input}_{output}_zvfofp8min()`
 *   - Direct widening conversion with full precision preservation
 *
 * ## Scaling Factor:
 * Narrowing conversion functions accept a scaling factor parameter:
 * - **scaling_factor == 1.0f**: No scaling is applied.
 * - **scaling_factor != 1.0f**: Input values are multiplied by the scaling
 * factor before conversion
 *
 * This enables us to do dynamic range adjustment to make input values fit into
 * the limited OFP8 range.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Converts F32 values to E4M3 OFP8 format with scaling.
 *
 * @param pDst - Output array where E4M3 values are stored.
 * @param pSrc - Input array where the FP32 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 32-bit floating-point values to 8-bit E4M3 OFP format. Input values
 * are multiplied by the scaling factor before conversion. Since E4M3 cannot
 * represent infinity, infinite results are converted to the canonical NaN
 * (`0x7f`).
 */
void skl_cvt_f32_f8e4m3_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                   float scaling_factor, size_t n);

/**
 * @brief Converts F32 values to E4M3 OFP8 format with scaling and saturation.
 *
 * @param pDst - Output array where E4M3 values are stored.
 * @param pSrc - Input array where the FP32 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 32-bit floating-point values to 8-bit E4M3 OFP format with
 * saturation. Input values are multiplied by the scaling factor before
 * conversion. Infinite results are clamped to the maximum-magnitude finite
 * value of the same sign.
 */
void skl_cvt_sat_f32_f8e4m3_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                       float scaling_factor, size_t n);

/**
 * @brief Converts F32 values to E5M2 OFP8 format with scaling.
 *
 * @param pDst - Output array where E5M2 values are stored.
 * @param pSrc - Input array where the FP32 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 32-bit floating-point values to 8-bit E5M2 OFP format. Input values
 * are multiplied by the scaling factor before conversion. Infinite results are
 * preserved as infinity in the target format.
 */
void skl_cvt_f32_f8e5m2_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                   float scaling_factor, size_t n);

/**
 * @brief Converts F32 values to E5M2 OFP8 format with scaling and saturation.
 *
 * @param pDst - Output array where E5M2 values are stored.
 * @param pSrc - Input array where the FP32 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 32-bit floating-point values to 8-bit E5M2 OFP format with
 * saturation. Input values are multiplied by the scaling factor before
 * conversion. Infinite results are clamped to the maximum-magnitude finite
 * value of the same sign.
 */
void skl_cvt_sat_f32_f8e5m2_zvfofp8min(uint8_t *pDst, const float *pSrc,
                                       float scaling_factor, size_t n);

/**
 * @brief Converts E4M3 OFP8 values to BF16 format.
 *
 * @param pDst - Output array where the BF16 values are stored.
 * @param pSrc - Input array where the E4M3 values are stored.
 * @param n - Number of elements to convert.
 *
 * Converts 8-bit E4M3 OFP format values to 16-bit brain floating-point format.
 * This is a widening conversion that preserves full precision.
 */
void skl_cvt_f8e4m3_bf16_zvfofp8min(__bf16 *pDst, const uint8_t *pSrc,
                                    size_t n);

/**
 * @brief Converts E5M2 OFP8 values to BF16 format.
 *
 * @param pDst - Output array where the BF16 values are stored.
 * @param pSrc - Input array where the E5M2 values are stored.
 * @param n - Number of elements to convert.
 *
 * Converts 8-bit E5M2 OFP format values to 16-bit brain floating-point format.
 * This is a widening conversion that preserves full precision.
 */
void skl_cvt_f8e5m2_bf16_zvfofp8min(__bf16 *pDst, const uint8_t *pSrc,
                                    size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
