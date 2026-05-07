// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

/**
 * @file cvt_bf16_f8_zvfofp8min_zvfbfmin.h
 * @brief BF16 to OFP8 Conversion Functions
 *
 * This header provides vectorized narrowing conversion functions from
 * BF16 to OFP8 formats (E4M3 and E5M2) using the
 * RISC-V Zvfofp8min and Zvfbfmin extension.
 *
 * ## Function Variants:
 *
 * ### Narrowing Conversions (BF16 → OFP8):
 * - **Standard**: `skl_cvt_bf16_{output}_zvfofp8min_zvfbfmin()`
 *   - E4M3: Infinite values are converted to the canonical NaN (E4M3 cannot
 * represent infinity)
 *   - E5M2: Infinite values are preserved as infinity in the target format
 * - **Saturating**: `skl_cvt_sat_bf16_{output}_zvfofp8min_zvfbfmin()`
 *   - Infinite results are clamped to the maximum-magnitude finite value of the
 * same sign
 *
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
 * @brief Converts BF16 values to E4M3 OFP8 format with scaling.
 *
 * @param pDst - Output array where E4M3 values are stored.
 * @param pSrc - Input array where the BF16 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 16-bit brain floating-point values to 8-bit E4M3 OFP format. Input
 * values are multiplied by the scaling factor before conversion. Since E4M3
 * cannot represent infinity, infinite results are converted to the canonical
 * NaN (`0x7f`).
 */
void skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin(uint8_t *pDst, const __bf16 *pSrc,
                                             float scaling_factor, size_t n);

/**
 * @brief Converts BF16 values to E4M3 OFP8 format with scaling and saturation.
 *
 * @param pDst - Output array where E4M3 values are stored.
 * @param pSrc - Input array where the BF16 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 16-bit brain floating-point values to 8-bit E4M3 OFP format with
 * saturation. Input values are multiplied by the scaling factor before
 * conversion. Infinite results are clamped to the maximum-magnitude finite
 * value of the same sign.
 */
void skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin(uint8_t *pDst,
                                                 const __bf16 *pSrc,
                                                 float scaling_factor,
                                                 size_t n);

/**
 * @brief Converts BF16 values to E5M2 OFP8 format with scaling.
 *
 * @param pDst - Output array where E5M2 values are stored.
 * @param pSrc - Input array where the BF16 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 16-bit brain floating-point values to 8-bit E5M2 OFP format. Input
 * values are multiplied by the scaling factor before conversion. Infinite
 * results are preserved as infinity in the target format.
 */
void skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin(uint8_t *pDst, const __bf16 *pSrc,
                                             float scaling_factor, size_t n);

/**
 * @brief Converts BF16 values to E5M2 OFP8 format with scaling and saturation.
 *
 * @param pDst - Output array where E5M2 values are stored.
 * @param pSrc - Input array where the BF16 values are stored.
 * @param scaling_factor - Scaling factor applied before conversion (1.0f
 * indicates scaling is not required.)
 * @param n - Number of elements to convert.
 *
 * Converts 16-bit brain floating-point values to 8-bit E5M2 OFP format with
 * saturation. Input values are multiplied by the scaling factor before
 * conversion. Infinite results are clamped to the maximum-magnitude finite
 * value of the same sign.
 */
void skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin(uint8_t *pDst,
                                                 const __bf16 *pSrc,
                                                 float scaling_factor,
                                                 size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
