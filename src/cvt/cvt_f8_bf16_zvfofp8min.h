// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

/**
 * @file cvt_f8_bf16_zvfofp8min.h
 * @brief OFP8 to BF16 Conversion Functions
 *
 * This header provides vectorized widening conversion functions from
 * OFP8 formats (E4M3 and E5M2) to BF16 using the
 * RISC-V Zvfofp8min extension.
 *
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

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
