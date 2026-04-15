// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zvfofp4min)
#error This file requires the Zvfofp4min extension
#endif

/**
 * @file cvt_f4_f8_zvfofp4min.h
 * @brief OFP4 to OFP8 Conversion Functions
 *
 * This header provides vectorized conversion functions from 4-bit OFP4
 * formats to 8-bit OFP8 E4M3 format using the RISC-V Zvfofp4min extension.
 *
 * ## Memory Layout Requirements:
 * - The start address of the input array must be byte-aligned
 * - Each byte contains two 4-bit values: lower 4 bits (the first element),
 * upper 4 bits (the second element)
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Converts packed E2M1 OFP4 values to E4M3 OFP8 format.
 *
 * @param pDst - Output array where E4M3 values are stored.
 * @param pSrc - Input array where the packed E2M1 values are stored.
 * @param n - Number of E2M1 elements to convert.
 *
 * Converts packed 4-bit E2M1 OFP values to 8-bit E4M3 OFP format. This is a
 * widening conversion that preserves full precision. Input format uses packed
 * storage where each byte contains two 4-bit E2M1 values: lower 4 bits contain
 * the first element, upper 4 bits contain the second element.
 *
 * Memory layout example:
 * ```
 * Input byte:  [E2M1[1]] [E2M1[0]]
 *              bits 7-4   bits 3-0
 * ```
 *
 * @note The start address of the input array must be byte-aligned, but the
 * length n is allowed to be odd. The function handles the case of odd n
 * correctly.
 */
void skl_cvt_f4e2m1_f8e4m3_zvfofp4min(uint8_t *pDst, const uint8_t *pSrc,
                                      size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
