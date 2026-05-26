// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Pack and pad an 8-bit row-major matrix into block-row-major format
 * with 4 x 1 blocks.
 *
 * @param m - Number of rows in the input matrix.
 * @param n - Number of columns in the input matrix.
 * @param src - Pointer to the input matrix in row-major format.
 * @param rs - Stride between rows of the input matrix.
 * @param dst - Pointer to the packed output matrix.
 * @param rs1 - Row stride between blocks of the output matrix.
 * @param pad - Padding value.
 *
 * @note
 * The output buffer must be at least ((m + 3) / 4) * 4 * n bytes.
 *
 * @note
 * This kernel can be used to pack the B matrix for
 * skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq.
 */
void skl_pack_e8_e8p4x1c_zve32x(size_t m, size_t n, const uint8_t *src,
                                size_t rs, uint8_t *dst, size_t rs1,
                                uint8_t pad);

#if defined(__cplusplus)
} // extern "C"
#endif
