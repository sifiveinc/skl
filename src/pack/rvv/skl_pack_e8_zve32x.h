// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

void skl_pack_e8rc_e8rcbrc_zve32x(
  size_t m,           // Num. rows in input matrix
  size_t n,           // Num. columns in input matrix
  const uint8_t* src,  // Input matrix
  size_t rs,          // Row stride of input matrix
  size_t cs,          // Column stride of input matrix
  size_t m0,          // Num. rows in a block of the input matrix
  size_t n0,          // Num. columns in a block of the input matrix
  uint8_t* dst,        // Output packed matrix [m1 x n1]
  size_t rs0,         // Row stride within a block of the output matrix
  size_t cs0,         // Column stride within a block of the output matrix
  size_t rs1,         // Row stride between blocks of the output matrix
  size_t cs1          // Column stride between blocks of the output matrix
);


#if defined(__cplusplus)
} // extern "C"
#endif
