// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    int32_t alpha, const int8_t *a, size_t rsa0, size_t csa0, size_t rsa1,
    size_t csa1, const int8_t *b, size_t rsb0, size_t csb0, size_t rsb1,
    size_t csb1, int32_t beta, int32_t *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1) {
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      int32_t *c_block = c + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          int32_t acc = 0;
          for (size_t kk1 = 0; kk1 < k1; ++kk1) {
            const int8_t *a_block = a + ii1 * rsa1 + kk1 * csa1;
            const int8_t *b_block = b + kk1 * rsb1 + jj1 * csb1;
            for (size_t kk0 = 0; kk0 < k0; ++kk0) {
              int8_t a_val = a_block[ii0 * rsa0 + kk0 * csa0];
              int8_t b_val = b_block[kk0 * rsb0 + jj0 * csb0];
              acc += (int32_t)a_val * (int32_t)b_val;
            }
          }
          c_block[ii0 * rsc0 + jj0 * csc0] =
              beta * c_block[ii0 * rsc0 + jj0 * csc0] + alpha * acc;
        }
      }
    }
  }
}
