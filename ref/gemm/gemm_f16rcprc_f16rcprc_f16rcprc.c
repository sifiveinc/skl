// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>

SKL_FUNC void skl_gemm_f16rcprc_f16rcprc_f16rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    _Float16 alpha, const _Float16 *a_pack, size_t rsa0, size_t csa0,
    size_t rsa1, size_t csa1, const _Float16 *b_pack, size_t rsb0, size_t csb0,
    size_t rsb1, size_t csb1, _Float16 beta, _Float16 *c_pack, size_t rsc0,
    size_t csc0, size_t rsc1, size_t csc1) {
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      _Float16 *cp_block = c_pack + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          _Float16 acc = 0;
          for (size_t kk1 = 0; kk1 < k1; ++kk1) {
            const _Float16 *ap_block = a_pack + ii1 * rsa1 + kk1 * csa1;
            const _Float16 *bp_block = b_pack + kk1 * rsb1 + jj1 * csb1;
            for (size_t kk0 = 0; kk0 < k0; ++kk0) {
              _Float16 a_val = ap_block[ii0 * rsa0 + kk0 * csa0];
              _Float16 b_val = bp_block[kk0 * rsb0 + jj0 * csb0];
              acc += a_val * b_val;
            }
          }
          cp_block[ii0 * rsc0 + jj0 * csc0] =
              beta * cp_block[ii0 * rsc0 + jj0 * csc0] + alpha * acc;
        }
      }
    }
  }
}
