// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <stddef.h>

SKL_FUNC void skl_gemm_bf16rcprc_bf16rcprc_f32rcprc_scalar(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    float alpha, const __bf16 *a_pack, size_t rsa0, size_t csa0, size_t rsa1,
    size_t csa1, const __bf16 *b_pack, size_t rsb0, size_t csb0, size_t rsb1,
    size_t csb1, float beta, float *c_pack, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1) {
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      float *cp_block = c_pack + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          float acc = 0;
          for (size_t kk1 = 0; kk1 < k1; ++kk1) {
            const __bf16 *ap_block = a_pack + ii1 * rsa1 + kk1 * csa1;
            const __bf16 *bp_block = b_pack + kk1 * rsb1 + jj1 * csb1;
            for (size_t kk0 = 0; kk0 < k0; ++kk0) {
              __bf16 a_val = ap_block[ii0 * rsa0 + kk0 * csa0];
              __bf16 b_val = bp_block[kk0 * rsb0 + jj0 * csb0];
              acc += (float)a_val * (float)b_val;
            }
          }
          cp_block[ii0 * rsc0 + jj0 * csc0] =
              beta * cp_block[ii0 * rsc0 + jj0 * csc0] + alpha * acc;
        }
      }
    }
  }
}
