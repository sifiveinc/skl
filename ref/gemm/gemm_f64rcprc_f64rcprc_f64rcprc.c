// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>

SKL_FUNC void skl_gemm_f64rcprc_f64rcprc_f64rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    double alpha, const double *a, size_t rsa0, size_t csa0, size_t rsa1,
    size_t csa1, const double *b, size_t rsb0, size_t csb0, size_t rsb1,
    size_t csb1, double beta, double *c, size_t rsc0, size_t csc0, size_t rsc1,
    size_t csc1) {
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      double *c_block = c + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          double acc = 0;
          for (size_t kk1 = 0; kk1 < k1; ++kk1) {
            const double *a_block = a + ii1 * rsa1 + kk1 * csa1;
            const double *b_block = b + kk1 * rsb1 + jj1 * csb1;
            for (size_t kk0 = 0; kk0 < k0; ++kk0) {
              double a_val = a_block[ii0 * rsa0 + kk0 * csa0];
              double b_val = b_block[kk0 * rsb0 + jj0 * csb0];
              acc += a_val * b_val;
            }
          }
          c_block[ii0 * rsc0 + jj0 * csc0] =
              beta * c_block[ii0 * rsc0 + jj0 * csc0] + alpha * acc;
        }
      }
    }
  }
}
