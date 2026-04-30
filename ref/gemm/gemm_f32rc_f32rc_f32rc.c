// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>

SKL_FUNC void skl_gemm_f32rc_f32rc_f32rc_ref(size_t m, size_t n, size_t k,
                                             float alpha, const float *a,
                                             size_t rsa, size_t csa,
                                             const float *b, size_t rsb,
                                             size_t csb, float beta, float *c,
                                             size_t rsc, size_t csc) {
  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj++) {
      float acc = 0;
      for (size_t kk = 0; kk < k; kk++) {
        acc += a[ii * rsa + kk * csa] * b[kk * rsb + jj * csb];
      }
      c[ii * rsc + jj * csc] = beta * c[ii * rsc + jj * csc] + alpha * acc;
    }
  }
}
