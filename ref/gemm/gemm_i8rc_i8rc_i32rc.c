// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_gemm_i8rc_i8rc_i32rc_ref(size_t m, size_t n, size_t k,
                                           int32_t alpha, const int8_t *a,
                                           size_t rsa, size_t csa,
                                           const int8_t *b, size_t rsb,
                                           size_t csb, int32_t beta, int32_t *c,
                                           size_t rsc, size_t csc) {
  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj++) {
      int32_t acc = 0;
      for (size_t kk = 0; kk < k; kk++) {
        acc +=
            (int32_t)a[ii * rsa + kk * csa] * (int32_t)b[kk * rsb + jj * csb];
      }
      c[ii * rsc + jj * csc] = beta * c[ii * rsc + jj * csc] + alpha * acc;
    }
  }
}
