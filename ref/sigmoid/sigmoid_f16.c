// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_f16_ref(_Float16 *out, _Float16 beta,
                                  const _Float16 *x, const _Float16 *y,
                                  const _Float16 *up, _Float16 delta,
                                  size_t n) {
  if (!x)
    return;

  for (size_t i = 0; i < n; i++) {
    float o;
    float a = x[i];
    /* Logistic */
    float b = beta * a;
    if (b >= 0)
      o = 1 / (1 + expf(-b));
    else {
      float e = expf(b);
      o = e / (1 + e);
    }
    /* Linear unit */
    if (y)
      o *= y[i];
    /* Gate */
    if (up) {
      o *= (up[i] + delta);
    }
    out[i] = (_Float16)o;
  }
}

SKL_FUNC void skl_logistic_f16_ref(_Float16 *out, const _Float16 *x, size_t n) {
  skl_sigmoid_f16_ref(out, (_Float16)1, x, NULL, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_silu_f16_ref(_Float16 *out, const _Float16 *x, size_t n) {
  skl_sigmoid_f16_ref(out, (_Float16)1, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swish_f16_ref(_Float16 *out, _Float16 beta, const _Float16 *x,
                                size_t n) {
  skl_sigmoid_f16_ref(out, beta, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_glu_f16_ref(_Float16 *out, const _Float16 *x,
                              const _Float16 *y, size_t n) {
  skl_sigmoid_f16_ref(out, (_Float16)1, y, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swiglu_f16_ref(_Float16 *out, const _Float16 *gate,
                                 const _Float16 *up, _Float16 delta, size_t n) {
  skl_sigmoid_f16_ref(out, (_Float16)1, gate, gate, up, delta, n);
}
