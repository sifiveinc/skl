// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

/* This reference is based on the observation that the GELU is a
   calculated offset from the ReLU */
SKL_FUNC void skl_gelu_f32_ref(float *out, const float *in, size_t n) {
  const double sqrt2 = 0x1.6a09e667f3bcdp0;
  for (size_t i = 0; i < n; i++) {
    float x = in[i];
    float b = (x > 0) ? x : -0.f; /* the RELU */
    float h = -0.5f * fabsf(x);
    double e = erfc(-sqrt2 * h); /* = erfc(a/sqrt(2)) */
    double g = fma(h, e, b);
    out[i] = (float)g;
  }
}
