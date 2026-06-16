// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_logistic_1u_f32_ref(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double x = in[i];
    double e = exp(-x);
    out[i] = (float)(1 / (1. + e));
  }
}
