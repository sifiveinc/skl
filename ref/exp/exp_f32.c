// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <math.h>
#include <stddef.h>

SKL_FUNC void skl_exp_f32_ref(float *out, const float *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = expf(in[i]);
  }
}
