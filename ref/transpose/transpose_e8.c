// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_transpose_e8_ref(size_t m, size_t n,
                                      const uint8_t *SKL_RESTRICT a, size_t rsa,
                                      uint8_t *SKL_RESTRICT at, size_t rsat) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      at[j * rsat + i] = a[i * rsa + j];
    }
  }
}
