// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @file skl-common.h
 * @brief Definitions shared by all SKL kernels.
 *
 * This is the only file on which any SKL source may depend.
 * To extract a function from the SKL project, simply copy its source file,
 * its declaration header, and `skl-common.h`.
 */

#if !defined(SKL_FUNC)
/** Declaration decorator for all public SKL functions.
 *
 * By default, this macro is defined to nothing, but when SKL is built as a
 * header, it can be defined to `static inline` to prevent redefinitions and
 * linker symbol conflicts.
 *
 * @note
 * Private functions (helper functions not exposed by the SKL API) are to be
 * declared with `SKL_FUNC_PRIVATE`.
 */
#define SKL_FUNC
#endif

#if !defined(SKL_FUNC_PRIVATE)
/** Declaration decorator for private functions within a SKL kernel.
 *
 * By default, this is macro is defined to `static`, but when SKL is built
 * as a header, it can be defined to `static inline` to prevent redefinitions
 * and linker symbol conflicts.
 *
 * @note
 * Private functions (helper functions not exposed by the SKL API) are to be
 * declared with `SKL_FUNC_PRIVATE`.
 */
#define SKL_FUNC_PRIVATE static
#endif

/** Declaration decorator for shared SKL utility functions.
 *
 * For utility functions shared between SKL kernels. Since not all functions
 * will be used by every kernel, definitions in `skl-common.h` are marked static
 * inline to avoid warnings about unused functions.
 *
 * These functions are considered private to SKL, and not part of the public
 * API.
 */
#define SKL_FUNC_UTIL static inline

/** Portable restrict pointer qualifier for C and C++ */
#if !defined(__cplusplus)
#define SKL_RESTRICT restrict
#else
#if defined(__GNUC__) || defined(__clang__)
#define SKL_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SKL_RESTRICT __restrict
#else
#define SKL_RESTRICT
#endif
#endif

/**
 * @brief SKL memcpy
 *
 * This version of memcpy is provided so that kernels requiring it can avoid a
 * dependency on libc.
 */
SKL_FUNC_UTIL void *skl_memcpy(void *dest, const void *src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    ((uint8_t *)dest)[i] = ((uint8_t *)src)[i];
  }

  return dest;
}
