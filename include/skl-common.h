// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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

/*
 * Xsfmm ABI macros:
 *
 * Placeholders for function attributes that indicate how Xsfmm matrix tile
 * state is used. These will be replaced with appropriate function attributes
 * once they are provided by the compiler.
 */

/** Xsfmm state shared with caller. Function reads state but does not modify. */
#define SKL_XSFMM_IN

/**
 * Xsfmm state shared with caller. Function ignores incoming state and
 * overwrites it.
 */
#define SKL_XSFMM_OUT

/** Xsfmm state shared with caller. Function reads and modifies state. */
#define SKL_XSFMM_INOUT

/** Function creates a new scope for Xsfmm state. */
#define SKL_XSFMM_NEW

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
 * @brief An instruction scheduling barrier.
 *
 * This function acts as a barrier to instruction scheduling, by pretending to
 * have arbitrary side-effects. This is a necessary hack in some cases to
 * override pessimal scheduling decisions by the compiler when writing RVV
 * intrinsics.
 *
 * @note This is very much a hack, and should be used with caution. It is not
 *       guaranteed to prevent reordering of instructions in all cases, and
 *       in particular only works when the instructions in question access
 *       memory.
 */
SKL_FUNC_UTIL void skl_instruction_schedule_barrier(void) {
  __asm__ volatile("" ::: "memory");
}

typedef void *(skl_memcpy)(void *dest, const void *src, size_t n);
typedef void *(skl_memset)(void *str, int c, size_t n);
