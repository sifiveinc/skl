// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

// NOLINTNEXTLINE(readability-non-const-parameter)
SKL_FUNC void skl_logistic_5u_f32_xsfvfexp32e(float *out, const float *in,
                                              size_t n) {
  size_t vl;
  const float b = -0x1.9fe370p+6f; /* Input lower bound */
  const float S = +0x1.35512ep-1f; /* Input split ratio */

  // clang-format off
  for (; n; in += vl, out += vl, n -= vl)
    __asm__ volatile(
      "     vsetvli  %[vl], %[n], e32, m8, ta, ma \n"
      /* 0. Load and clamp */
      "\t   vle32.v     v8,  (%[in])     \n"
      "\t   vfsgnj.vf   v8,  v8, %[b]    \n"
      "\t   vfmax.vf    v8,  v8, %[b]    \n"
      /* 1. Split x = a + e */
      "\t   vfmul.vf   v16,  v8, %[S]    \n"
      "\t   vfsub.vv   v24, v16, v8      \n"
      /* 2. Compute o = exp(a) and p = exp(e) */
      "\t   sf.vfexp.v v16, v16          \n"
      "\t   sf.vfexp.v v24, v24          \n"
      "\t   vmslt.vi    v0,  v8, 0       \n"
      "\t   vfadd.vv    v8, v16, v24     \n"
      /* 3. Assemble numerator: n = x>0 ? p : o */
      "\t   vmerge.vvm v16, v16, v24, v0 \n"
      "\t   vfmv.v.f    v0, %[one]       \n"
      /* 4. Approximate 1 / (o + p) */
      "\t   vfrec7.v   v24, v8 \n"
      "\t   vfnmsac.vv  v0, v8, v24      \n"
      "\t   vfmadd.vv  v24, v0, v24      \n"
      "\t   vfmul.vv    v0, v0, v0       \n"
      "\t   vfmadd.vv  v24, v0, v24      \n"
      /* 5. Finally, n * r ~ n / d */
      "\t   vfmul.vv   v16, v16, v24     \n"
      "\t   vse32.v    v16, (%[out])"
      : [vl] "=&r"(vl)
      : [n] "r"(n), [in] "r"(in), [out] "r"(out), [S] "f"(S),
        [b] "f"(b), [one] "f"(1.0f)
      : "vtype", "vl", "memory");
  // clang-format on
}
