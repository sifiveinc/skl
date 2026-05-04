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
  vfloat32m8_t w, x, y, z;	   /* NOLINT(*-isolate-declaration) */

  // clang-format off
  for (; n; in += vl, out += vl, n -= vl)
    __asm__ volatile(
      "     vsetvli  %[vl], %[n], e32, m8, ta, ma \n"
      /* 0. Load and clamp */
      "\t   vle32.v    %[x], (%[in])          \n"
      "\t   vfsgnj.vf  %[x], %[x], %[b]       \n"
      "\t   vfmax.vf   %[x], %[x], %[b]       \n"
      /* 1. Split x = a - e */
      "\t   vfmul.vf   %[y], %[x], %[S]       \n"
      "\t   vfsub.vv   %[z], %[y], %[x]       \n"
      /* 2. Compute o = exp(a) and p = exp(e) */
      "\t   sf.vfexp.v %[y], %[y]             \n"
      "\t   sf.vfexp.v %[z], %[z]             \n"
      "\t   vmslt.vi   %[w], %[x], 0          \n"
      "\t   vfadd.vv   %[x], %[y], %[z]       \n"
      /* 3. Assemble numerator: n = x>0 ? p : o */
      "\t   vmerge.vvm %[y], %[y], %[z], %[w] \n"
      "\t   vfmv.v.f   %[w], %[one]           \n"
      /* 4. Approximate r = 1 / (d = o + p) */
      "\t   vfrec7.v   %[z], %[x] \n"
      "\t   vfnmsac.vv %[w], %[x], %[z]       \n"
      "\t   vfmadd.vv  %[z], %[w], %[z]       \n"
      "\t   vfmul.vv   %[w], %[w], %[w]       \n"
      "\t   vfmadd.vv  %[z], %[w], %[z]       \n"
      /* 5. Finally, n / d ~ n * r */
      "\t   vfmul.vv   %[y], %[y], %[z]       \n"
      "\t   vse32.v    %[y], (%[out])"
      : [vl] "=&r"(vl),
        [w] "=vr"(w), [x] "=vd"(x), [y] "=vd"(y), [z] "=vd"(z)
      : [n] "r"(n), [in] "r"(in), [out] "r"(out),
        [b] "f"(b), [S] "f"(S), [one] "f"(1.0f)
      : "vtype", "vl", "memory");
  // clang-format on
}
