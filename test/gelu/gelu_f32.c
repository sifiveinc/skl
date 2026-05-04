#if !defined(__riscv_zve32f)
#error This file require the Zve32f extension
#endif

#if !defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#error Must define at least one of ENABLE_TEST or ENABLE_BENCHMARK
#endif

#if !defined(NUM_ELEMS)
#if defined(ENABLE_TEST) && !defined(ENABLE_BENCHMARK)
#define NUM_ELEMS 8192
#else
#define NUM_ELEMS 1024 // Default input length
#endif
#endif

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#if defined(ENABLE_TEST)
#include <math.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ALIGN = 4096 };
__attribute__((aligned(ALIGN))) float input[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float output[NUM_ELEMS];
__attribute__((aligned(ALIGN))) float ref_output[NUM_ELEMS];

static void init_random(float *arr, size_t len) {
#if defined(ENABLE_TEST)
  // If testing, fill an array with random floats in [-14.356,
  // +5.348], and a few special cases.
  const float min = -0x1.cb64e8p3f; // ~14.356
  const float max = +0x1.563db2p2f; //  ~5.348
  float frac;
  for (size_t i = 0; i < len; ++i) {
    // clang-format off
    switch (i) {
    case 0: arr[0] = -INFINITY; break;
    case 1: arr[1] = +INFINITY; break;
    case 2: arr[2] = -50.f; break;
    case 3: arr[3] = +50.f; break;
    case 4: arr[4] = nanf(""); break;
    case 5: arr[5] = -0.f; break;
    case 6: arr[6] = +0.f; break;
    default:
      frac = (float)rand() / (float)RAND_MAX;
      arr[i] = frac * (max - min) + min;
    }
    // clang-format on
  }
#else
  // Otherwise, save simulation time using increasing ints
  int32_t next = 0;
  for (size_t i = 0; i < len; i++)
    arr[i] = (float)(next++);
#endif
}

#if defined(ENABLE_TEST)
/** Scalar GELU using FP64 intermediates for high accuracy */
static void reference_gelu_f32(float *out, const float *in, size_t n) {
  const double sqrt2 = 0x1.6a09e667f3bcdp0;
  for (size_t i = 0; i < n; ++i) {
    double h = 0.5 * in[i];
    if (h > 0.) {
      double e = erf(h * sqrt2);
      out[i] = (float)fma(h, e, h);
    } else {
      if (isinf(h)) {
        out[i] = -0.0f;
      } else {
        // Protect against catastrophic loss of precision for large
        // magnitude inputs less than 0.  Note also that the
        // formulation with FMA above produces the wrong sign for such
        // inputs.
        double e = erfc(-h * sqrt2);
        out[i] = (float)(h * e);
      }
    }
  }
}

enum {
  CONFORM = 0,
  NINF_NAN = 1, /* gelu(-infty) may be NaN */
};
static int check_error(const char *name, const float *in, const float *res,
                       const float *ref, float spanning_tol, float gen2_tol,
                       float gen1_tol, float ge0_tol, uint8_t specials,
                       size_t len) {
  // NOLINTBEGIN(*-signed-bitwise,*-braces-around-statements,*-isolate-declaration)
  float spanning_max = 0, gen2_max = 0, gen1_max = 0, ge0_max = 0;
  for (size_t i = 0; i < len; i++) {
    float err = 0;
    if (in[i] == -INFINITY && isnan(res[i])) {
      if (!(specials & NINF_NAN)) {
        printf("%15s : result is NaN for -infty\n", name);
        err = INFINITY;
      } // else err = 0
    } else {
      err = skl_abs_error_ulp_f32(res[i], ref[i]);
      if (err > spanning_tol) {
        uint32_t x, Y, y;
        memcpy(&x, &in[i], sizeof(float));
        memcpy(&Y, &res[i], sizeof(float));
        memcpy(&y, &ref[i], sizeof(float));
        printf("%15s : gelu(%" PRIx32 ") = %#" PRIx32 ", %#" PRIx32 " => ",
               name, x, Y, y);
        print_float(err);
        printf(" ulp\n");
      }
    }
    if (err > spanning_max)
      spanning_max = err;
    if (in[i] >= -2.f && err > gen2_max)
      gen2_max = err;
    if (in[i] >= -1.f && err > gen1_max)
      gen1_max = err;
    if (in[i] >= 0.f && err > ge0_max)
      ge0_max = err;
  }
  // NOLINTEND(*-signed-bitwise,*-braces-around-statements,*-isolate-declaration)
  int ret = spanning_max > spanning_tol || gen2_max > gen2_tol ||
            gen1_max > gen1_tol || ge0_max > ge0_tol;
  // clang-format off
  printf ("%15s : maximum error (", name);
  print_float(spanning_max); printf(", ");
  print_float(gen2_max);     printf(", ");
  print_float(gen1_max);     printf(", ");
  print_float(ge0_max);
  printf(") ulp%s\n", ret ? " !" : "");
  // clang-format on
  return ret;
}
#endif

int main(void) {
  int ret = 0; // return value
  printf("Measuring %d-element GELU:\n", NUM_ELEMS);
  init_random(input, NUM_ELEMS); // input data

#if defined(ENABLE_TEST)
  memset(ref_output, 0, NUM_ELEMS * sizeof(*ref_output));
  reference_gelu_f32(ref_output, input, NUM_ELEMS);
#define CHECK_RESULT(FUNCTION, NAME, TOL, GEN2, GEN1, GE0, SPECIALS)           \
  ret += check_error(NAME, input, output, ref_output, TOL, GEN2, GEN1, GE0,    \
                     SPECIALS, NUM_ELEMS);
#else
#define CHECK_RESULT(FUNCTION, NAME, TOL, GEN2, GEN1, GE0, SPECIALS)
#endif

#define RUN_(FUNCTION, NAME, TOL, GEN2, GEN1, GE0, SPECIALS)                   \
  memset(output, 0, NUM_ELEMS * sizeof(*output));                              \
  SKL_BENCHMARK_RUN(NAME, NUM_ELEMS, SKL_TEST_WARMUP, FUNCTION, output, input, \
                    NUM_ELEMS);                                                \
  CHECK_RESULT(FUNCTION, NAME, TOL, GEN2, GEN1, GE0, SPECIALS);

#define PASTE3(a, b, c) a##b##c
#define RUN(VARIANT, ...)                                                      \
  RUN_(PASTE3(skl_gelu_, VARIANT, _f32_zve32f), "zve32f," #VARIANT, __VA_ARGS__)

  // NOLINTBEGIN(*-signed-bitwise)
  // clang-format off
  RUN(p9,  1.7e7, 5.401e5, 9.384e4, 3.703e4, NINF_NAN);
  RUN(p13, 1.7e7, 6.908e4,    6810,    2300, NINF_NAN);
  RUN(p17, 1.7e7, 2.732e3,     239,     239, NINF_NAN);
  RUN(rat, 1.7e7,      62,       5,       4, NINF_NAN);
  // clang-format on
  // NOLINTEND(*-signed-bitwise)

  return ret > 0;
}
