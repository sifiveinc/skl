#include "skl-ref.h" // NOLINT(misc-include-cleaner)
#include "skl-test-driver.h"
#include "skl.h" // NOLINT(misc-include-cleaner)
#include "softmax/softmax_f32.h"
#include <stddef.h>

/**
 * @brief Test cases for 2D Softmax
 */

// clang-format off
#define FUNC_PARAMS(VAR)                                                       \
  .func = (void *)skl_softmax_2d_f32_##VAR, .name = "skl_softmax_2d_f32_" #VAR

#define VARIANT_BENCHMARKS(VAR)                                                \
  {BENCHMARK, FUNC_PARAMS(VAR), .beta = 1.0f, .m = 128, .n = 128},             \
  {BENCHMARK, FUNC_PARAMS(VAR), .beta = 1.1f, .m = 128, .n = 128}

#define DOMAIN_TESTS(...)                                                      \
  {__VA_ARGS__, .a.min = -1, .a.max = +1},                                     \
  {__VA_ARGS__, .a.min = +40, .a.max = +100}

#define ALIGN(N) (((N) / 128) * 128 + 128)

#define ALIGN_TESTS(N,...)                                                     \
  DOMAIN_TESTS(__VA_ARGS__, .n = N),                                           \
  DOMAIN_TESTS(__VA_ARGS__, .n = N, .rss = ALIGN(N), .rsa = ALIGN(N))

#define COLS_TESTS(...)                                                        \
  ALIGN_TESTS(1, __VA_ARGS__),                                                 \
  ALIGN_TESTS(27, __VA_ARGS__),                                                \
  ALIGN_TESTS(131, __VA_ARGS__),                                               \
  ALIGN_TESTS(380, __VA_ARGS__),                                               \
  ALIGN_TESTS(732, __VA_ARGS__),                                               \
  ALIGN_TESTS(1312, __VA_ARGS__)

#define ROWS_TESTS(...)                                                        \
  COLS_TESTS(__VA_ARGS__, .m = 1),                                             \
  COLS_TESTS(__VA_ARGS__, .m = 12),                                            \
  COLS_TESTS(__VA_ARGS__, .m = 37),                                            \
  COLS_TESTS(__VA_ARGS__, .m = 135)

#define BETA_TESTS(...)                                                        \
  ROWS_TESTS(__VA_ARGS__, .beta = 0x1.62e43p-1f),                              \
  ROWS_TESTS(__VA_ARGS__, .beta = 1.0f),                                       \
  ROWS_TESTS(__VA_ARGS__, .beta = 0x1.0a2b24p0f)

#define ZERO_TESTS(...)                                                        \
  {__VA_ARGS__, .m = 0, .n = 8, .beta = 1.0f},                                 \
  {__VA_ARGS__, .m = 8, .n = 0, .beta = 1.0f},                                 \
  BETA_TESTS(__VA_ARGS__)

#define VARIANT_TESTS(VAR) ZERO_TESTS(TEST, FUNC_PARAMS(VAR))

softmax_f32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
#ifdef __riscv_zve32f
  VARIANT_BENCHMARKS(zve32f),
#endif
#ifdef __riscv_xsfvfexpa
  VARIANT_BENCHMARKS(xsfvfexpa),
#endif
#ifdef __riscv_xsfvfexp32e
  VARIANT_BENCHMARKS(xsfvfexp32e),
#endif
#endif
#ifdef SKL_ENABLE_TESTS
  VARIANT_TESTS(ref),
#ifdef __riscv_zve32f
  VARIANT_TESTS(zve32f),
#endif
#ifdef __riscv_xsfvfexpa
  VARIANT_TESTS(xsfvfexpa),
#endif
#ifdef __riscv_xsfvfexp32e
  VARIANT_TESTS(xsfvfexp32e),
#endif
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_softmax_2d_f32",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_f32_t),
                                 .tests = tests};

typedef void (*skl_softmax_2d_f32_t)(float *, size_t, const float *, size_t,
                                     const float, size_t, size_t);

static void execute(skl_test_t *t) {
  const softmax_f32_t *h = (softmax_f32_t *)t->harness;
  skl_softmax_2d_f32_t fn = (skl_softmax_2d_f32_t)(h->func);
  fn(h->ctx.s, h->rss, h->a.data, h->rsa, h->beta, h->m, h->n);
}

int main(void) {
  // Set execution step
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].steps.execute = execute;
    if (tests[i].steps.verify == NULL)
      tests[i].steps.warmup = execute;
  }
  // Run the suite
  return skl_test_driver_run_suite(&suite);
}
