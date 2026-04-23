#include "skl-ref.h" // NOLINT(misc-include-cleaner)
#include "skl-test-driver.h"
#include "skl.h" // NOLINT(misc-include-cleaner)
#include "softmax/softmax_f32.h"
#include <stddef.h>

/**
 * @brief Test cases for 1D Softmax
 */

// clang-format off
#define FUNC_PARAMS(VAR)                                                       \
  .func = (void *)skl_softmax_f32_##VAR, .name = "skl_softmax_f32_" #VAR

#define BASE_PARAMS(VAR) FUNC_PARAMS(VAR), .m = 1

#define VARIANT_BENCHMARKS(VAR)                                                \
  {BENCHMARK, BASE_PARAMS(VAR), .beta = 1.0f, .n = 1024},                      \
  {BENCHMARK, BASE_PARAMS(VAR), .beta = 1.1f, .n = 1024}

#define DOMAIN_TESTS(...)                                                      \
  {__VA_ARGS__, .a.min = -1, .a.max = +1},                                     \
  {__VA_ARGS__, .a.min = +40, .a.max = +100}

#define COLS_TESTS(...)                                                        \
  DOMAIN_TESTS(__VA_ARGS__, .n = 0),                                           \
  DOMAIN_TESTS(__VA_ARGS__, .n = 1),                                           \
  DOMAIN_TESTS(__VA_ARGS__, .n = 27),                                          \
  DOMAIN_TESTS(__VA_ARGS__, .n = 131),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 380),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 732),                                         \
  DOMAIN_TESTS(__VA_ARGS__, .n = 1312)

#define BETA_TESTS(...)                                                        \
  COLS_TESTS(__VA_ARGS__, .beta = 0x1.62e43p-1f),                              \
  COLS_TESTS(__VA_ARGS__, .beta = 1.0f),                                       \
  COLS_TESTS(__VA_ARGS__, .beta = 0x1.0a2b24p0f)

#define VARIANT_TESTS(VAR) BETA_TESTS(TEST, BASE_PARAMS(VAR))

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

static skl_test_suite_t suite = {.name = "skl_softmax_f32",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(softmax_f32_t),
                                 .tests = tests};

typedef void (*skl_softmax_f32_t)(float *, const float *, const float,
                                  const size_t);

static void execute(skl_test_t *t) {
  const softmax_f32_t *h = (softmax_f32_t *)t->harness;
  skl_softmax_f32_t fn = (skl_softmax_f32_t)(h->func);
  fn(h->ctx.s, h->a.data, h->beta, h->n);
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
