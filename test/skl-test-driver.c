#include "skl-test-driver.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SKL_TEST_DRIVER_ALIGNMENT = 4096 };

#if !defined(SKL_TEST_LOG_LEVEL)
#define SKL_TEST_LOG_LEVEL SKL_TEST_LOG_INFO
#endif

#if defined(__riscv)
#if __riscv_xlen == 32
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint32_t lo;                                                               \
    uint32_t hi0, hi1;                                                         \
    /* guard against overflow between reads of lo/hi counter halves */         \
    do {                                                                       \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi0));                      \
      __asm__ volatile("rd" #COUNTER " %0" : "=r"(lo));                        \
      __asm__ volatile("rd" #COUNTER "h %0" : "=r"(hi1));                      \
    } while (hi0 != hi1);                                                      \
    return (uint64_t)lo + ((uint64_t)hi1 << 32);                               \
  }
#elif __riscv_xlen == 64
#define RISCV_READ_COUNTER_FUNC(COUNTER)                                       \
  static inline uint64_t riscv_read_m##COUNTER(void) {                         \
    uint64_t res;                                                              \
    __asm__ volatile("rd" #COUNTER " %0" : "=r"(res));                         \
    return res;                                                                \
  }
#endif

RISCV_READ_COUNTER_FUNC(cycle)   // riscv_read_mcycle()
RISCV_READ_COUNTER_FUNC(instret) // riscv_read_minstret()

static inline void riscv_fence(void) {
  __asm__ volatile("fence" : : : "memory");
}
#else
// Stub implementations for non-RISC-V platforms
static inline uint64_t riscv_read_mcycle(void) { return 0; }
static inline uint64_t riscv_read_minstret(void) { return 0; }
static inline void riscv_fence(void) {}
#endif

void skl_test_driver_update_counters(skl_test_counters_t *counters) {
  riscv_fence();
  counters->cycles = riscv_read_mcycle() - counters->cycles;
  counters->instret = riscv_read_minstret() - counters->instret;
}

void *skl_test_driver_alloc(size_t region, size_t size) {
  (void)region;
  return aligned_alloc(SKL_TEST_DRIVER_ALIGNMENT, size);
}

void skl_test_driver_free(size_t region, void *ptr) {
  (void)region;
  free(ptr);
}

static size_t fprintf_prefix(FILE *stream, skl_test_t *t) {
  if (t == NULL) {
    return 0;
  }
  return fprintf(stream, "[%u]: ", t->id);
}

void skl_test_driver_log(skl_test_t *t, FILE *stream, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  (void)fprintf_prefix(stream, t);
  (void)vfprintf(stream, fmt, args);
  va_end(args);
}

static void log_separator(skl_test_t *t, char sep) {
  enum { SEP_LEN = 10 };
  char buf[SEP_LEN];
  memset(buf, sep, sizeof(buf));
  buf[sizeof(buf) - 1] = '\0';
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "%s\n", buf);
}

int skl_test_driver_run_suite(skl_test_suite_t *suite) {
  int failures = 0;

  // Log test suite start (and avoid lint warning)
  int log_level = SKL_TEST_LOG_LEVEL;
  if (suite->num_tests > 0 && suite->name != NULL &&
      log_level >= SKL_TEST_LOG_INFO) {
    skl_test_driver_log(NULL, stdout, "Running test suite %s with %zd tests\n",
                        suite->name, suite->num_tests);
  }

  // Main loop over all tests
  for (size_t i = 0; i < suite->num_tests; ++i) {
    skl_test_t t = {.harness = (char *)suite->tests + i * suite->test_size,
                    .suite_name = suite->name,
                    .id = i,
                    .log_level = log_level,
                    .counters = {.cycles = 0, .instret = 0},
                    .status = SKL_TEST_PASS};
    log_separator(&t, '=');

#define TRY(STEP)                                                              \
  do {                                                                         \
    if ((STEP) != NULL) {                                                      \
      SKL_TEST_REQUIRE(&t, (STEP)(&t) == 0);                                   \
      if (t.status != SKL_TEST_PASS)                                           \
        goto cleanup;                                                          \
    }                                                                          \
  } while (0)

    // Retrieve test steps: first harness struct member must be a
    // skl_test_steps_t .
    skl_test_steps_t *steps = (skl_test_steps_t *)t.harness;

    TRY(steps->init);

    // Run the test, including optional cache warmup
    TRY(steps->warmup);
    skl_test_driver_update_counters(&t.counters);
    TRY(steps->execute);
    skl_test_driver_update_counters(&t.counters);

    // Verify test results, if verification is enabled
    TRY(steps->verify);
    if (steps->verify != NULL) {
      SKL_TEST_LOG(&t, SKL_TEST_LOG_INFO, "Verification passed\n");
    }

  cleanup:
    // Always run cleanup if provided
    if (steps->cleanup != NULL) {
      int cleanup_status = steps->cleanup(&t);
      if (cleanup_status != SKL_TEST_PASS) {
        skl_test_driver_log(&t, stderr, "Cleanup function failed\n");
        t.status = SKL_TEST_FAIL;
      }
    }

    // Always run report if provided, even after failure
    if (steps->report != NULL) {
      int report_status = steps->report(&t);
      if (report_status != SKL_TEST_PASS) {
        skl_test_driver_log(&t, stderr, "Report function failed\n");
        t.status = SKL_TEST_FAIL;
      }
    }

    // Check final status and update failure count
    if (t.status != SKL_TEST_PASS) {
      SKL_TEST_LOG(&t, SKL_TEST_LOG_ERROR, "Test failed\n");
      failures++;
    }
    log_separator(&t, '=');
    SKL_TEST_LOG(&t, SKL_TEST_LOG_INFO, "\n");
  }

  if (log_level >= SKL_TEST_LOG_INFO) {
    skl_test_driver_log(NULL, stdout,
                        "Test suite %s completed %zd tests with %d failures\n",
                        suite->name, suite->num_tests, failures);
  }

  return failures;
}
