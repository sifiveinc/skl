# SKL Test and Benchmark Collection

## Goals and Coverage

SKL provides a set of tests and benchmarks for each family of kernels that can be used to test their operation and performance.
These can be located within the `test/` directory.
Each kernel family (defined by a scalar reference routine) has a corresponding test program that can exercise all of its variants, as supported by a given compilation target.

The tests are intended to demonstrate proper usage of each kernel and to provide a minimal level of validation for CI purposes.
They are not intended to be exhaustive, and are not suitable for use as a standalone validation suite.
The only requirement to run the tests, aside from a compatible toolchain, is RISC-V QEMU with appropriate ISA support.

The benchmarks are intended to showcase kernel performance, and illustrate optimal parameter settings for chosen platforms.
Use of the benchmarks requires either access to RISC-V silicon or a cycle-accurate simulator.

## Test Driver Framework

The test and benchmark collection is structured to make writing tests as simple as possible.
It provides an abstraction layer for common test-related functionality, such as allocating and initializing test data buffers, and measuring runtime.
This abstraction is implemented in the form of a test driver framework, consisting of three parts:
- **Harness**: A test harness is a C file that defines the test configuration and callbacks for a given kernel family.
  It is responsible for setting up the test data, executing the kernel, and verifying the results based on the configured parameters.
  The harness is the only part of the test that is specific to a given kernel family.
- **Test Suite**: A test suite is a list of test cases that share a common harness.
  Each test case is defined by a set of parameters, and is executed by the harness.
  The test suite is responsible for defining the test cases, and for executing the harness for each test case.
- **Driver**: The test driver is a C program that executes a test suite and interfaces with the system environment.
  It provides functions for I/O, memory allocation, and performance measurement, and calls the harness callbacks to execute the tests.

### Test Harness

A harness encapsulates all of the test logic for a given kernel family, and defines a _test configuration_ structure that is used to parametrize individual test cases.
For example, the `gemm_f32rc_f32rc_f32rc` [harness](gemm/gemm_f32rc_f32rc_f32rc.h) defines a configuration structure that contains the matrix dimensions, strides, and scaling factors for a GEMM test case:
```c
typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to GEMM function)
  size_t m, n, k;
  float alpha;
  size_t rsa, csa;
  size_t rsb, csb;
  float beta;
  size_t rsc, csc;

  // Buffer generation settings for A, B, C
  SKL_TEST_BUFFER(float) a, b, c;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *a_wide, *b_wide;
    float *ref_c;
    double *bound;
  } ctx;
} gemm_f32rc_f32rc_f32rc_t;
```

Generally, such structures will separate configurable settings from derived parameters and test state, with the latter contained in a nested structure as shown above.
Test buffers are an exception, because the `SKL_TEST_BUFFER` structure contains both the buffer pointer and the configuration parameters for buffer generation (see [Buffer Management](#buffer-management) below).

It is required that the first member of a harness struct be of type `skl_test_steps_t`,
which contains pointers to a collection of _callback functions_.
These callback functions are:
- `init`: Initialization function called at the start of each test case.
  It is responsible for allocating and initializing the test data buffers, and for setting up any other test-specific state, as well as computing any derived parameters.
- `warmup`: Warmup function called before the main test execution.
  It is responsible for any necessary warmup iterations, such as to ensure that the instruction cache is fully loaded.
- `execute`: The main test execution function.
  It is responsible for calling the kernel function with the appropriate parameters.
- `verify`: Verification function called after the test execution.
  It is responsible for checking the results of the test against a reference implementation.
- `report`: Reporting function called after the test execution.
  It is responsible for printing any test-specific results or statistics.
- `cleanup`: Cleanup function called at the end of each test case.
  It is responsible for freeing any test-specific resources.

The above callback functions are intended to be implemented by the harness;
the typical exception to this is `execute`,
which should likely be implemented by the test suite C file
because it is the only one that is specific to a given kernel variant.
These callbacks will be called by the driver to execute the test.

Most of these functions should be reusable across different kernels in the same family, so only the `execute` function needs to be customized for each kernel variant to call the appropriate kernel function and pass it the correct parameters.

All callback functions take a `skl_test_t` pointer as their only argument, which is a generic test context structure defined in [skl-test-driver.h](skl-test-driver.h).
The `skl_test_t` structure contains a pointer to the test configuration structure defined by the harness, as well as other fields used by the driver for bookkeeping and reporting:
```c
typedef struct skl_test_t {
  void *harness;                  // Harness-specific data (set by harness in init)
  const char *suite_name;         // Name of the suite this test is part of (set by driver before init)
  unsigned int id;                // Identifier for this test within a suite (set by driver before init)
  int log_level;                  // Logging verbosity level (set by driver before init)
  skl_test_counters_t counters;   // Performance counters (updated by driver)
  skl_test_step_status_t status;  // Status of each step of the test (set by harness, checked by driver)
} skl_test_t;
```
In order to access the test configuration structure, the callback function should cast the `harness` pointer to the appropriate type.
For example, the `gemm_f32rc_f32rc_f32rc` harness uses the following pattern in its `init` function:
```c
gemm_f32rc_f32rc_f32rc_t *h = t->harness;
```

### Test Suite

A test suite is a collection of related tests that share a common harness, and may be intended for use with a specific driver or environment.

These tests are enumerated in a C file as an array of test configuration structures.
For example, the `skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f` [test suite](gemm/xsfmm/skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c) defines an array of `gemm_f32rc_f32rc_f32rc_t` structures, each of which configures a different set of matrix dimensions and strides for the GEMM test case:
```c
#define TEST                                                                   \
  GEMM_F32RC_F32RC_F32RC_DEFAULTS,                                             \
      .steps = {                                                               \
          .init = gemm_f32rc_f32rc_f32rc_init,                                 \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_f32rc_f32rc_f32rc_verify,                             \
          .report = NULL,                                                      \
          .cleanup = gemm_f32rc_f32rc_f32rc_cleanup,                           \
  }
#define BENCH                                                                  \
  GEMM_F32RC_F32RC_F32RC_DEFAULTS,                                             \
      .steps = {                                                               \
          .init = gemm_f32rc_f32rc_f32rc_init,                                 \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_f32rc_f32rc_f32rc_report,                             \
          .cleanup = gemm_f32rc_f32rc_f32rc_cleanup,                           \
  }
static void execute(skl_test_t *t);

// clang-format off
gemm_f32rc_f32rc_f32rc_t tests[] = {
    // Benchmark tests
    {BENCH, .m = 128, .n = 128, .k = 2048, .beta = 0.f},
    {BENCH, .m = 128, .n = 128, .k = 2048, .beta = 1.f},
    ...
    // Verification tests
    {TEST, .rsa = 1, .m = 16,  .n = 16, .k = 16},
    {TEST, .rsa = 1, .m = 16,  .n = 16, .k = 16, .beta = 1.f},
    ...
};
```
The list of test cases may include both correctness tests and performance benchmarks, distinguished by the callback functions they specify.
Any callback except `execute` may be provided as `NULL` to skip the execution of such step.

The test suite must also define a `skl_test_suite_t` structure that describes the test suite to the driver, and includes the array of test configurations:
```c
skl_test_suite_t suite = {.name = "skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f",
                          .num_tests = sizeof(tests) / sizeof(tests[0]),
                          .test_size = sizeof(gemm_f32rc_f32rc_f32rc_t),
                          .tests = tests};
```
The test suite C file must also call `skl_test_driver_run_suite` to execute the test suite, generally inside its own `main` function:
```c
int main() {
  return skl_test_driver_run_suite(&suite);
}
```

### Driver

The test driver is a C program that executes a test suite and interfaces with the system environment.
It provides functions for I/O, memory allocation, and performance measurement, and calls the harness callbacks to execute the tests.

The SKL repo provides its own simple driver implementation in [skl-test-driver.c](skl-test-driver.c), which is sufficient for running the tests in QEMU.
Other environments will need to provide their own driver implementation that overrides the default implementations as needed.
However, all drivers share the same interface, defined in [skl-test-driver.h](skl-test-driver.h).

Functions provided by the driver are:
- `skl_test_driver_run_suite`: Runs a test suite.
- `skl_test_driver_alloc`: Allocates memory for a test buffer.
- `skl_test_driver_free`: Frees memory for a test buffer.
- `skl_test_driver_update_counters`: Reads the current values of the performance counters and records the delta since the last update.
- `skl_test_driver_log`: Prints a log message with a given log level.
- `skl_test_driver_status`: Returns the current pass/fail status of the test.


### Buffer Management

SKL _test buffers_ refer to arrays of input data used by tests, and defined by how they are initialized as well as where they are stored in memory.
The `SKL_TEST_BUFFER` macro is used to define a test buffer in a test configuration structure, and the `SKL_TEST_BUF_CREATE` macro is used to allocate and initialize the buffer at runtime.
This pair of abstractions allows tests to express succinct data ranges of interest (min, max, random, sequential, etc.) in their configurations.
It also allows test suites to tell a driver that they want buffers to be allocated in a particular way, for example to exercise different memory regions or alignment constraints.

The `SKL_TEST_BUFFER` macro defines a structure that contains the following members:
```c
// For TYPE given by the SKL_TEST_BUFFER macro:
struct {
  TYPE *data; // Pointer to the buffer data (set by driver)
  size_t len; // Length of the buffer (set by test configuration)
  skl_test_init_mode_t mode; // enum: random, static, sequential
  TYPE min, max;
  const TYPE *static_data; // Contains actual data for static mode
  size_t static_data_len;
  size_t region; // Meaning defined by the driver
};
```
Test configurations will populate the `len`, `mode`, `min`, and `max` fields typically, though some harnesses may also provide configuration options for `static_data` and `region` as well.
The range fields allow targeted test specification, while static data serves two purposes:
- It allows for the definition of "special" input patterns that are difficult to express with ranges alone, such as all-zero or all-one vectors.
- It allows for fast initialization by memcpy, and avoids costly calls to `rand()` in slow simulation environments.

The `SKL_TEST_BUF_CREATE` macro should be called in the `init` function of the test harness to allocate and initialize each test buffer.
It injects type-dependent code to initialize the buffer according to the `mode` field of the `SKL_TEST_BUFFER` structure.
All buffers must be freed in the `cleanup` function with the `SKL_TEST_BUF_FREE` macro.
