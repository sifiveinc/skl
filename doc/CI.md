# SKL CI

CI for SKL uses GitHub Actions.
The workflow script can be found at `.github/workflows/ci.yml`.
CI performs a license header check, a lint check using clang-tidy, and a code format check using clang-format and runs the entire SKL test suite (tests and benchmarks).
Note that benchmark run times in CI do not reflect the actual performance of any kernel in SKL because the test suite is run in QEMU.
Pull requests must pass all CI checks before they can be merged into `main`.
CI runs can only be launched by official maintainers.
Contributors can visit the Actions tab of the repository to view more detailed information for a CI run and see which checks passed or failed.
