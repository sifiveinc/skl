# SKL CI

CI for SKL uses GitHub Actions.
The workflow script can be found at `.github/workflows/ci.yml`.
CI runs the entire SKL test suite for the three possible configurations (tests only, benchmarks only, both tests and benchmarks), a lint check using clang-tidy, and a code format check using clang-format.
Pull requests must pass all CI checks before they can be merged into `main`.
CI is triggered automatically for new pull requests and whenever changes are pushed to an existing pull request.
Contributors can visit the Actions tab of the repository to view more detailed information for a CI run and see which checks passed or failed.
