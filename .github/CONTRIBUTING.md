# Contributing to yolort

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to `yolort`, there are many ways to help out. Your contributions may fall into the following categories:

- It helps the project if you could

  - Report issues you're facing
  - Give a :+1: on issues that others reported and that are relevant to you

- Answering queries on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug

  - please pick one from the [list of open issues labelled as "help wanted"](https://github.com/zhiqwang/yolov5-rt-stack/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
  - comment on the issue that you want to work on this issue
  - send a PR with your fix, see below.

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## Development installation

### Install PyTorch and TorchVision

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# or with pip (see https://pytorch.org/get-started/locally/)
# pip install numpy
# pip install torch torchvision
```

### Install yolort

```bash
git clone https://github.com/zhiqwang/yolort.git
cd yolov5-rt-stack
pip install -e .
```

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `main`.
1. If you have modified the code (new feature or bug-fix), please add unit tests.
1. If you have changed APIs, update the documentation. Make sure the documentation builds.
1. Ensure the test suite passes.
1. Make sure your code passes the formatting checks.

### Code formatting

Contributions should be compatible with Python 3.X versions and be compliant with PEP8. We have already installed the [pre-commit service](https://github.com/apps/pre-commit-ci) to auto fix the pull requests. If you want to check the codebase locally, please either run

```bash
pre-commit run --all-files
```

or run

```bash
pre-commit install
```

once to perform these checks automatically before every `git commit`. If `pre-commit` is not available you can install it with

```bash
pip install pre-commit
```

### Unit tests

If you have modified the code by adding a new feature or a bug-fix, please add unit tests for that. To run a specific test:

```bash
pytest test/<test-module.py> -vvv -k <test_myfunc>
# e.g. pytest test/test_models.py -vvv -k test_load_from_yolov5
```

If you would like to run all tests:

```bash
pytest test -vvv
```

### Documentation

TBD

### Pull Request Recommendations

If all previous checks (flake8, mypy, unit tests) are passing, please send a PR. Submitted PR will pass other tests on different operation systems, python versions and hardwares. To allow your work to be integrated as seamlessly as possible, we advise you to:

- :white_check_mark: Verify your PR is **up-to-date with upstream/main**. You could update your PR to upstream/main by running the following code, don't forget replacing 'feature' with the name of your local branch:

  ```bash
  git remote add upstream https://github.com/zhiqwang/yolov5-rt-stack.git
  git fetch upstream
  git rebase upstream/main
  git checkout -b feature  # <--- REPLACE 'feature' WITH YOUR LOCAL BRANCH NAME
  # ADD YOUR PROPOSED CHANGES HERE
  git add .
  git commit -m "YOUR REVISION MESSAGE"
  git push origin feature
  ```

- :white_check_mark: Verify all Continuous Integration (CI) **checks are passing**.

- :white_check_mark: Reduce changes to the absolute **minimum** required for your bug fix or feature addition. _"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_ — Bruce Lee

For more details about pull requests workflow, please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License

By contributing to `yolort`, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
