# Best Practices for Managing Keras Projects on GitHub

This document describes the best practices for managing the projects under
"keras-team" on GitHub which use GitHub as the source of truth, including
[keras-tuner](https://github.com/keras-team/keras-tuner),
[autokeras](https://github.com/keras-team/autokeras),
[keras-cv](https://github.com/keras-team/keras-cv),
[keras-nlp](https://github.com/keras-team/keras-nlp),
and maybe more in the future. It covers linting, formating, testing, continuous
integration, issues and pull requests tagging, and so on.

The goal of this document is to:
* Improve the overall quality of the projects. The fact that projects all
  follow the same standard for dev process, which may evolve through time, will
  ensure the quality from all aspects.
* Unify the external contributing experience. The external open-source
  contributors may contribute to multiple Keras projects by submitting issues
  or pull requests. They don't need to learn from different contributing
  guides.
* Save time for the project leads. They save time by copying and pasting the
  same setup and by avoiding the listed caveats.

## Testing

### Testing framework

We use [pytest](https://docs.pytest.org/en/6.2.x/) for writing tests for the
projects, which is the most widely used testing framework for Python in the OSS
world. The configuration of pytest is
[here](https://github.com/keras-team/keras-tuner/blob/1.1.0/setup.cfg#L4-L16).

### File locations for the tests

Unit tests should be contained in sibling files, relative to the class or
utility files they are testing. The name of a test file should follow the
pattern of `*_test.py`. For example, the tests for
`/keras_tuner/engine/hyperparameters.py` are in
`/keras_tuner/engine/hyperparameters_tests.py`.

Integration tests may be contained in their own `/keras_tuner/integration_tests`
directory, as they may require extra files such as data.

While our unit test placement is not suggested in the
[good practices of pytest](https://docs.pytest.org/en/6.2.x/goodpractices.html)
doc, we recommend this approach to improve the discoverability of the unit
tests for new contributors. This discoverability doubles up as a method of
documentation; when users want to see what `util.utility_function()` does, they
can simply open the conveniently located sibling file, `util_test.py`.

### Test Coverage

We use [CodeCov](https://about.codecov.io/) to track the test coverage.You may
also refer to
[these settings](https://github.com/keras-team/keras-tuner/blob/1.1.0/setup.cfg#L24-L28)
in `setup.cfg`. We will see more about it in the continuous integration section.

Pytest CodeCov supports a wildcard exclude field, which should be set to
include `*_test.py`, as to ensure that tests are not included in the code
coverage count.

### Useful code snippets
Fix the random seed for all tests:
[Link1](https://github.com/keras-team/keras-tuner/blob/1.1.0/tests/conftest.py#L8-L17),
[Link2](https://github.com/keras-team/keras-tuner/blob/master/tests/unit_tests/randomness_test.py),
[Link3](https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed).

Create a temporary path for testing: [Link](https://docs.pytest.org/en/6.2.x/tmpdir.html).

## Code styles

### Importing Keras modules

For projects based on Keras and TensorFlow, top-level imports are encouraged, like
shows in the following example.

```py
import tensorflow as tf
from tensorflow import keras
```

Exceptions may be acceptable when the module appeared too many times in the code,
like `keras.layers`.

### Linting and formatting

We use
[black](https://black.readthedocs.io/en/stable/),
[isort](https://pycqa.github.io/isort/), 
[flake8](https://flake8.pycqa.org/en/latest/)
to lint and format the code. black is to generally format the code. isort is to
sort the imports. flake8 is for some additional checks that black doesn't do,
like the long lines with a single string. You can see the relevant sections of
[setup.cfg](https://github.com/keras-team/keras-tuner/blob/1.1.0/setup.cfg) for
the detailed configuration of these tools.

The user does not need to know how to use these tools to lint or format the
code. We provide them with two shell scripts:
[`/shell/lint.sh`](https://github.com/keras-team/keras-tuner/blob/master/shell/lint.sh)
and
[`/shell/format.sh`](https://github.com/keras-team/keras-tuner/blob/master/shell/format.sh).
In these scripts, we also check and add the Apache 2.0 License head to every
file.

## Releasing

### Release setups

The version number of the package is stored only in `/package_name/__init__.py`
with a single line of `__version__ = 'master'` on the master branch.
[example](https://github.com/keras-team/keras-tuner/blob/1e13aabe5b6659340a8ee81328805479a57b2105/keras_tuner/__init__.py#L35)

We also need the `setup.py` file for the PyPI release.
[example](https://github.com/keras-team/keras-tuner/blob/1e13aabe5b6659340a8ee81328805479a57b2105/setup.py)

For the `setup.py` file to grab the current version number from
`/package_name/__init__.py`, we need additional lines in `setup.cfg`.
[example](https://github.com/keras-team/keras-tuner/blob/1.1.0/setup.cfg#L1-L2)

### Draft a new release

For releasing a new version of the package, please following these steps:
* Create a new branch from the master branch.
* Modify the `__version__` value in the new branch.
* Create a new release on GitHub.
  [Official tutorial](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

Note that the continuous integration will upload it to PyPI automatically.

### Excluding Sibling Test

Unit tests are hosted in sibling files relative to the files containing the
code they are testing. `SetupTools.find_packages()` supports an
[exclude field](https://github.com/pypa/setuptools/blob/f838bc6a170046c9fdfc2251e5466040a669ca12/setuptools/__init__.py#L52).
This field should contain `*_test.py` to ensure that tests are not packaged
with the release.

## Continuous integration

We use [GitHub Actions](https://github.com/features/actions) for continuous
integrations. It automates running tests, checking the code styles, uploading
test coverages to CodeCov, and uploading new releases to PyPI.

You can refer to
[this file](https://github.com/keras-team/keras-tuner/blob/master/.github/workflows/actions.yml)
for how to set it up. We use a single YAML file for all the GitHub Actions to
avoid installing the dependencies multiple times.

To use this setup, you also need to upload your CodeCov and PyPI credentials to
the project. Here is the
[official tutorial](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository).

Make sure you follow the naming of the following secrets for the GitHub Actions YAML file to work.
Name the CodeCov token as `CODECOV_TOKEN`.
Name the PyPI username and password as `PYPI_USERNAME` and `PYPI_PASSWORD`.

We should also test against tf-nightly every day to discover bugs and
incompatible issues early and well before the stable release of TensorFlow.
The CI setup for it is
[here](https://github.com/keras-team/keras-tuner/blob/master/.github/workflows/nightly.yml).

## Contributing experience

We will have a common CONTRIBUTING.md in `keras-team/governance` to be
distributed to the other repos. This
[GitHub Action](https://github.com/marketplace/actions/file-sync) may be a good
way to sync a centralized contributing guide to different repos.
We should also have
[this directory](https://github.com/keras-team/keras-tuner/tree/master/.devcontainer)
to support GitHub Codespaces, which is a trend on GitHub. It provides a
web-based IDE to save the contributors from setting up their own dev
environment, which would attract more contributors.

## Issues and pull requests

We will have the same issue and pull request
[templates](https://github.com/keras-team/keras/tree/master/.github/ISSUE_TEMPLATE)
across projects in `keras-team`. They will also be stored in
`keras-team/governance` and be distributed to the other repos.

Also need to confirm if there is a way to unify the taggings between the repos.
