# Deploy to https://pypi.org/project/loudml-py/
# Requires a ~/.pypirc file in the developer machine with proper credentials

#!/usr/bin/env bash

LOUDML_PYTHON_VERSION=$(git describe --tags --match 'v*.*.*' \
  | sed -e 's/^v//' -e 's/-/./g')
docker build -t loudml-py-publish .
docker run -it -e LOUDML_PYTHON_VERSION=$LOUDML_PYTHON_VERSION loudml-py-publish
