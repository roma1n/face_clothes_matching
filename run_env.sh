#!/bin/bash

# project environment variables
export PROJECT_DIR=$(pwd)

export PATH="$PROJECT_DIR:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# run environment
pipenv install
pipenv shell
