#!/bin/bash

PREV_TASK=$TASK
export TASK=$1

echo "Executing task: ${TASK}"

cd ${PROJECT_DIR}

python bin/main.py ${TASK}

echo "Finished task: ${TASK}"

export TASK=${PREV_TASK}
