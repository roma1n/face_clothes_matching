#!/bin/bash

COMPETITION_ID=$1

export KAGGLE_USERNAME=$(cat $PROJECT_DIR/tokens/kaggle.token | sed "1q;d")
export KAGGLE_KEY=$(cat $PROJECT_DIR/tokens/kaggle.token | sed "2q;d")

echo "Downloading dataset for ${COMPETITION_ID} from kaggle"

kaggle competitions download -c $COMPETITION_ID -p "${PROJECT_DIR}/data/${COMPETITION_ID}" --force

unzip "${PROJECT_DIR}/data/${COMPETITION_ID}/${COMPETITION_ID}.zip" -d "${PROJECT_DIR}/data/${COMPETITION_ID}"
