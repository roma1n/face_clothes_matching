#!/bin/bash

${PROJECT_DIR}/scripts/download_dataset_from_kaggle.sh imaterialist-fashion-2019-FGVC6

run_task.sh transform_segmentation_dataset
