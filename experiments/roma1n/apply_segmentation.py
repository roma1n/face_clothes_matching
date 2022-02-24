import json
import logging
import os
import sys

from lib.models import fashion_item_segmentation
from lib.utils import pipeline


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    # input_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'img')
    # segmentation_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'segmentation')

    input_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation', 'fashion_items')
    segmentation_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation', 'segmentation')

    segmentation_model = fashion_item_segmentation.FashionItemSegmentation()
    pipeline.apply_segmentation(
        segmentation_model=segmentation_model,
        input_path=input_path,
        output_path=segmentation_path,
        batch_size=100,
    )


if __name__ == '__main__':
    main()
