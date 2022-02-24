import json
import logging
import os
import sys

from lib.models import (
    face_embedder,
    face_fashion_item_matching,
    fashion_item_embedder,
)
from lib.utils import pipeline


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    # faces_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation', 'faces')
    # fashoin_items_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation', 'segmentation')
    validation_dataset_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation')
    result_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'validation', 'result_vgg.json')

    logging.info('Loading face processor')
    face_processor = face_embedder.FaceEmbedder(deepface_model='VGG-Face')
    logging.info('Loading fashion item processor')
    fashion_item_processor = fashion_item_embedder.FashionItemEmbedder()
    logging.info('Loadiong face - fashion item matching model')
    matching_model = face_fashion_item_matching.FaceFashionItemMatching()

    logging.info('Validating pipeline')
    result = pipeline.validate_with_lamoda(
        validation_dataset_path=validation_dataset_path,
        face_processor=face_processor,
        fashion_item_processor=fashion_item_processor,
        matching_model=matching_model,
    )

    logging.info('Saving result')

    with open(result_path, 'w') as f:
        f.write(json.dumps(result, indent=4))

    logging.info('Done!')


if __name__ == '__main__':
    main()
