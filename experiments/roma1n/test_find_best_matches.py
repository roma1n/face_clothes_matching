import json
import os

import config
from lib.models import (
    face_embedder,
    face_fashion_item_matching,
)
from lib.utils import pipeline

def main():
    print('Loading face processor')
    face_processor = face_embedder.FaceEmbedder(deepface_model='VGG-Face')

    print('Loadiong face - fashion item matching model')
    matching_model = face_fashion_item_matching.FaceFashionItemMatching()

    print('Loading fashion items')
    with open(config.LAMODA_FASHION_ITEM_EMBDDEINGS_PATH, 'r') as f:
        fashion_items = json.loads(f.read())

    print('Finding best matches')
    print(pipeline.find_best_matches(
        matching_model=matching_model,
        face_processor=face_processor,
        face_img_path=os.path.join(
            os.environ['PROJECT_DIR'], 'notebooks', 'roma1n', 'data', 'preprocessing', 'original2.jpg',
        ),
        fashion_items=fashion_items,
        top_k=5,
    ))

if __name__ == '__main__':
    main()
