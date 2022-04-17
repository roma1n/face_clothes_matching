import json
import os
import sys

import config
from lib.models import (
    face_embedder,
    face_fashion_item_matching,
)
from lib.parsers import lamoda_parser
from lib.utils import pipeline


class FashionItemRecomedator:
    def __init__(self):
        self.face_processor = face_embedder.FaceEmbedder(deepface_model='VGG-Face')

        self.matching_model = face_fashion_item_matching.FaceFashionItemMatching()

        with open(config.LAMODA_FASHION_ITEM_EMBDDEINGS_PATH, 'r') as f:
            self.fashion_items = json.loads(f.read())

        self.id_to_img_path = {
            v.split('_')[0]: os.path.join(config.LAMODA_IMG_DIR, v) for v in os.listdir(config.LAMODA_IMG_DIR)
        }


    def process_face_img(self, face_img_path):
        ids = pipeline.find_best_matches(
            matching_model=self.matching_model,
            face_processor=self.face_processor,
            face_img_path=face_img_path,
            fashion_items=self.fashion_items,
            top_k=5,
        )

        urls = list(map(lambda x: lamoda_parser.suggest_url(x.lower()), ids))

        img_paths = list(map(lambda x: self.id_to_img_path[x], ids))

        return {
            'ids': ids,
            'urls': urls,
            'img_paths': img_paths,
        }
