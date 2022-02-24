import json
import os
import tqdm

from lib.models import face_embedder
from lib.models import fashion_item_embedder


def get_embeddings_dataset():
    transformed_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'segmentation')
    orig_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'img')

    face_processor = face_embedder.FaceEmbedder(deepface_model='VGG-Face')
    fashion_item_processor = fashion_item_embedder.FashionItemEmbedder()

    dataset = []

    for fname in tqdm.tqdm(os.listdir(orig_path)):
        try:
            face_embedding = face_processor.apply(os.path.join(orig_path, fname))
            fashion_item_embedding = fashion_item_processor.apply(
                os.path.join(transformed_path, '{}.npy'.format(fname)),
                preprocessed=True,
            )

            dataset.append({
                'fname': fname,
                'face': face_embedding,
                'fashion_item': fashion_item_embedding,
            })
        except Exception as e:
            print(e)

    return dataset


def main():
    print('Building dataset')

    dataset = get_embeddings_dataset()

    print('Dataset length: {}'.format(len(dataset)))

    print('Writing file')

    with open(os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda', 'embeddings_no_faces_vgg.json'), 'w') as f:
        f.write(json.dumps(dataset, indent=4))

    print('Done!')


if __name__ == '__main__':
    main()
