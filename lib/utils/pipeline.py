import collections
import logging
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn import mixture
import skimage.transform as sktransform

import config


def validate_with_lamoda(
    validation_dataset_path,
    face_processor,
    fashion_item_processor,
    matching_model,
):
    logging.info('Loading validation meta')
    validation_meta = pd.read_csv(os.path.join(validation_dataset_path, 'meta.csv'))

    logging.info('Calculating face embeddings')
    uniq_face_imgs = np.unique(validation_meta['face_img'])
    logging.info('{} uniq faces found in meta'.format(len(uniq_face_imgs)))
    face_embeddings = {}
    for face_img in tqdm.tqdm(uniq_face_imgs):
        face_embeddings[face_img] = face_processor.apply(
            os.path.join(validation_dataset_path, 'faces', face_img),
        )

    logging.info('Calculating fashion item embeddings')
    fashion_item_embeddings = {}
    for lamoda_id in tqdm.tqdm(validation_meta['id']):
        fashion_item_embeddings[lamoda_id] = fashion_item_processor.apply(
            os.path.join(validation_dataset_path, 'segmentation', '{}.jpg.npy'.format(lamoda_id)),
            preprocessed=True,
        )

    logging.info('Calculating matching score for each face and each clothes')
    result = []
    for lamoda_id in tqdm.tqdm(fashion_item_embeddings):
        for face_img in face_embeddings:
            result.append({
                'lamoda_id': lamoda_id,
                'face_img': face_img,
                'score': matching_model.apply({
                    'x': face_embeddings[face_img],
                    'y': fashion_item_embeddings[lamoda_id],
                }),
            })

    return result


def apply_segmentation(
    segmentation_model,
    input_path,
    output_path,
    batch_size=100,
):
    def postprocess(mask, image):
        mask = mask.reshape(10, 224**2).T
        gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(
            mask[::41],  # 41 for acceleration
        )
        labels = gm.predict(mask)
        cl = labels.reshape(224, 224)
        return image * cl[None, :, :] + np.ones((3, 224, 224)) * (1 - cl[None, :, :])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    files = list(map(lambda x: os.path.join(input_path, x), os.listdir(input_path)))
    outputs = list(map(lambda x: os.path.join(output_path, '{}.npy'.format(x)), os.listdir(input_path)))

    for batch_begin in tqdm.tqdm(range(0, len(files), batch_size)):
        applied, input = segmentation_model.apply_batched(files[batch_begin: batch_begin + batch_size], return_input=True)

        for t, i, fname in zip(applied, input, outputs[batch_begin: batch_begin + batch_size]):
            with open(fname, 'wb') as f:
                np.save(f, postprocess(t, i))


def process_dir_with_model(
    model,
    batch_size,
    input_path,
    output_path=None,
    preprocessed=False,
    postprocess_result=None,
):
    '''
    If output path is None the result is return
    If output path is not None the result is saved to specified dir (each element saved to separated file)
    '''
    postprocess_result = postprocess_result or (lambda x: x)

    if output_path is None:
        result = []


        filenames = os.listdir(input_path)
        absolute_filenames = list(map(lambda x: os.path.join(input_path, x), filenames))

        for batch_begin in tqdm.tqdm(range(0, len(filenames), batch_size)):
            result.extend([
                {
                    'filename': filename,
                    'result': postprocess_result(result),
                } for filename, result in zip(
                    filenames[batch_begin: batch_begin + batch_size],
                    model.apply_batched(
                        absolute_filenames[batch_begin: batch_begin + batch_size],
                        preprocessed=preprocessed,
                    ),
                )
            ])

        return result
    else:
        # TODO: implement
        assert False, 'Not implemented'


def find_best_matches(
    matching_model,
    face_processor,
    face_img_path,
    fashion_items,
    top_k=5,
    batch_size=100,
):
    face_embedding = face_processor.apply(face_img_path)

    result = dict()

    for batch_begin in tqdm.tqdm(range(0, len(fashion_items), batch_size), file=sys.stderr):
        batch = fashion_items[batch_begin: batch_begin + batch_size]
        batch_res = matching_model.apply_batched({
            'x': [face_embedding for _ in range(len(batch))],
            'y': [fashion_item['result'] for fashion_item in batch],
        })

        for fashion_item, fashion_item_res in zip(batch, batch_res):
            result[fashion_item['filename'].split('_')[0]] = fashion_item_res

    result = sorted(result, key=result.get, reverse=True)
    return result[:top_k]
