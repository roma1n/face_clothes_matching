import logging
import numpy as np
import os
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


def process_with_model(
    model,
    input,
    output_path=None,
):
    '''
    If output path is None the result is return
    If output path is not None the result is saved to specified dir # TODO: implement
    '''
    if output_path is None:
        pass
    else:
        assert False, 'Not implemented'
