import functools
import multiprocessing
import os

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform as sktransform
import tqdm

import torch

import config


def preprocess_image(image, output_shape=config.EMBEDDER_INPUT_SHAPE):
    return sktransform.resize(image, output_shape)


class SegmenationDataset(torch.utils.data.Dataset):
    '''
    Dataset handler for imaterialist-fashion-2019-FGVC6
    https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6
    '''
    def __init__(
        self,
        path, 
        num_classes=46,
        threshold=0.01,
        output_shape=config.EMBEDDER_INPUT_SHAPE,
    ):
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.threshold = threshold
        self.output_shape = output_shape

        self.images_path = os.path.join(self.path, 'sample')
        self.files = sorted(os.listdir(self.images_path))
        self.labels = pd.read_csv(os.path.join(self.path, 'minitrain.csv'))

    def __getitem__(self, idx):
        fname = self.files[idx]

        try:
            image = plt.imread(os.path.join(self.images_path, fname))
            input_shape = image.shape
            image = preprocess_image(image, self.output_shape)
            mask = np.zeros((self.num_classes, *self.output_shape))

            mask_info = self.labels.loc[
                self.labels.ImageId == fname
            ].reset_index()
            cids = []
            for num, row in mask_info.iterrows():
                try:
                    cid = int(row['ClassId'])
                    assert 0 <= cid < self.num_classes
                except:
                    continue

                mask_encoded = row['EncodedPixels']
                mask_encoded = np.array(mask_encoded.split(' ')).astype(int)
                mask_encoded = mask_encoded.reshape(mask_encoded.shape[0] // 2, 2)

                mask_decoded = np.zeros((input_shape[1], input_shape[0])).flatten()
                for pos, l in mask_encoded:
                    mask_decoded[pos:pos + l] = 1
                mask_decoded = mask_decoded.reshape(input_shape[1], input_shape[0])
                if mask_decoded.mean() > self.threshold:
                  mask_decoded = sktransform.resize(mask_decoded, self.output_shape)
                  mask[cid] = mask_decoded.T
                  cids.append(cid)

            return image.transpose(2, 0, 1), mask[cids], cids
        except Exception as e:
            return None, None, []
    
    def __len__(self):
        return len(self.files)


def process_image(num, train_dataset, output_path, desc_output_path):
    try:
        image, mask, cids = train_dataset[num]
        if len(cids) > 0:
            arr = np.concatenate([image, mask])

            np.save(os.path.join(output_path, '{}').format(num), arr)

            with open(
                os.path.join(desc_output_path, '{}.json'.format(num)), 'w') as f:
                f.write(json.dumps(
                    {
                        'num': num,
                        'cids': cids,
                    }
                ))
    except Exception as e:
        pass
                

def transform_dataset():
    project_dir = os.environ.get('PROJECT_DIR')
    dataset_path = os.path.join(project_dir, 'data', 'imaterialist-fashion-2019-FGVC6')
    train_dataset = SegmenationDataset(dataset_path)

    output_path = os.path.join(dataset_path, 'transformed')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    desc_output_path = os.path.join(dataset_path, 'transformed_desc')
    if not os.path.exists(desc_output_path):
        os.mkdir(desc_output_path)
        
    total = len(train_dataset)

    with multiprocessing.Pool(7) as pool:
        for _ in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    process_image,
                    train_dataset=train_dataset,
                    output_path=output_path,
                    desc_output_path=desc_output_path,
                ),
                list(range(total))
            ),
            total=total
        ):
            pass
