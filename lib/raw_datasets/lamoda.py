import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as sktransform
import tqdm

import torch

import config


def preprocess_image(image, output_shape=config.EMBEDDER_INPUT_SHAPE):
    return sktransform.resize(image, output_shape)


class LamodaClothes(torch.utils.data.Dataset):
    def __init__(self, images_path, output_shape=config.EMBEDDER_INPUT_SHAPE):
        super().__init__()
        
        self.images_path = images_path
        self.output_shape = output_shape

        self.files = []
        for super_dir in os.listdir(self.images_path):
            for sub_dir in os.listdir(os.path.join(self.images_path, super_dir)):
                self.files.extend(os.listdir(os.path.join(self.images_path, super_dir, sub_dir)))

        
    def __getitem__(self, idx):
        fname = self.files[idx]
        try:
            image = plt.imread(os.path.join(self.images_path, fname[0], fname[1], fname))
            image = preprocess_image(image, self.output_shape)
            return image.transpose(2, 0, 1), fname
        except Exception as e:
            print(e)
    
    def __len__(self):
        return len(self.files)

def process_image(num, train_dataset, output_path):
    try:
        image, fname = train_dataset[num]
        np.save(os.path.join(output_path, '{}').format(fname), image)

    except Exception as e:
        print(e)


def transform_dataset():
    project_dir = os.environ.get('PROJECT_DIR')

    dataset = LamodaClothes(os.path.join(project_dir, 'data', 'lamoda', 'raw_img', 'img600x866'))

    output_path = os.path.join(project_dir, 'data', 'lamoda', 'transformed')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm.tqdm(range(len(dataset))):
        process_image(i, dataset, output_path)
