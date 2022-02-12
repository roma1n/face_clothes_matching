import numpy as np
from skimage import filters


def augmentate_image(img):
    # Random vertical flip
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1]
        
    # Gaussian smoothing
    img[:3, :, :] = filters.gaussian(img[:3, :, :], np.random.rand() * 3)

    # Noise with different scale in first 3 channels
    noise = ((np.random.rand(3, *img.shape[1:]) - 0.5) * np.random.rand(3, 1, 1))
    img[:3, :, :] += noise
    img = np.clip(img, 0., 1.)

    return img
