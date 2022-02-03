import numpy as np
from skimage import filters


def augmentate_image(img):
    # Случайный разворот относительно вертикальной оси
    if np.random.rand() < 0.5:
        img = img[:,::-1]
        
    # Гауссовское размытие со случайным скейлом. Применяется только к первым трем каналам
    img[:3, :, :] = filters.gaussian(img[:3, :, :], np.random.rand() * 3)

    # Шум с разным скейлом в разных каналах. Применяется только к первым трем каналам
    noise = ((np.random.rand(3, *img.shape[1:]) - 0.5) * np.random.rand(3, 1, 1))
    img[:3, :, :] += noise
    img = np.clip(img, 0., 1.)

    return img
