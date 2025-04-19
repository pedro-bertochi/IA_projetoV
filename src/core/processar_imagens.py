import cv2
import numpy as np
import os


def combinar_imagens(path1, path2, size=(224, 224)):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise ValueError(f"Erro ao carregar imagens: {path1}, {path2}")
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    return np.concatenate([img1, img2], axis=1)
