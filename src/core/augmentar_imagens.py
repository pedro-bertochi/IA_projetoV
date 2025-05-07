import cv2
import numpy as np
import random

def rotacionar_imagem(img, angulo):
    altura, largura = img.shape[:2]
    centro = (largura // 2, altura // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(img, matriz, (largura, altura), flags=cv2.INTER_LINEAR)

def aplicar_flip(img):
    flip_code = random.choice([-1, 0, 1])  # -1: ambos, 0: vertical, 1: horizontal
    return cv2.flip(img, flip_code)

def ajustar_brilho(img, fator=0.5):
    fator = random.uniform(0.5, 1.5)  # brilho aleatório entre 50% e 150%
    return cv2.convertScaleAbs(img, alpha=fator, beta=0)

def adicionar_ruido(img, intensidade=10):
    ruido = np.random.randint(-intensidade, intensidade, img.shape, dtype='int16')
    img_ruidosa = img.astype('int16') + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype('uint8')
    return img_ruidosa

def combinar_com_augmentacao(path1, path2, size=(224, 224), angulos=[0, 90, 180, 270]):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise ValueError(f"Erro ao carregar imagens: {path1}, {path2}")

    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    # Aplicar augmentações aleatórias
    # img1 = rotacionar_imagem(img1, random.choice(angulos))
    # img2 = rotacionar_imagem(img2, random.choice(angulos))

    if random.random() < 0.5:
        img1 = aplicar_flip(img1)
    if random.random() < 0.5:
        img2 = aplicar_flip(img2)

    if random.random() < 0.5:
        img1 = ajustar_brilho(img1)
    if random.random() < 0.5:
        img2 = ajustar_brilho(img2)

    if random.random() < 0.2:
        img1 = adicionar_ruido(img1)
    if random.random() < 0.2:
        img2 = adicionar_ruido(img2)

    return np.concatenate([img1, img2], axis=1)
