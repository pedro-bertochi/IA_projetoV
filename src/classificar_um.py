#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import tensorflow as tf
import numpy as np
from core import combinar_imagens  # Se não usar, pode remover
from utils import save_accuracy
from PIL import Image

# Carrega o modelo (ajuste o caminho do modelo que seu amigo te passou)
modelo = tf.keras.models.load_model('../model/model_final_best.h5')

# Caminho das novas imagens
caminho = '../image/classificacao/novas'
imagens = sorted([f for f in os.listdir(caminho) if f.lower().endswith('.jpg')])

# Diretório para salvar os resultados
os.makedirs('../results', exist_ok=True)

for nome_img in imagens:
    caminho_img = os.path.join(caminho, nome_img)

    # Carrega e processa a imagem (ajuste tamanho conforme seu modelo espera)
    imagem = Image.open(caminho_img).convert('RGB')
    imagem = imagem.resize((224, 224))  # ajuste para o tamanho do seu modelo
    entrada = np.array(imagem) / 255.0
    entrada = np.expand_dims(entrada, axis=0)

    # Faz a predição
    pred = modelo.predict(entrada, verbose=0)[0][0]
    destino = 'boa' if pred >= 0.8 else 'ruim'

    print(f"Predição: {pred:.2f} - Imagem: {nome_img} - Destino: {destino}")

    save_accuracy("../results/accuracy.txt", pred, nome_img)

    # Cria o diretório de destino, se não existir
    os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)

    # Move a imagem para a pasta correta
    shutil.move(caminho_img, os.path.join(f'../image/classificacao/{destino}', nome_img))