#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import shutil
import tensorflow as tf
from core import criar_modelo, combinar_imagens
from utils import save_accuracy
import numpy as np


# In[16]:


# modelo = tf.keras.models.load_model('../model/modelo_solda_resnet50.h5')
modelo = tf.keras.models.load_model('../model/modelo_final_media_funcional.h5')
caminho = '../image/classificacao/novas'
imagens = sorted([f for f in os.listdir(caminho) if f.lower().endswith('.jpg')])

os.makedirs('../results', exist_ok=True)


# In[18]:


if len(imagens) % 2 != 0:
    print("[AVISO] Número ímpar de imagens! A última será ignorada.")
    imagens = imagens[:-1]


# In[19]:


for i in range(0, len(imagens), 2):
    p1 = os.path.join(caminho, imagens[i])
    p2 = os.path.join(caminho, imagens[i+1])

    # Preparar a entrada para o modelo
    entrada = combinar_imagens(p1, p2)
    entrada = np.expand_dims(entrada / 255.0, axis=0)

    # Predição do modelo
    pred = modelo.predict(entrada, verbose=0)[0][0]

    destino = 'boa' if pred >= 0.8 else 'ruim'
    print(f"Predição: {pred:.2f} - Imagens: {imagens[i]} e {imagens[i+1]} - Destino: {destino}")


    save_accuracy("../results/accuracy.txt", pred, imagens[i], imagens[i+1])

    # Criar diretório de destino se não existir
    os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)

    # Mover as imagens para o diretório de destino
    shutil.move(p1, os.path.join(f'../image/classificacao/{destino}', os.path.basename(p1)))
    shutil.move(p2, os.path.join(f'../image/classificacao/{destino}', os.path.basename(p2)))

