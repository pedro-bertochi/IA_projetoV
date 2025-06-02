import os
from src.core import criar_modelo, combinar_imagens
import numpy as np
import shutil
import tensorflow as tf

modelo = criar_modelo()
train_gen = ParImageGenerator('../image/treino', batch_size=8)
modelo.fit(train_gen, epochs=10)
modelo.save('../model/modelo_final_media.h5')

modelo = tf.keras.models.load_model('../model/modelo_final_media.h5')
caminho = '../image/classificacao/novas'
imagens = sorted([f for f in os.listdir(caminho) if f.lower().endswith('.jpg')])

if len(imagens) % 2 != 0:
    print("[AVISO] Número ímpar de imagens! A última será ignorada.")
    imagens = imagens[:-1]


for i in range(0, len(imagens), 2):
    p1 = os.path.join(caminho, imagens[i])
    p2 = os.path.join(caminho, imagens[i+1])
    entrada = combinar_imagens(p1, p2)
    entrada = np.expand_dims(entrada / 255.0, axis=0)
    pred = modelo.predict(entrada)[0][0]
    destino = 'boa' if pred >= 0.8 else 'ruim'
    os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)
    shutil.move(p1, f'../image/classificacao/{destino}/{os.path.basename(p1)}')
    shutil.move(p2, f'../image/classificacao/{destino}/{os.path.basename(p2)}')
