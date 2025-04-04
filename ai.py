#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import tensorflow as tf
import shutil


# In[ ]:


# Carregar o modelo treinado
model = tf.keras.models.load_model(r"model/modelo_solda.h5")

# Diretório de novas imagens
input_dir = r"image/classificacao/novas"
train_dir = r"image/classificacao"  # Onde a IA vai mover as imagens classificadas
classes = ["boa", "ruim", "incertas"]


# In[ ]:


# Função para garantir que as pastas existam
def verificar_pastas():
    for classe in classes:
        pasta = os.path.join(train_dir, classe)
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f"Pasta {classe} criada em {pasta}")
    incerta_pasta = os.path.join(train_dir, "incertas")
    if not os.path.exists(incerta_pasta):
        os.makedirs(incerta_pasta)
        print(f"Pasta 'incertas' criada em {incerta_pasta}")

verificar_pastas()


# In[ ]:


# Listar imagens e organizar em pares (assumindo que cada par tem um nome base comum)
imagens = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])


# In[ ]:


pares = {}
for img in imagens:
    nome_base = img.rsplit('_', 1)[0]  # Remove a parte '_frente' ou '_tras'
    if nome_base not in pares:
        pares[nome_base] = []
    pares[nome_base].append(img)


# In[ ]:


# Função para processar duas imagens e concatená-las
def processar_imagens(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))

    img_concatenada = np.concatenate((img1, img2), axis=1)  # Junta as imagens lado a lado
    img_concatenada = img_concatenada / 255.0  # Normalizar
    img_concatenada = np.expand_dims(img_concatenada, axis=0)  # Expandir dimensão para o modelo

    return img_concatenada


# In[ ]:


# Classificar os pares de imagens
for nome_base, imagens in pares.items():
    if len(imagens) == 2:
        img_path1 = os.path.join(input_dir, imagens[0])
        img_path2 = os.path.join(input_dir, imagens[1])

        img_processada = processar_imagens(img_path1, img_path2)
        predicao = model.predict(img_processada)[0]
        classe_predita = np.argmax(predicao)
        classe_nome = classes[classe_predita]
        certeza = predicao[classe_predita] * 100  # Converte para porcentagem

        print(f"Par {imagens[0]} e {imagens[1]} classificado como: {classe_nome} ({certeza:.2f}% de certeza)")

        resposta = input(f"A classificação está correta? (s/n): ").strip().lower()

        if resposta == "n":
            nova_classe = input("Digite a classe correta (boa/ruim/incerta): ").strip().lower()
            if nova_classe in classes:
                destino = os.path.join(train_dir, nova_classe)
                shutil.move(img_path1, os.path.join(destino, imagens[0]))
                shutil.move(img_path2, os.path.join(destino, imagens[1]))
                print(f"Imagens movidas para: {destino}")
            else:
                print("Classe inválida! As imagens não foram movidas.")
        else:
            destino = os.path.join(train_dir, classe_nome)
            shutil.move(img_path1, os.path.join(destino, imagens[0]))
            shutil.move(img_path2, os.path.join(destino, imagens[1]))
            print(f"Imagens movidas para: {destino}")

print("Classificação finalizada!")

