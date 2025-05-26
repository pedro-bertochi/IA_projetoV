#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from core import criar_modelo  # mesma função usada no treino


# In[ ]:


# Caminho onde estão os modelos fine-tuned
model_dir = '../model'
n_folds = 5


# In[ ]:


# Carrega todos os modelos fine-tuned
modelos = []
for fold in range(1, n_folds + 1):
    caminho = f'{model_dir}/fine_tuned_fold{fold}.h5'
    modelos.append(load_model(caminho))


# In[ ]:


# Verifica se as arquiteturas são compatíveis
for i in range(1, len(modelos)):
    if len(modelos[i].get_weights()) != len(modelos[0].get_weights()):
        raise ValueError("Modelos têm arquiteturas diferentes!")


# In[ ]:


# Faz a média dos pesos
pesos_medios = []
for pesos_camadas in zip(*[m.get_weights() for m in modelos]):
    pesos_medios.append(np.mean(pesos_camadas, axis=0))


# In[ ]:


# Cria novo modelo com a mesma arquitetura e aplica os pesos médios
modelo_final = criar_modelo()
modelo_final.set_weights(pesos_medios)


# In[ ]:


# Salva o modelo final (pode ser .h5 ou SavedModel para produção)
modelo_final.save(f'{model_dir}/modelo_final_media.h5')
print("✅ Modelo final salvo com sucesso!")

