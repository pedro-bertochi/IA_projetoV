import numpy as np
import os
from tensorflow.keras.models import load_model
from src.core.processar_imagens import combinar_imagens

modelo = load_model('model/modelo_solda_resnet50.h5')

def classificar_pares(caminhos_imgs):
    resultados = []
    for i in range(0, len(caminhos_imgs), 2):
        p1, p2 = caminhos_imgs[i], caminhos_imgs[i + 1]
        entrada = combinar_imagens(p1, p2)
        entrada = np.expand_dims(entrada / 255.0, axis=0)
        pred = modelo.predict(entrada)[0][0]
        classificacao = 'boa' if pred >= 0.8 else 'ruim'
        resultados.append({
            'imagem1': os.path.basename(p1),
            'imagem2': os.path.basename(p2),
            'confiança': float(pred),
            'classificacao': classificacao
        })
    return resultados
