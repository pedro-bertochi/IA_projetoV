from flask import Flask, request, jsonify
import os
from src.core import criar_modelo, combinar_imagens
import numpy as np
import tensorflow as tf
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

modelo = tf.keras.models.load_model('../model/modelo_solda_resnet50.h5')

@app.route('/classificar', methods=['POST'])
def classificar():
    files = request.files.getlist('files')
    
    # Salva temporariamente
    temp_dir = './temp_upload'
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        path = os.path.join(temp_dir, filename)
        file.save(path)
        file_paths.append(path)

    if len(file_paths) % 2 != 0:
        file_paths = file_paths[:-1]  # ignora última se for ímpar

    resultados = []

    for i in range(0, len(file_paths), 2):
        p1, p2 = file_paths[i], file_paths[i+1]
        entrada = combinar_imagens(p1, p2)
        entrada = np.expand_dims(entrada / 255.0, axis=0)
        pred = modelo.predict(entrada)[0][0]
        destino = 'boa' if pred >= 0.8 else 'ruim'
        resultados.append({
            'par': [os.path.basename(p1), os.path.basename(p2)],
            'classificacao': destino,
            'confiança': float(pred)
        })

        # Opcional: mover para pasta destino
        os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)
        shutil.move(p1, f'../image/classificacao/{destino}/{os.path.basename(p1)}')
        shutil.move(p2, f'../image/classificacao/{destino}/{os.path.basename(p2)}')

    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
