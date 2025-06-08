from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Pasta para armazenar relatórios
os.makedirs('static/reports', exist_ok=True)

modelo = tf.keras.models.load_model('model/model_final_best.h5')

def preprocessar_imagem(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))  # ajuste ao input do modelo
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def gerar_pdf(resultados, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    c.drawString(50, height - 50, "Relatório de Classificação de Soldas (1 imagem)")
    c.drawString(50, height - 70, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 100

    for res in resultados:
        caminho = res['imagem']
        classificacao = res['classificacao']
        confianca = res['confiança']

        try:
            img = ImageReader(caminho)
            c.drawImage(img, 50, y - 150, width=150, height=150)
        except:
            c.drawString(50, y, f"(Erro ao carregar imagem: {os.path.basename(caminho)})")

        y -= 170
        c.drawString(50, y, f"Classificação: {classificacao}")
        y -= 20
        c.drawString(50, y, f"Confiança: {confianca:.2f}")

        y -= 40

        if y < 200:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    c.save()

@app.route('/classificar', methods=['POST'])
def classificar():
    files = request.files.getlist('files')
    
    temp_dir = './temp_upload'
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        path = os.path.join(temp_dir, filename)
        file.save(path)
        file_paths.append(path)

    resultados = []

    for caminho in file_paths:
        entrada = preprocessar_imagem(caminho)
        pred = modelo.predict(entrada)[0][0]
        destino = 'boa' if pred >= 0.5 else 'ruim'
        resultados.append({
            'imagem': caminho,
            'classificacao': destino,
            'confiança': float(pred)
        })

    pdf_filename = f"relatorio_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_path = os.path.join('static/reports', pdf_filename)
    gerar_pdf(resultados, pdf_path)

    for res in resultados:
        caminho = res['imagem']
        destino = res['classificacao']
        os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)
        shutil.move(caminho, f'../image/classificacao/{destino}/{os.path.basename(caminho)}')

    return jsonify({
        'status': 'sucesso',
        'relatorio_url': f"static/reports/{pdf_filename}"
    })

@app.route('/static/reports/<path:filename>')
def download_pdf(filename):
    pdf_path = os.path.join('static/reports', filename)
    return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True)
