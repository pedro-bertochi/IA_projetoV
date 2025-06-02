from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from src.core import criar_modelo, combinar_imagens
import numpy as np
import tensorflow as tf
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

modelo = tf.keras.models.load_model('model/modelo_final_media.h5')

def gerar_pdf(resultados, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    c.drawString(50, height - 50, "Relatório de Classificação de Soldas")
    c.drawString(50, height - 70, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 100

    for idx, res in enumerate(resultados, 1):
        p1, p2 = res['imagens']
        classificacao = res['classificacao']
        confianca = res['confiança']

        # Inserir imagens
        try:
            img1 = ImageReader(p1)
            c.drawImage(img1, 50, y - 150, width=150, height=150)
        except:
            c.drawString(50, y, f"(Erro ao carregar imagem: {os.path.basename(p1)})")

        try:
            img2 = ImageReader(p2)
            c.drawImage(img2, 220, y - 150, width=150, height=150)
        except:
            c.drawString(220, y, f"(Erro ao carregar imagem: {os.path.basename(p2)})")

        # Escreve texto abaixo
        y -= 170
        c.drawString(50, y, f"Classificação: {classificacao}")
        y -= 20
        c.drawString(50, y, f"Confiança: {confianca:.2f}")

        y -= 40  # Espaço entre pares

        if y < 200:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    c.save()

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
            'imagens': (p1, p2),
            'classificacao': destino,
            'confiança': float(pred)
        })

    # Gera nome único pro PDF
    pdf_filename = f"relatorio_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_path = os.path.join('static/reports', pdf_filename)

    # Gera PDF com imagens antes de mover
    gerar_pdf(resultados, pdf_path)

    # Agora move as imagens
    for res in resultados:
        p1, p2 = res['imagens']
        destino = res['classificacao']
        os.makedirs(f'../image/classificacao/{destino}', exist_ok=True)
        shutil.move(p1, f'../image/classificacao/{destino}/{os.path.basename(p1)}')
        shutil.move(p2, f'../image/classificacao/{destino}/{os.path.basename(p2)}')

    return jsonify({
        'status': 'sucesso',
        'relatorio_url': f"static/reports/{pdf_filename}"  # URL relativa
    })

@app.route('/static/reports/<path:filename>')
def download_pdf(filename):
    pdf_path = os.path.join('static/reports', filename)
    return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True)
