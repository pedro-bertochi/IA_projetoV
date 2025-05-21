from flask import Blueprint, request, jsonify, send_file, current_app
import os
import uuid
from services.classifier import classificar_pares
from services.report import gerar_pdf

api_bp = Blueprint('api', __name__)

@api_bp.route('/classify', methods=['POST'])
def classify():
    arquivos = request.files.getlist('images')
    if not arquivos or len(arquivos) % 2 != 0:
        return jsonify({'erro': 'Envie um número par de imagens.'}), 400

    caminhos = []
    for arquivo in arquivos:
        nome_arquivo = f"{uuid.uuid4().hex}_{arquivo.filename}"
        caminho = os.path.join(current_app.config['UPLOAD_FOLDER'], nome_arquivo)
        arquivo.save(caminho)
        caminhos.append(caminho)

    resultados = classificar_pares(caminhos)

    nome_pdf = f"relatorio_{uuid.uuid4().hex}.pdf"
    caminho_pdf = os.path.join(current_app.config['REPORT_FOLDER'], nome_pdf)
    gerar_pdf(resultados, caminho_pdf)

    return jsonify({
        'resultados': resultados,
        'relatorio': f"/report/{nome_pdf}"
    })

@api_bp.route('/report/<nome>')
def baixar_pdf(nome):
    caminho_pdf = os.path.join(current_app.config['REPORT_FOLDER'], nome)
    if not os.path.exists(caminho_pdf):
        return jsonify({'erro': 'Relatório não encontrado'}), 404
    return send_file(caminho_pdf, as_attachment=True)
