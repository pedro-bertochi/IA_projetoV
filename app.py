# --- Importações Necessárias ---
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
import io
import json # Para lidar com o histórico de análises (ainda útil para o PDF)

# --- Configuração de GPU (CUDA) ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) detectada(s) e configurada(s) para uso dinâmico de memória: {gpus}")
        print("TensorFlow utilizará a(s) GPU(s) disponível(is) para aceleração.")
    else:
        print("❌ Nenhuma GPU compatível com CUDA detectada. O TensorFlow será executado na CPU.")
        print("Continuando a execução na CPU.")
except RuntimeError as e:
    print(f"❌ Erro ao configurar GPU: {e}")
    print("O TensorFlow será executado na CPU devido ao erro na configuração da GPU.")
# --- Fim da Configuração de GPU ---


# --- Configurações da Aplicação e Caminhos ---
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir requisições de diferentes origens

# Define o diretório raiz do projeto a partir do qual a API está sendo executada
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Diretório para salvar o modelo treinado
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'model_novo')
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'model_final_best.h5')

# Diretórios para as imagens classificadas
CLASSIFIED_IMAGES_BASE_DIR = os.path.join(PROJECT_ROOT, 'image', 'classificacao')
OUTPUT_GOOD_DIR = os.path.join(CLASSIFIED_IMAGES_BASE_DIR, 'boa')
OUTPUT_BAD_DIR = os.path.join(CLASSIFIED_IMAGES_BASE_DIR, 'ruim')

# Pasta temporária para uploads (onde as imagens ficam ANTES da análise)
TEMP_UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'temp_upload')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Pasta para armazenar relatórios PDF
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'static', 'reports') # Manter dentro de 'static' é uma boa prática
os.makedirs(REPORTS_DIR, exist_ok=True)

# Caminho para o log de execução
EXECUTION_LOG_PATH = os.path.join(PROJECT_ROOT, 'results_novo', 'execution_log.txt')
os.makedirs(os.path.dirname(EXECUTION_LOG_PATH), exist_ok=True)

# Caminho para o arquivo JSON de histórico de análises (simula um DB)
ANALYSIS_HISTORY_PATH = os.path.join(PROJECT_ROOT, 'results_novo', 'analysis_history.json')
# Garante que o arquivo de histórico exista e seja um JSON válido
if not os.path.exists(ANALYSIS_HISTORY_PATH):
    with open(ANALYSIS_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump([], f) # Inicializa com uma lista vazia

# Parâmetros de Imagem (devem ser os mesmos usados no treinamento)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3 # RGB

# Nomes das classes (deve ser a mesma ordem usada no treinamento: 0 -> "ruim", 1 -> "boa")
CLASS_NAMES = ["solda_ruim", "solda_boa"]

# Limiar de probabilidade para classificar como "solda_boa"
CLASSIFICATION_THRESHOLD = 0.8 # Default 0.8

# --- Carregar o Modelo TensorFlow (feito uma vez na inicialização da API) ---
modelo = None # Inicializa como None
try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERRO FATAL: Não foi possível carregar o modelo de {MODEL_PATH}: {e}")
    print("A API não poderá funcionar sem o modelo. Algumas funcionalidades serão limitadas.")


# --- Função de Log Simples ---
def save_log(message: str, log_file=EXECUTION_LOG_PATH):
    """
    Salva uma mensagem de log em um arquivo.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# --- Funções de Histórico de Análises (Simulação de DB - usada para o PDF) ---
def load_analysis_history():
    """Carrega o histórico de análises do arquivo JSON."""
    if not os.path.exists(ANALYSIS_HISTORY_PATH):
        return []
    with open(ANALYSIS_HISTORY_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            save_log(f"AVISO: Arquivo de histórico '{ANALYSIS_HISTORY_PATH}' está vazio ou corrompido. Reiniciando.")
            return []

def save_analysis_history(history_data):
    """Salva o histórico de análises no arquivo JSON."""
    with open(ANALYSIS_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4)

# --- Função de Pré-processamento de Imagem (adaptada para bytes) ---
def preprocess_image_for_inference(image_bytes: bytes, target_height: int, target_width: int, num_channels: int):
    """
    Decodifica bytes da imagem, redimensiona e pré-processa para a entrada do modelo.
    """
    try:
        img = tf.image.decode_image(image_bytes, channels=num_channels)
        
        if tf.reduce_prod(tf.shape(img)) == 0 or tf.rank(img) != 3:
            raise ValueError(f"Imagem decodificada tem shape inválido ou é vazia: {tf.shape(img)}")
        if tf.shape(img)[2] != num_channels:
            raise ValueError(f"Imagem decodificada tem {tf.shape(img)[2]} canais, esperado {num_channels}.")

        img = tf.image.resize(img, (target_height, target_width))
        img = tf.image.convert_image_dtype(img, tf.float32) # Normaliza para [0, 1]
        img = tf.expand_dims(img, axis=0) # Adiciona dimensão de batch (para 1 imagem)
        return img
    except Exception as e:
        raise ValueError(f"Erro ao pré-processar imagem: {e}")

# --- Função para Gerar Relatório PDF ---
def gerar_pdf(resultados, pdf_path):
    """
    Gera um relatório PDF com os resultados da classificação.
    """
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Relatório de Classificação de Soldas")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Data de Geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, height - 80, width - 50, height - 80) # Linha divisória

    y = height - 120
    item_count = 0

    for res in resultados:
        # Se for o primeiro item em uma nova página, ajusta o cabeçalho
        if y < 100 and item_count > 0: # 100 é o limite inferior para iniciar um novo item
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Relatório de Classificação de Soldas (Continuação)")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Data de Geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.line(50, height - 80, width - 50, height - 80)
            y = height - 120 # Reseta y para o início da nova página

        # 'caminho_temp' aqui virá do `shutil.move` que o envia para a pasta final
        caminho_img_para_pdf = res.get('caminho_temp')
        
        # Em `results_for_pdf` no endpoint `/classificar`, 'original_filename' é usado.
        original_filename = res.get('original_filename', 'N/A')
        classificacao = res.get('classificacao', 'N/A')
        confianca = res.get('confianca', 0.0) # Já está em 0-1, não precisa de /100

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Arquivo: {original_filename}")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Classificação: {classificacao}")
        y -= 15
        c.drawString(50, y, f"Confiança: {confianca:.2%}")
        y -= 10 # Pequeno espaço

        try:
            if caminho_img_para_pdf and os.path.exists(caminho_img_para_pdf):
                img_to_draw = ImageReader(caminho_img_para_pdf)
                img_width, img_height = img_to_draw.getSize()
                aspect_ratio = img_height / float(img_width)
                
                draw_width = 150
                draw_height = draw_width * aspect_ratio

                if draw_height > 100:
                    draw_height = 100
                    draw_width = draw_height / aspect_ratio

                if y - draw_height - 20 < 50:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(50, height - 50, "Relatório de Classificação de Soldas (Continuação)")
                    c.setFont("Helvetica", 10)
                    c.drawString(50, height - 70, f"Data de Geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    c.line(50, height - 80, width - 50, height - 80)
                    y = height - 120
                
                c.drawImage(img_to_draw, 50, y - draw_height - 10, width=draw_width, height=draw_height)
                y -= (draw_height + 20)
            else:
                c.drawString(50, y, f"(Imagem não disponível ou movida: {original_filename})")
                y -= 20
        except Exception as e:
            save_log(f"Erro ao carregar imagem {original_filename} no PDF: {e}")
            c.drawString(50, y, f"(Erro ao carregar imagem no PDF: {e})")
            y -= 20

        c.line(50, y - 10, width - 50, y - 10)
        y -= 30
        item_count += 1

    c.save()

# --- Endpoint de Classificação (Reativado e Modificado para seu uso original) ---
@app.route('/classificar', methods=['POST'])
async def classificar_imagens():
    save_log("Requisição recebida no endpoint /classificar.")
    files = request.files.getlist('files')
    
    if not files:
        save_log("Erro: Nenhuma imagem recebida.")
        return jsonify({'status': 'erro', 'message': 'Nenhuma imagem foi enviada.'}), 400

    if modelo is None:
        save_log("Erro: Modelo não carregado. Não é possível classificar.")
        return jsonify({'status': 'erro', 'message': 'Modelo de IA não está carregado. Não é possível classificar.'}), 503

    results_for_pdf = []
    
    # Lista para armazenar caminhos temporários para limpeza posterior (se não movidos)
    temp_file_paths_to_clean = [] 

    for file_storage in files:
        original_filename = secure_filename(file_storage.filename)
        # Use um ID único para o arquivo temporário, para evitar sobrescrever
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{original_filename}"
        temp_filepath = os.path.join(TEMP_UPLOAD_DIR, unique_filename)
        
        try:
            file_storage.save(temp_filepath)
            temp_file_paths_to_clean.append(temp_filepath)
            save_log(f"Arquivo recebido e salvo temporariamente: {original_filename}")

            with open(temp_filepath, 'rb') as f:
                image_bytes = f.read()

            processed_image = preprocess_image_for_inference(
                image_bytes, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS
            )
            
            prediction_probability = modelo.predict(processed_image, verbose=0)[0][0]
            
            if prediction_probability >= CLASSIFICATION_THRESHOLD:
                predicted_class_index = 1
                destination_dir = OUTPUT_GOOD_DIR
                classificacao_texto = "solda_boa"
            else:
                predicted_class_index = 0
                destination_dir = OUTPUT_BAD_DIR
                classificacao_texto = "solda_ruim"
            
            # Mover a imagem para o diretório de classificação final
            final_dest_path = os.path.join(destination_dir, original_filename)
            # Verifica se o arquivo já existe no destino final para evitar erro de sobrescrita se o nome original não for único
            if os.path.exists(final_dest_path):
                name, ext = os.path.splitext(original_filename)
                final_dest_path = os.path.join(destination_dir, f"{name}_{datetime.now().strftime('%H%M%S%f')}{ext}")

            shutil.move(temp_filepath, final_dest_path)
            save_log(f"Movido '{original_filename}' para: {final_dest_path}")

            results_for_pdf.append({
                'original_filename': original_filename,
                'caminho_temp': final_dest_path, # Agora aponta para o destino final para o PDF
                'classificacao': classificacao_texto,
                'confianca': float(prediction_probability)
            })

        except Exception as e:
            save_log(f"❌ ERRO ao processar '{original_filename}': {e}")
            # Tenta limpar o arquivo temporário mesmo em caso de erro
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                save_log(f"Arquivo temporário '{unique_filename}' removido após erro de processamento.")
            
            results_for_pdf.append({
                'original_filename': original_filename,
                'caminho_temp': None, # Indica que a imagem não foi salva
                'classificacao': 'ERRO',
                'confianca': 0.0
            })
            
    # Gerar o relatório PDF
    if results_for_pdf:
        pdf_filename = f"relatorio_soldas_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        # O REPORTS_DIR já está dentro de 'static'.
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        try:
            gerar_pdf(results_for_pdf, pdf_path)
            save_log(f"Relatório PDF gerado: {pdf_path}")
            # A URL para o PDF precisa refletir o caminho 'static' para ser acessível pelo navegador
            return jsonify({
                'status': 'sucesso',
                'message': f'Classificação de {len(results_for_pdf)} imagens concluída. Relatório PDF disponível.',
                'relatorio_url': f"static/reports/{pdf_filename}"
            })
        except Exception as e:
            save_log(f"❌ ERRO ao gerar PDF: {e}")
            return jsonify({'status': 'erro', 'message': f'Classificação concluída, mas erro ao gerar PDF: {e}'}), 500
    else:
        save_log("Nenhuma imagem processada com sucesso para gerar PDF.")
        return jsonify({'status': 'erro', 'message': 'Nenhuma imagem foi processada com sucesso.'}), 400

# --- Endpoint para Download de Relatórios PDF ---
@app.route('/static/reports/<path:filename>')
def download_pdf(filename):
    """
    Permite o download de relatórios PDF.
    """
    pdf_path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(pdf_path):
        save_log(f"Servindo PDF: {filename}")
        return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=filename)
    else:
        save_log(f"Erro: PDF não encontrado para download: {filename}")
        return "Relatório não encontrado.", 404

# --- Endpoint de Saúde da API ---
@app.route('/health')
def health_check():
    """
    Verifica se a API está funcionando e o modelo carregado.
    """
    if modelo is not None:
        return jsonify({"status": "ok", "message": "API e Modelo de solda carregados."}), 200
    else:
        return jsonify({"status": "erro", "message": "API funcionando, mas modelo não carregado."}), 500

# --- Bloco de Execução Principal da Flask App ---
if __name__ == '__main__':
    # Cria os diretórios iniciais
    os.makedirs(OUTPUT_GOOD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BAD_DIR, exist_ok=True)
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EXECUTION_LOG_PATH), exist_ok=True)

    save_log("Iniciando a aplicação Flask...")
    # debug=True é ótimo para desenvolvimento, mas DEVE ser desativado em produção (debug=False)
    app.run(debug=True, host='0.0.0.0', port=5000)