# --- Importações Necessárias ---
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from werkzeug.utils import secure_filename # Para nomes de arquivo seguros
from reportlab.lib.pagesizes import letter # Para PDF
from reportlab.pdfgen import canvas     # Para PDF
from reportlab.lib.utils import ImageReader # Para PDF
from datetime import datetime
import io # Para lidar com bytes de imagem

# --- Configuração de GPU (CUDA) ---
# O TensorFlow detecta e usa GPUs automaticamente se o ambiente estiver configurado.
# Se nenhuma GPU compatível for encontrada ou configurada, ele usará a CPU.
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
# Isso é crucial para resolver caminhos relativos de forma consistente.
# Assume que app.py está na raiz do projeto (C:\cod\python\AI\)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Diretório para salvar o modelo treinado (ajustado para ser absoluto)
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'model_novo') # Onde o modelo_final_best.h5 está
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'model_final_best.h5')

# Diretórios para as imagens classificadas (ajustados para serem absolutos)
CLASSIFIED_IMAGES_BASE_DIR = os.path.join(PROJECT_ROOT, 'image', 'classificacao')
OUTPUT_GOOD_DIR = os.path.join(CLASSIFIED_IMAGES_BASE_DIR, 'boa')
OUTPUT_BAD_DIR = os.path.join(CLASSIFIED_IMAGES_BASE_DIR, 'ruim')

# Pasta temporária para uploads (ajustada para ser absoluta e dentro do projeto)
TEMP_UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'temp_upload')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Pasta para armazenar relatórios PDF (ajustada para ser absoluta e servida como static)
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Caminho para o log de execução (ajustado para ser absoluto)
EXECUTION_LOG_PATH = os.path.join(PROJECT_ROOT, 'results_novo', 'execution_log.txt')
os.makedirs(os.path.dirname(EXECUTION_LOG_PATH), exist_ok=True) # Garante que a pasta 'results_novo' exista

# Parâmetros de Imagem (devem ser os mesmos usados no treinamento)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3 # RGB

# Nomes das classes (deve ser a mesma ordem usada no treinamento: 0 -> "ruim", 1 -> "boa")
CLASS_NAMES = ["solda_ruim", "solda_boa"] 

# Limiar de probabilidade para classificar como "solda_boa"
CLASSIFICATION_THRESHOLD = 0.8 # Default 0.8


# --- Carregar o Modelo TensorFlow (feito uma vez na inicialização da API) ---
try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERRO FATAL: Não foi possível carregar o modelo de {MODEL_PATH}: {e}")
    print("A API não poderá funcionar sem o modelo. Encerrando.")
    exit() # Encerra a aplicação se o modelo não puder ser carregado


# --- Função de Log Simples ---
def save_log(message: str, log_file=EXECUTION_LOG_PATH):
    """
    Salva uma mensagem de log em um arquivo.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


# --- Função de Pré-processamento de Imagem (adaptada para bytes) ---
def preprocess_image_for_inference(image_bytes: bytes, target_height: int, target_width: int, num_channels: int):
    """
    Decodifica bytes da imagem, redimensiona e pré-processa para a entrada do modelo.
    """
    try:
        img = tf.image.decode_image(image_bytes, channels=num_channels)
        # Verifica se o tensor da imagem decodificada tem um shape válido e não está vazio
        if tf.reduce_prod(tf.shape(img)) == 0 or tf.rank(img) != 3:
             raise ValueError(f"'images' contains no valid shape after decoding. Shape: {tf.shape(img)}")
        if tf.shape(img)[2] != num_channels:
             raise ValueError(f"Decoded image has {tf.shape(img)[2]} channels, expected {num_channels}.")

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
    c.setFont("Helvetica", 12)

    c.drawString(50, height - 50, "Relatório de Classificação de Soldas")
    c.drawString(50, height - 70, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 100

    for res in resultados:
        caminho_temp = res['caminho_temp'] # Caminho onde a imagem original foi salva temporariamente
        classificacao = res['classificacao']
        confianca = res['confianca']
        original_filename = res['original_filename'] # Nome original do arquivo

        c.drawString(50, y, f"Arquivo: {original_filename}")
        y -= 20
        c.drawString(50, y, f"Classificação: {classificacao}")
        y -= 20
        c.drawString(50, y, f"Confiança: {confianca:.4f}")
        y -= 10 # Pequeno espaço

        try:
            # reportlab ImageReader precisa de um caminho de arquivo válido ou um objeto de arquivo aberto
            img_to_draw = ImageReader(caminho_temp)
            # Redimensiona a imagem para caber no PDF
            img_width, img_height = img_to_draw.getSize()
            aspect_ratio = img_height / float(img_width)
            draw_width = 150
            draw_height = draw_width * aspect_ratio
            if draw_height > 150: # Limita a altura também
                draw_height = 150
                draw_width = draw_height / aspect_ratio

            c.drawImage(img_to_draw, 50, y - draw_height - 10, width=draw_width, height=draw_height)
            y -= (draw_height + 10) # Espaço para a imagem
        except Exception as e:
            c.drawString(50, y, f"(Erro ao carregar imagem no PDF: {original_filename} - {e})")
            y -= 20 # Espaço para a mensagem de erro

        y -= 40 # Espaço entre os resultados

        if y < 100: # Se o espaço restante for muito pequeno, cria nova página
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    c.save()


# --- Endpoint de Classificação ---
@app.route('/classificar', methods=['POST'])
async def classificar_imagens():
    save_log("Requisição recebida no endpoint /classificar.")
    files = request.files.getlist('files') # Obtém a lista de arquivos enviados
    
    if not files:
        save_log("Erro: Nenhuma imagem recebida.")
        return jsonify({'status': 'erro', 'message': 'Nenhuma imagem foi enviada.'}), 400

    results_for_pdf = []
    
    # Lista para armazenar caminhos temporários para limpeza posterior
    temp_file_paths_to_clean = [] 

    for file_storage in files:
        original_filename = secure_filename(file_storage.filename)
        temp_filepath = os.path.join(TEMP_UPLOAD_DIR, original_filename)
        
        try:
            # Salva o arquivo temporariamente para processamento e inclusão no PDF
            file_storage.save(temp_filepath)
            temp_file_paths_to_clean.append(temp_filepath)
            save_log(f"Arquivo recebido e salvo temporariamente: {original_filename}")

            # Lê o conteúdo do arquivo para o pré-processamento TensorFlow
            with open(temp_filepath, 'rb') as f:
                image_bytes = f.read()

            # Pré-processar a imagem
            processed_image = preprocess_image_for_inference(
                image_bytes, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS
            )
            
            # Fazer a predição
            prediction_probability = modelo.predict(processed_image, verbose=0)[0][0]
            
            # Classificar e determinar o diretório de destino final
            if prediction_probability >= CLASSIFICATION_THRESHOLD:
                predicted_class_index = 1
                destination_dir = OUTPUT_GOOD_DIR
            else:
                predicted_class_index = 0
                destination_dir = OUTPUT_BAD_DIR
            
            predicted_class_name = CLASS_NAMES[predicted_class_index]

            save_log(f"  Classificado '{original_filename}': {predicted_class_name} (Prob: {prediction_probability:.4f})")

            # Mover a imagem para o diretório de classificação final
            # Usa shutil.move, que removerá do temp_upload_dir
            final_dest_path = os.path.join(destination_dir, original_filename)
            shutil.move(temp_filepath, final_dest_path)
            save_log(f"  Movido '{original_filename}' para: {final_dest_path}")

            results_for_pdf.append({
                'original_filename': original_filename,
                'caminho_temp': final_dest_path, # Agora aponta para o destino final para o PDF
                'classificacao': predicted_class_name,
                'confianca': float(prediction_probability)
            })

        except Exception as e:
            save_log(f"❌ ERRO ao processar '{original_filename}': {e}")
            # Se ocorrer um erro, a imagem pode ter sido movida para temp_upload_dir, mas não para o destino final.
            # Limpeza manual ou mover para uma pasta de erros seria necessário aqui.
            results_for_pdf.append({
                'original_filename': original_filename,
                'caminho_temp': temp_filepath if os.path.exists(temp_filepath) else 'Erro no upload/processamento',
                'classificacao': 'ERRO',
                'confianca': 0.0
            })
            # Não remove o arquivo temporário imediatamente se houver erro, para depuração.

    # Gerar o relatório PDF
    if results_for_pdf:
        pdf_filename = f"relatorio_soldas_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        try:
            gerar_pdf(results_for_pdf, pdf_path)
            save_log(f"Relatório PDF gerado: {pdf_path}")
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
    # debug=True é ótimo para desenvolvimento (reinicia o servidor em mudanças, mostra erros)
    # Mas DEVE ser desativado em produção (debug=False)
    app.run(debug=True, host='0.0.0.0', port=5000)