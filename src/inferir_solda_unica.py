# src/inferir_solda_unica.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

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


# --- Configurações da Inferência ---
# Caminho para o modelo final salvo pelo script de treinamento (model_final_best.h5)
MODEL_PATH = 'C:\\cod\\python\\AI\\model_novo\\model_final_best.h5' 

# Caminho para a imagem que você quer classificar
# Coloque uma imagem AQUI que NÃO FAZ PARTE do seu dataset de treino/validação/teste
IMAGE_TO_CLASSIFY_PATH = 'C:\\cod\\python\\AI\\image\\classificacao\\novas\\SNAP0111.jpg' # <-- ALTERE ESTE CAMINHO

# Parâmetros de Imagem (devem ser os mesmos usados no treinamento)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3 # RGB

# Nomes das classes (deve ser a mesma ordem usada no treinamento: 0 -> "ruim", 1 -> "boa")
CLASS_NAMES = ["solda_ruim", "solda_boa"] 

# Limiar de probabilidade para classificar como "solda_boa"
# O modelo outputa um valor entre 0 e 1 (probabilidade). Se > LIMIAR_BOA, é "boa".
CLASSIFICATION_THRESHOLD = 0.5 


# --- Função de Pré-processamento (Deve ser idêntica à do treinamento para imagens únicas) ---
def preprocess_single_image_for_inference(image_path: str, target_height: int, target_width: int, num_channels: int):
    """
    Carrega, redimensiona e pré-processa uma única imagem para inferência.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")

    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=num_channels)
        img = tf.image.resize(img, (target_height, target_width))
        img = tf.image.convert_image_dtype(img, tf.float32) # Normaliza para [0, 1]
        img = tf.expand_dims(img, axis=0) # Adiciona dimensão de batch (para 1 imagem)
        return img
    except Exception as e:
        raise ValueError(f"Erro ao carregar ou pré-processar a imagem '{image_path}': {e}")


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    # Crie o diretório para as novas imagens de teste se não existir
    os.makedirs(os.path.dirname(IMAGE_TO_CLASSIFY_PATH) or '.', exist_ok=True)
    
    print(f"Iniciando inferência para a imagem: {IMAGE_TO_CLASSIFY_PATH}")

    # 1. Carregar o modelo treinado
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERRO ao carregar o modelo de {MODEL_PATH}: {e}")
        print("Certifique-se de que o modelo foi treinado e salvo corretamente.")
        exit()

    # 2. Pré-processar a imagem para inferência
    try:
        processed_image = preprocess_single_image_for_inference(
            IMAGE_TO_CLASSIFY_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS
        )
        print("✅ Imagem pré-processada com sucesso.")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ ERRO na imagem: {e}")
        print("Verifique o caminho da imagem e se o arquivo é válido.")
        exit()

    # 3. Fazer a predição
    print("Fazendo a predição...")
    # O modelo retorna uma probabilidade entre 0 e 1
    # [0][0] para extrair o valor escalar do tensor de saída
    prediction_probability = model.predict(processed_image, verbose=0)[0][0] 

    # 4. Classificar a predição baseada no limiar
    if prediction_probability >= CLASSIFICATION_THRESHOLD:
        predicted_class_index = 1 # Corresponde a 'boa'
    else:
        predicted_class_index = 0 # Corresponde a 'ruim'
    
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    print("\n--- Resultado da Classificação ---")
    print(f"Probabilidade: {prediction_probability:.4f}")
    print(f"Classe Predita: {predicted_class_name}")

    if predicted_class_name == "solda_boa":
        print("🎉 A solda foi classificada como BOA!")
    else:
        print("⚠️ A solda foi classificada como RUIM.")

    print("\nInferência concluída.")