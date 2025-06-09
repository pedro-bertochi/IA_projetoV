#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image # Apenas para verificação, tf.data faz a leitura

import shutil # Para mover os arquivos


# In[2]:


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


# In[3]:


# --- Configurações da Inferência em Lote ---
# Caminho para o modelo final salvo pelo script de treinamento (model_final_best.h5)
MODEL_PATH = 'C:\\cod\\python\\AI\\model_novo\\model_final_best.h5' 

# Diretório de onde o script vai LER as imagens para classificar
INPUT_IMAGES_DIR = 'C:\\cod\\python\\AI\\image\\classificacao\\novas' 

# Diretórios de DESTINO para onde as imagens classificadas serão MOVIDAS
OUTPUT_BASE_DIR = 'C:\\cod\\python\\AI\\image\\classificacao' # Base para as pastas 'boa' e 'ruim'
OUTPUT_GOOD_DIR = os.path.join(OUTPUT_BASE_DIR, 'boa')
OUTPUT_BAD_DIR = os.path.join(OUTPUT_BASE_DIR, 'ruim')


# In[4]:


# Parâmetros de Imagem (devem ser os mesmos usados no treinamento)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3 # RGB


# In[5]:


# Nomes das classes (deve ser a mesma ordem usada no treinamento: 0 -> "ruim", 1 -> "boa")
CLASS_NAMES = ["solda_ruim", "solda_boa"]


# In[6]:


# Limiar de probabilidade para classificar como "solda_boa"
CLASSIFICATION_THRESHOLD = 0.8


# In[7]:


# --- Função de Pré-processamento (Idêntica à do treinamento para imagens únicas) ---
def preprocess_single_image_for_inference(image_path: str, target_height: int, target_width: int, num_channels: int):
    """
    Carrega, redimensiona e pré-processa uma única imagem para inferência.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")

    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=num_channels)
        
        # Opcional: Adicionar um tf.print para depurar o shape se houver problemas
        # tf.print("DEBUG: Shape after decode_image:", tf.shape(img), "for path:", image_path)

        img = tf.image.resize(img, (target_height, target_width))
        img = tf.image.convert_image_dtype(img, tf.float32) # Normaliza para [0, 1]
        img = tf.expand_dims(img, axis=0) # Adiciona dimensão de batch (para 1 imagem)
        return img
    except Exception as e:
        # Tenta pegar o nome do arquivo para depuração
        file_name = os.path.basename(image_path)
        raise ValueError(f"Erro ao carregar ou pré-processar a imagem '{file_name}': {e}")


# In[8]:


# salvar log de execução .txt
def save_log(message: str, log_file='../results_novo/execution_log.txt'):
    """
    Salva uma mensagem de log em um arquivo.
    """
    with open(log_file, 'a') as f:
        f.write(message + '\n')


# In[9]:


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    print(f"Iniciando inferência em lote para imagens em: {INPUT_IMAGES_DIR}")

    # 1. Criar diretórios de saída se não existirem
    os.makedirs(OUTPUT_GOOD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BAD_DIR, exist_ok=True)
    
    # 2. Carregar o modelo treinado
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERRO ao carregar o modelo de {MODEL_PATH}: {e}")
        print("Certifique-se de que o modelo foi treinado e salvo corretamente.")
        exit()

    # 3. Listar imagens no diretório de entrada
    if not os.path.isdir(INPUT_IMAGES_DIR):
        print(f"❌ ERRO: O diretório de entrada '{INPUT_IMAGES_DIR}' não foi encontrado.")
        exit()

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em '{INPUT_IMAGES_DIR}' para classificar.")
        exit()

    print(f"Encontradas {len(image_files)} imagens para classificar.")

    # 4. Iterar sobre as imagens, classificar e mover
    classified_count = 0
    error_count = 0

    for image_file in image_files:
        image_path = os.path.join(INPUT_IMAGES_DIR, image_file)
        print(f"\nClassificando: {image_file}")

        try:
            # Pré-processar a imagem
            processed_image = preprocess_single_image_for_inference(
                image_path, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS
            )
            
            # Fazer a predição
            prediction_probability = model.predict(processed_image, verbose=0)[0][0] 

            # Classificar e determinar o diretório de destino
            if prediction_probability >= CLASSIFICATION_THRESHOLD:
                predicted_class_index = 1 # Corresponde a 'boa'
                destination_dir = OUTPUT_GOOD_DIR
            else:
                predicted_class_index = 0 # Corresponde a 'ruim'
                destination_dir = OUTPUT_BAD_DIR
            
            predicted_class_name = CLASS_NAMES[predicted_class_index]

            print(f"  Probabilidade: {prediction_probability:.4f}")
            print(f"  Classe Predita: {predicted_class_name}")

            # Mover a imagem para o diretório de destino
            shutil.move(image_path, os.path.join(destination_dir, image_file))
            print(f"  Movida para: {destination_dir}")
            classified_count += 1

            save_log(f"Classificada: {image_file} -> {predicted_class_name} (Probabilidade: {prediction_probability:.4f})")

        except Exception as e:
            error_count += 1
            print(f"❌ ERRO ao classificar e mover '{image_file}': {e}")
            # Opcional: Mover para uma pasta de 'erros' para inspeção manual
            # error_quarantine_dir = os.path.join(OUTPUT_BASE_DIR, 'erros_classificacao')
            # os.makedirs(error_quarantine_dir, exist_ok=True)
            # try:
            #     shutil.move(image_path, os.path.join(error_quarantine_dir, image_file))
            #     print(f"  -> Imagem movida para pasta de erros: {error_quarantine_dir}")
            # except Exception as move_e:
            #     print(f"  -> AVISO: Não foi possível mover para pasta de erros: {move_e}")
            save_log(f"Erro ao processar {image_file}: {e}")

    print("\n--- Processamento em Lote Concluído ---")
    print(f"Total de imagens classificadas e movidas: {classified_count}")
    print(f"Imagens com erro durante o processamento: {error_count}")
    print(f"Resultados da classificação em: {os.path.abspath(OUTPUT_GOOD_DIR)} e {os.path.abspath(OUTPUT_BAD_DIR)}")
    print("Verifique também a pasta de entrada, imagens com erro não serão removidas dela, a menos que movidas para a pasta de erros.")

