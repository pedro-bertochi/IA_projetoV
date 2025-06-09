#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[3]:


# --- Configuração para uso de GPU (CUDA) ---
# O TensorFlow detecta e usa GPUs automaticamente se o ambiente estiver configurado.
# Se nenhuma GPU compatível for encontrada ou configurada, ele usará a CPU.
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Configura a memória da GPU para crescer dinamicamente.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) detectada(s) e configurada(s) para uso dinâmico de memória: {gpus}")
        print("TensorFlow utilizará a(s) GPU(s) disponível(is) para aceleração.")
    else:
        print("❌ Nenhuma GPU compatível com CUDA detectada. O TensorFlow será executado na CPU.")
        print("Continuando a execução na CPU.")
except RuntimeError as e:
    # Captura erros que podem ocorrer se a GPU não estiver configurada corretamente
    print(f"❌ Erro ao configurar GPU: {e}")
    print("O TensorFlow será executado na CPU devido ao erro na configuração da GPU.")
# --- Fim da Configuração de GPU ---


# In[4]:


# --- Função para carregar e pré-processar uma única imagem ---
def load_and_preprocess_single_image(image_path: str, target_size: tuple = (224, 224)):
    """
    Carrega uma única imagem, redimensiona e aplica o pré-processamento
    necessário para o modelo ResNet50 pré-treinado no ImageNet.

    Args:
        image_path (str): Caminho para o arquivo da imagem.
        target_size (tuple): Tamanho (largura, altura) para redimensionar a imagem.
                             Para ResNet50, o padrão é (224, 224).

    Returns:
        numpy.ndarray: Imagem pré-processada, pronta para a entrada do modelo.
                       Formato (1, height, width, channels).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")

    try:
        # Carregar a imagem usando PIL (Pillow)
        img = Image.open(image_path).convert('RGB') # Garante 3 canais de cor
        # Redimensionar a imagem
        img = img.resize(target_size, Image.LANCZOS) # LANCZOS para melhor qualidade de redimensionamento
        # Converter para array NumPy
        img_array = np.array(img)
        # Adicionar dimensão de batch (o modelo espera um lote de imagens)
        img_array = np.expand_dims(img_array, axis=0)
        # Aplicar o pré-processamento específico do ResNet50 (normalização, etc.)
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed
    except Exception as e:
        raise ValueError(f"Erro ao carregar ou pré-processar a imagem '{image_path}': {e}")



# In[5]:


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    # --- Configurações ---
    # Caminho para a imagem que você quer analisar
    IMAGE_TO_ANALYZE_PATH = '../image/treinamento/boa\SNAP0454.jpg'

    # Crie o diretório de teste se não existir
    os.makedirs(os.path.dirname(IMAGE_TO_ANALYZE_PATH), exist_ok=True)

    print(f"Iniciando análise da imagem: {IMAGE_TO_ANALYZE_PATH}")

    # 1. Carregar o modelo ResNet50 pré-treinado no ImageNet
    # 'include_top=True' é crucial para ter a camada de classificação ImageNet
    print("Carregando modelo ResNet50 pré-treinado no ImageNet...")
    try:
        model = ResNet50(weights='imagenet', include_top=True)
        print("✅ Modelo ResNet50 carregado com sucesso.")
    except Exception as e:
        print(f"❌ ERRO ao carregar o modelo ResNet50: {e}")
        print("Verifique sua conexão com a internet ou a instalação do TensorFlow/Keras.")
        exit()

    # 2. Carregar e pré-processar a imagem
    try:
        processed_image = load_and_preprocess_single_image(IMAGE_TO_ANALYZE_PATH)
        print("✅ Imagem pré-processada com sucesso.")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ ERRO na imagem: {e}")
        print("Por favor, verifique se o caminho da imagem está correto e se o arquivo é válido.")
        exit()

    # 3. Fazer a predição
    print("Fazendo a predição...")
    predictions = model.predict(processed_image)

    # 4. Decodificar as predições (transformar os números em nomes de classes ImageNet)
    # 'top=5' mostra as 5 principais previsões
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    print("\n--- Principais Predições (ImageNet) ---")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}: {label} ({score*100:.2f}%)")

    print("\nAnálise concluída.")

