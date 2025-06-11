#!/usr/bin/env python
# coding: utf-8

# In[65]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from PIL import Image


# In[66]:


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
    print(f"❌ Erro ao configurar GPU: {e}")
    print("O TensorFlow será executado na CPU devido ao erro na configuração da GPU.")


# In[67]:


DATA_DIR = 'C:\\cod\\python\\AI\\image\\treinamento'
MODEL_SAVE_DIR = 'C:\\cod\\python\\AI\\model_novo' # Novo diretório para não conflitar com o projeto existente
LOGS_SAVE_DIR = 'C:\\cod\\python\\AI\\logs_novo'   # Novo diretório para logs
RESULTS_SAVE_DIR = 'C:\\cod\\python\\AI\\results_novo'


# In[68]:


# Parâmetros de Imagem
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3 # RGB


# In[69]:


BATCH_SIZE = 16 # Aumentei o batch size para aproveitar melhor a GPU
EPOCHS_PHASE1 = 5  # Épocas para a fase 1 (treinar apenas o cabeçalho)
EPOCHS_PHASE2 = 10 # Épocas para a fase 2 (fine-tuning das últimas camadas)
LEARNING_RATE_PH1 = 1e-3 # LR maior para a fase 1 (cabeçalho)
LEARNING_RATE_PH2 = 1e-5 # LR menor para a fase 2 (fine-tuning)
ES_PATIENCE = 5


# In[70]:


# Camadas da ResNet50 a descongelar na Fase 2
# 0: Congela toda a base. -1: Descongela toda a base. >0: Descongela as últimas N camadas.
UNFREEZE_LAST_N_LAYERS_PH2 = 3 # Exemplo: descongelar as últimas 3 camadas da ResNet50 na Fase 2


# In[71]:


# Parâmetros de Divisão de Dados
TRAIN_RATIO = 0.70 # 70% para treino
VAL_RATIO = 0.15   # 15% para validação
TEST_RATIO = 0.15  # 15% para teste
GLOBAL_RANDOM_STATE = 42 # Para reprodutibilidade


# In[72]:


# Nomes das classes (a ordem deve corresponder aos rótulos numéricos 0 e 1)
# 'ruim' corresponde ao rólo 0, 'boa' corresponde ao rótulo 1
CLASS_NAMES = ["solda_ruim", "solda_boa"]

# --- Criação de Diretórios ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)


# In[73]:


# --- 1. Função para Carregar Caminhos das Imagens e Rótulos (Imagens Únicas) ---
def load_image_paths_and_labels_single_image(data_dir):
    """
    Carrega os caminhos das imagens e seus rótulos para imagens únicas.
    Assume a estrutura: data_dir/boa/img_x.jpg, data_dir/ruim/img_y.jpg
    """
    all_paths = []
    all_labels = []

    # A ordem das classes aqui define qual rótulo numérico (0 ou 1) elas receberão
    # 'ruim' será 0, 'boa' será 1
    for i, class_name_folder in enumerate(['ruim', 'boa']): # Iterar sobre as pastas
        class_path = os.path.join(data_dir, class_name_folder)
        if not os.path.isdir(class_path):
            print(f"AVISO: Diretório de classe '{class_path}' não encontrado. Pulando.")
            continue

        # Lista os arquivos de imagem
        files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for file_name in files:
            path = os.path.join(class_path, file_name)
            all_paths.append(path)
            all_labels.append(i) # Rótulo numérico: 0 para 'ruim', 1 para 'boa'

    return all_paths, all_labels


# In[74]:


# --- Função auxiliar para imprimir erros de forma legível dentro do tf.data pipeline ---
def _print_debug_error(image_path_tensor, error_message_tensor):
    """Auxiliar para imprimir erros com caminho da imagem resolvido."""
    # tf.py_function passa tensores, precisamos convertê-los para strings Python
    image_path_str = image_path_tensor.numpy().decode('utf-8')
    error_message_str = error_message_tensor.numpy().decode('utf-8')
    print(f"DEBUG_ERROR: Erro ao processar imagem '{image_path_str}': {error_message_str}. Retornando imagem dummy.")
    # tf.py_function precisa retornar algo, mesmo que não seja usado.
    return 0 # Valor dummy


# In[ ]:


# --- 2. Função para Pré-processar e Augmentar Imagens Únicas (para tf.data.Dataset) ---
def preprocess_and_augment_single_image(image_path, label, augment=False):
    """
    Carrega, redimensiona e pré-processa uma única imagem.
    Aplica aumento de dados se 'augment' for True.
    """
    try:
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=NUM_CHANNELS)
        img.set_shape([None, None, NUM_CHANNELS])  # Define o shape inicial como desconhecido

        # --- DEBUG: Verifica o shape APÓS a decodificação ---
        # Converte o caminho do tensor para string para usar nas mensagens de erro
        image_path_str_tensor = tf.strings.format("{}", image_path)

        # Verifica se o tensor da imagem decodificada tem um shape válido e não está vazio
        if tf.reduce_prod(tf.shape(img)) == 0 or tf.rank(img) != 3:
            # Chama a função auxiliar Python para imprimir o erro com o caminho resolvido
            tf.py_function(
                _print_debug_error,
                [image_path, tf.constant("Decode_image resultou em tensor vazio ou com shape inválido ('images' contains no shape).")],
                tf.int32 # O tipo de retorno da função Python
            )
            return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype=tf.float32), tf.cast(label, tf.float32)

        # Garante o número correto de canais
        if tf.shape(img)[2] != NUM_CHANNELS:
            tf.py_function(
                _print_debug_error,
                [image_path, tf.constant(f"Número de canais incorreto ({tf.shape(img)[2]}), esperado {NUM_CHANNELS}.")],
                tf.int32
            )
            return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype=tf.float32), tf.cast(label, tf.float32)

        img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Aplicar aumento de dados
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

        return img, tf.cast(label, tf.float32)

    except tf.errors.InvalidArgumentError as e:
        # Erro de argumento inválido ao decodificar (formato de arquivo inválido, etc.)
        tf.py_function(
            _print_debug_error,
            [image_path, tf.constant(f"tf.errors.InvalidArgumentError ao decodificar: {str(e)}")],
            tf.int32
        )
        return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype=tf.float32), tf.cast(label, tf.float32)
    except Exception as e:
        # Captura outros erros inesperados
        tf.py_function(
            _print_debug_error,
            [image_path, tf.constant(f"Erro inesperado no pré-processamento: {str(e)}")],
            tf.int32
        )
        return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype=tf.float32), tf.cast(label, tf.float32)


# In[76]:


# --- 3. Função para Criar o Modelo ResNet50 (para Imagens Únicas) ---
def create_resnet_model_for_single_images(train_last_n_base_layers: int = 0):
    """
    Cria e compila o modelo ResNet50 adaptado para imagens únicas.

    Args:
        train_last_n_base_layers (int): Número de últimas camadas da base ResNet50
                                        a serem descongeladas e treináveis.
                                        0: Congela toda a base (treina apenas o cabeçalho).
                                        -1: Descongela toda a base.
                                        >0: Descongela as últimas N camadas.
    """
    # O input_shape agora é (altura, largura, canais)
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congela TODAS as camadas da base por padrão.
    base_model.trainable = False

    # Descongela as últimas N camadas da base, se o parâmetro for > 0
    if train_last_n_base_layers is not None and train_last_n_base_layers > 0:
        num_base_layers = len(base_model.layers)
        start_unfreeze_idx = max(0, num_base_layers - train_last_n_base_layers)
        for layer in base_model.layers[start_unfreeze_idx:]:
            layer.trainable = True
        print(f"Modelo: Descongeladas as últimas {train_last_n_base_layers} camadas da base ResNet50.")
    elif train_last_n_base_layers is not None and train_last_n_base_layers < 0:
        # Descongela todas as camadas da base
        for layer in base_model.layers:
            layer.trainable = True
        print("Modelo: Descongeladas TODAS as camadas da base ResNet50.")
    else: # train_last_n_base_layers == 0
        print("Modelo: Todas as camadas da base ResNet50 estão congeladas (treinando apenas o cabeçalho).")
    # Adiciona as camadas de topo (cabeçalho de classificação)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x) # Saída binária

    model = Model(inputs=base_model.input, outputs=output)

    # Compilação será feita na função de treinamento para permitir diferentes LRs por fase
    return model


# In[77]:


# --- 4. Função para Criar Callbacks do TensorBoard ---
def create_tensorboard_callback(log_dir_base, sub_dir_name):
    """Cria e retorna um callback do TensorBoard."""
    log_dir = os.path.join(log_dir_base, sub_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"Logs do TensorBoard serão salvos em: {os.path.abspath(log_dir)}")
    return tensorboard_callback


# In[78]:


# --- 5. Função para Treinar uma Fase do Modelo ---
def train_model_phase(model, train_ds, val_ds, phase_name, epochs, learning_rate,
                      checkpoint_path, history_log_path, tensorboard_log_subdir, patience):
    """
    Executa uma fase de treinamento para o modelo.
    """
    print(f"\n--- Iniciando {phase_name} (LR: {learning_rate}, Épocas: {epochs}) ---")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                 save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                   restore_best_weights=True, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger(history_log_path, append=False)
    tensorboard_cb = create_tensorboard_callback(LOGS_SAVE_DIR, tensorboard_log_subdir)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping, csv_logger, tensorboard_cb],
        verbose=1
    )
    print(f"--- {phase_name} concluída ---")
    return model


# In[79]:


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    print("Iniciando o script de treinamento de ResNet50 para solda (imagens únicas)...")

    # 1. Carregar caminhos das imagens e rótulos
    print("Carregando caminhos das imagens e rótulos...")
    all_image_paths, all_labels = load_image_paths_and_labels_single_image(DATA_DIR)
    print(f"Total de imagens encontradas: {len(all_image_paths)}")

    if not all_image_paths:
        print(f"❌ Nenhuma imagem encontrada em '{DATA_DIR}'. Certifique-se de que as pastas 'boa' e 'ruim' existem e contêm imagens.")
        exit()

    # 2. Dividir os dados em Treino / Validação / Teste
    print("Dividindo os dados em Treino / Validação / Teste...")
    # Primeiro, separamos o Teste
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=TEST_RATIO, random_state=GLOBAL_RANDOM_STATE,
        stratify=all_labels
    )
    # Segundo, separamos Treino e Validação do restante
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), # Proporção de val no conjunto train_val
        random_state=GLOBAL_RANDOM_STATE,
        stratify=train_val_labels
    )

    print(f"Imagens de Treino: {len(train_paths)}")
    print(f"Imagens de Validação: {len(val_paths)}")
    print(f"Imagens de Teste: {len(test_paths)}")

    # 3. Criar os tf.data.Dataset
    print("Criando pipelines de dados (tf.data.Dataset)...")
    # Conjunto de Treino (com aumento de dados)
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(lambda p, l: preprocess_and_augment_single_image(p, l, augment=True),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Conjunto de Validação (sem aumento de dados)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(lambda p, l: preprocess_and_augment_single_image(p, l, augment=False),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Conjunto de Teste (sem aumento de dados)
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(lambda p, l: preprocess_and_augment_single_image(p, l, augment=False),
                        num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    # 4. Treinamento - FASE 1: Treinar apenas o cabeçalho de classificação
    print("\n--- INICIANDO FASE 1 DE TREINAMENTO (Treinar apenas o cabeçalho) ---")
    model_phase1 = create_resnet_model_for_single_images(train_last_n_base_layers=0) # Congela toda a base
    trained_model_phase1 = train_model_phase(
        model=model_phase1,
        train_ds=train_ds,
        val_ds=val_ds,
        phase_name="Fase 1",
        epochs=EPOCHS_PHASE1,
        learning_rate=LEARNING_RATE_PH1,
        checkpoint_path=os.path.join(MODEL_SAVE_DIR, 'model_phase1_best.h5'),
        history_log_path=os.path.join(RESULTS_SAVE_DIR, 'history_phase1.csv'),
        tensorboard_log_subdir='phase_1',
        patience=ES_PATIENCE
    )

    # Carregar o melhor modelo da Fase 1 para a Fase 2 (garantir que é o salvo pelo checkpoint)
    try:
        model_for_phase2 = load_model(os.path.join(MODEL_SAVE_DIR, 'model_phase1_best.h5'))
        print("✅ Melhor modelo da Fase 1 carregado para a Fase 2.")
    except Exception as e:
        print(f"❌ ERRO: Não foi possível carregar o melhor modelo da Fase 1. Erro: {e}")
        print("A Fase 2 não poderá iniciar. Encerrando o script.")
        exit()


    # 5. Treinamento - FASE 2: Fine-tuning das últimas camadas da ResNet50
    print("\n--- INICIANDO FASE 2 DE TREINAMENTO (Fine-tuning das últimas camadas) ---")
    # Reconfigura o modelo para descongelar as últimas N camadas
    if hasattr(model_for_phase2.layers[0], 'layers'): # Verifica se a primeira camada é a base ResNet50
        base_model_ph2 = model_for_phase2.layers[0]
        base_model_ph2.trainable = True # Garante que a base está treinável antes de congelar partes
        num_base_layers = len(base_model_ph2.layers)
        start_unfreeze_idx = max(0, num_base_layers - UNFREEZE_LAST_N_LAYERS_PH2)
        for layer in base_model_ph2.layers[start_unfreeze_idx:]:
            layer.trainable = True
        print(f"Modelo: Descongeladas as últimas {UNFREEZE_LAST_N_LAYERS_PH2} camadas da base ResNet50 para Fase 2.")
    else:
        print("AVISO: A primeira camada do modelo não parece ser a base ResNet50. Fine-tuning da Fase 2 pode não funcionar como esperado.")

    final_trained_model = train_model_phase(
        model=model_for_phase2, # Continua treinando o modelo da Fase 1
        train_ds=train_ds,
        val_ds=val_ds,
        phase_name="Fase 2",
        epochs=EPOCHS_PHASE2,
        learning_rate=LEARNING_RATE_PH2,
        checkpoint_path=os.path.join(MODEL_SAVE_DIR, 'model_final_best.h5'), # Salva o modelo final
        history_log_path=os.path.join(RESULTS_SAVE_DIR, 'history_phase2.csv'),
        tensorboard_log_subdir='phase_2',
        patience=ES_PATIENCE
    )


    # 6. Avaliação Final no Conjunto de Teste
    print("\n--- AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---")
    test_loss, test_accuracy = final_trained_model.evaluate(test_ds, verbose=1)
    print(f"🎉 Resultado Final no Conjunto de Teste:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.4f}")

    # Mapear a acurácia para as classes
    # A acurácia é uma métrica geral, não específica de classe.
    # Para ver a performance por classe, precisaríamos de metrics.Precision, metrics.Recall ou um classification_report.
    # Aqui, apenas indicamos a acurácia geral e quais classes o modelo está classificando.
    print(f"   O modelo foi treinado para classificar entre: {CLASS_NAMES[0]} e {CLASS_NAMES[1]}")


    # Salvar resultados finais em um arquivo
    with open(os.path.join(RESULTS_SAVE_DIR, 'final_test_results.txt'), 'w') as f:
        f.write("--- Resultados Finais do Treinamento ---\n")
        f.write(f"Total de imagens: {len(all_image_paths)}\n")
        f.write(f"Imagens de Treino: {len(train_paths)}\n")
        f.write(f"Imagens de Validação: {len(val_paths)}\n")
        f.write(f"Imagens de Teste: {len(test_paths)}\n\n")
        f.write(f"Loss no Teste: {test_loss:.4f}\n")
        f.write(f"Accuracy no Teste: {test_accuracy:.4f}\n")
        f.write(f"Classes classificadas: {CLASS_NAMES[0]} (0) e {CLASS_NAMES[1]} (1)\n")
    print(f"Resultados finais salvos em: {os.path.abspath(os.path.join(RESULTS_SAVE_DIR, 'final_test_results.txt'))}")

    print("\nScript de treinamento concluído com sucesso!")

