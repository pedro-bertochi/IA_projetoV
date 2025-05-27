# src/core/callbacks.py

import os
import datetime
import tensorflow as tf

def create_tensorboard_callback(log_dir_base='../logs', fold_name=None):
    """
    Cria e retorna um callback do TensorBoard.

    Args:
        log_dir_base (str): O diretório base onde os logs do TensorBoard serão salvos.
                            Padrão para '../logs'.
        fold_name (str, optional): Um nome para identificar o fold atual (ex: 'fold_1').
                                   Se None, um timestamp será usado.

    Returns:
        tf.keras.callbacks.TensorBoard: Uma instância do callback do TensorBoard.
    """
    # 1. Garante que o diretório base exista primeiro
    # Converte o caminho base para absoluto para maior robustez
    abs_log_dir_base = os.path.abspath(log_dir_base)
    os.makedirs(abs_log_dir_base, exist_ok=True)


    if fold_name:
        log_dir = os.path.join(abs_log_dir_base, fold_name)
    else:
        log_dir = os.path.join(abs_log_dir_base, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 2. Agora, garante que o diretório específico do log exista
    os.makedirs(log_dir, exist_ok=True)


    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,      # Calcula histogramas para ativações e pesos a cada época
        write_graph=True,      # Escreve o grafo do modelo
        write_images=False,    # Não escreve imagens para visualização (pode ser lento)
        update_freq='epoch',   # Registra métricas a cada época
        profile_batch=0,       # Desabilita o profiling
        embeddings_freq=0,     # Desabilita o registro de embeddings
        embeddings_metadata=None
    )
    print(f"✅ Logs do TensorBoard serão salvos em: {os.path.abspath(log_dir)}")
    return tensorboard_callback