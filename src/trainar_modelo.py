#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from core import criar_modelo, ParImageGenerator, create_tensorboard_callback
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping


# In[ ]:


# Caminhos
os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../logs', exist_ok=True)


# In[ ]:


# ============================
# 🔁 Carregar todos os dados
# ============================
gen_temporario = ParImageGenerator('../image/treinamento', batch_size=1, augmentacao=False)
dados = list(zip(gen_temporario.imagens, gen_temporario.labels))


# In[ ]:


# ============================
# 🔁 K-Fold Cross-Validation
# ============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[ ]:


fold = 1
accuracies = []
losses = []

log_path_cv = '../results/cv_info.txt' # Renomeei para evitar confusão com logs de fase
with open(log_path_cv, 'w') as log_file_cv: # Renomeei a variável

    for train_idx, val_idx in kf.split(dados):
        log_file_cv.write(f"[Fold {fold}]\n")
        log_file_cv.write(f"Treino: {list(train_idx)}\n")
        log_file_cv.write(f"Validação: {list(val_idx)}\n\n")

        print(f"\n🌀 Treinando Fold {fold} - Fase 1 (Fine-tuning inicial)...")

        train_data = [dados[i] for i in train_idx]
        val_data = [dados[i] for i in val_idx]

        train_gen = ParImageGenerator(dados=train_data, batch_size=8, augmentacao=True)
        val_gen = ParImageGenerator(dados=val_data, batch_size=8, augmentacao=False)

        # --- FASE 1: Fine-tuning Inicial (todas as camadas treináveis) ---
        modelo = criar_modelo() # Cria o modelo com todas as camadas da base treináveis por padrão

        # Compila com learning rate para a Fase 1
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Seu LR atual, bom para 1a fase de fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks para a Fase 1
        checkpoint_ph1 = ModelCheckpoint(
            filepath=f'../model/modelo_ph1_fold{fold}.h5', # Salva o modelo da Fase 1
            monitor='val_loss', # Monitore val_loss para evitar overfitting
            save_best_only=True,
            verbose=1
        )
        logger_ph1 = CSVLogger(f'../results/historico_fase1_fold{fold}.csv', append=False)
        early_stopping_ph1 = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        tensorboard_callback_ph1 = create_tensorboard_callback(fold_name=f'fold_{fold}/fase_1') # Logs separados para fase 1

        print(f"--- Iniciando Fase 1 para Fold {fold} ---")
        modelo.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            steps_per_epoch=len(train_gen),
            callbacks=[checkpoint_ph1, logger_ph1, early_stopping_ph1, tensorboard_callback_ph1],
            verbose=1
        )
        print(f"--- Fase 1 para Fold {fold} concluída ---")


        # --- FASE 2: Fine-tuning Aprofundado (carrega melhor da Fase 1, ajusta camadas e LR) ---
        print(f"\n🔧 Treinando Fold {fold} - Fase 2 (Fine-tuning aprofundado)...")

        # Carrega o MELHOR modelo da Fase 1 para continuar o treinamento
        # É importante carregar o que foi salvo pelo checkpoint, pois o modelo 'modelo'
        # após o fit pode não ser o melhor restaurado se early_stopping tiver parado cedo.
        model_path_ph1 = f'../model/modelo_ph1_fold{fold}.h5'
        if not os.path.exists(model_path_ph1):
            print(f"AVISO: Modelo da Fase 1 para Fold {fold} não encontrado em {model_path_ph1}. Pulando Fase 2 para este fold.")
            continue # Pula para o próximo fold se o modelo da fase 1 não foi salvo

        modelo_ph2 = tf.keras.models.load_model(model_path_ph1)

        # Ajusta as camadas treináveis para a Fase 2 (congelando as 100 primeiras)
        if hasattr(modelo_ph2.layers[0], 'layers'): # Verifica se a primeira camada é o ResNet50
            base_model_ph2 = modelo_ph2.layers[0]
            base_model_ph2.trainable = True # Garante que a base está treinável antes de congelar partes
            for layer in base_model_ph2.layers[:100]: # Congela as primeiras 100 camadas
                layer.trainable = False
        else:
            print(f"AVISO: A primeira camada do modelo do Fold {fold} não parece ser a base ResNet50. Fine-tuning da Fase 2 pode não funcionar como esperado.")


        # Compila com learning rate para a Fase 2 (mais baixo)
        modelo_ph2.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Seu LR mais baixo para fine-tuning aprofundado
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks para a Fase 2
        checkpoint_ph2 = ModelCheckpoint(
            filepath=f'../model/final_tuned_fold{fold}.h5', # Salva o modelo final do fold após a Fase 2
            monitor='val_loss', # Mude para val_loss para salvar o menos overfitado
            save_best_only=True,
            verbose=1
        )
        logger_ph2 = CSVLogger(f'../results/historico_fase2_fold{fold}.csv', append=False)
        early_stopping_ph2 = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        tensorboard_callback_ph2 = create_tensorboard_callback(fold_name=f'fold_{fold}/fase_2') # Logs separados para fase 2

        print(f"--- Iniciando Fase 2 para Fold {fold} ---")
        modelo_ph2.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            steps_per_epoch=len(train_gen),
            callbacks=[checkpoint_ph2, logger_ph2, early_stopping_ph2, tensorboard_callback_ph2],
            verbose=1
        )
        print(f"--- Fase 2 para Fold {fold} concluída ---")

        # Avaliação final do modelo da Fase 2 para este fold
        loss, accuracy = modelo_ph2.evaluate(val_gen, verbose=1)
        print(f"📊 [Fold {fold}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        log_file_cv.write(f"Resultado Final Fold {fold}: Loss={loss:.4f}, Accuracy={accuracy:.4f}\n\n")

        accuracies.append(accuracy)
        losses.append(loss)
        fold += 1


# In[ ]:


# ============================
# 📈 Resultados finais
# ============================
media_acc = np.mean(accuracies)
media_loss = np.mean(losses)


# In[ ]:


print(f"\n✅ Cross-validation finalizada!")
print(f"📉 Média da Loss: {media_loss:.4f}")
print(f"✅ Média da Accuracy: {media_acc:.4f}")


# In[ ]:


# Salvar resumo no log geral de CV
with open(log_path_cv, 'a') as log_file_cv:
    log_file_cv.write("=== Resultado Final CV ===\n")
    log_file_cv.write(f"Média da Loss: {media_loss:.4f}\n")
    log_file_cv.write(f"Média da Accuracy: {media_acc:.4f}\n")

