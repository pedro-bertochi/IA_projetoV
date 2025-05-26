#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from core import ParImageGenerator, criar_modelo

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.model_selection import KFold
import numpy as np


# In[ ]:


# Caminhos
model_dir = '../model'
results_dir = '../results'
image_dir = '../image/treinamento'
log_txt_path = os.path.join(results_dir, "fine_tuning_summary.txt")

os.makedirs(results_dir, exist_ok=True)


# In[3]:


# Carrega todos os dados para gerar os validadores novamente
gen_temporario = ParImageGenerator(image_dir, batch_size=1, augmentacao=False)
dados = list(zip(gen_temporario.imagens, gen_temporario.labels))


# In[4]:


# Define os folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[ ]:


fold = 1
accuracies = []
losses = []
with open(log_txt_path, 'w') as log_file:

    for train_idx, val_idx in kf.split(dados):
        print(f"\n🔧 Fine-tuning do modelo do Fold {fold}")
        log_file.write(f"\n[FOLD {fold}]\n")

        # Dados
        val_data = [dados[i] for i in val_idx]
        val_gen = ParImageGenerator(dados=val_data, batch_size=8, augmentacao=False)

        # Carrega modelo salvo anteriormente
        model_path = f"{model_dir}/melhor_modelo_fold{fold}.h5"
        model = load_model(model_path)

        # Descongela camadas
        if hasattr(model.layers[0], 'layers'):
            base_model = model.layers[0]
            base_model.trainable = True
            for layer in base_model.layers[:100]:
                layer.trainable = False

        # Compila
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        checkpoint = ModelCheckpoint(
            filepath=f"{model_dir}/fine_tuned_fold{fold}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        logger = CSVLogger(f'{results_dir}/fine_tuning_log_fold{fold}.csv', append=False)

        # Fine-tuning
        model.fit(
            val_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=[checkpoint, logger],
            verbose=1
        )

        # Avaliação
        loss, acc = model.evaluate(val_gen, verbose=0)
        log_file.write(f"Accuracy: {acc:.4f}, Loss: {loss:.4f}\n")

        accuracies.append(acc)
        losses.append(loss)

        fold += 1

    # Média dos resultados
    media_acc = np.mean(accuracies)
    media_loss = np.mean(losses)

    log_file.write("\n=== RESULTADO FINAL ===\n")
    log_file.write(f"Média Accuracy: {media_acc:.4f}\n")
    log_file.write(f"Média Loss: {media_loss:.4f}\n")

print("\n✅ Fine-tuning finalizado. Logs gravados com sucesso!")

