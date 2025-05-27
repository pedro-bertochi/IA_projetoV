import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .core import criar_modelo, ParImageGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

tf.config.run_functions_eagerly(True)

os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# Caminho para salvar o modelo principal
caminho_modelo = '../model/modelo_solda_resnet50.h5'

# Carregar ou criar o modelo
if os.path.exists(caminho_modelo):
    print("Carregando modelo salvo...")
    modelo = tf.keras.models.load_model(caminho_modelo)
else:
    print("Criando novo modelo...")
    modelo = criar_modelo()

# Recompilar com otimizador novo
modelo.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='binary_crossentropy',
                metrics=['accuracy'])

# ============================================
# 🔁 Carregar dados e dividir entre treino/validação
# ============================================
gen_temporario = ParImageGenerator('../image/treinamento', batch_size=1, augmentacao=False)
dados = list(zip(gen_temporario.imagens, gen_temporario.labels))

train_data, val_data = train_test_split(dados, test_size=0.2, shuffle=True)

# Criar geradores com dados separados
train_gen = ParImageGenerator(train_data, batch_size=8, augmentacao=True)
val_gen = ParImageGenerator(val_data, batch_size=8, augmentacao=False)

# ============================================
# 📦 Callbacks: checkpoint + logger
# ============================================
checkpoint = ModelCheckpoint(
    filepath='../model/melhor_modelo.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

logger = CSVLogger('../results/historico_treinamento.csv', append=True)

# ============================================
# ▶️ Treinamento
# ============================================
modelo.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    steps_per_epoch=len(train_gen),
    callbacks=[checkpoint, logger],
    verbose=1
)

# Avaliação final no conjunto de validação
loss, accuracy = modelo.evaluate(val_gen, verbose=1)
print(f"[AVALIAÇÃO FINAL] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Salvar o modelo final
print("Salvando modelo final...")
modelo.save(caminho_modelo)
