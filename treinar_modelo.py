from src.core import criar_modelo
from src.data import ParImageGenerator

modelo = criar_modelo()
train_gen = ParImageGenerator('../image/treino', batch_size=8)
modelo.fit(train_gen, epochs=10)
modelo.save('../model/modelo_solda_resnet50.h5')