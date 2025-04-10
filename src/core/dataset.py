import os
from src.processar_imagens import combinar_imagens
import numpy as np
from tensorflow.keras.utils import Sequence

class ParImageGenerator(Sequence):
    def __init__(self, pasta, batch_size=32):
        self.imagens = []
        self.labels = []
        self.batch_size = batch_size

        for classe in ['boa', 'ruim']:
            caminho = os.path.join(pasta, classe)
            arquivos = sorted(os.listdir(caminho))
            pares = [(arquivos[i], arquivos[i + 1]) for i in range(0, len(arquivos), 2)]
            for par in pares:
                self.imagens.append((
                    os.path.join(caminho, par[0]),
                    os.path.join(caminho, par[1])
                ))
                self.labels.append(1 if classe == 'boa' else 0)

    def __len__(self):
        return len(self.imagens) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.imagens[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        imagens_processadas = [combinar_imagens(p1, p2) for p1, p2 in batch_x]
        return np.array(imagens_processadas) / 255.0, np.array(batch_y)
