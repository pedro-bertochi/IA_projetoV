import os
import random
import numpy as np
from keras.utils import Sequence
from .processar_imagens import combinar_imagens
from .augmentar_imagens import combinar_com_augmentacao

class ParImageGenerator(Sequence):
    def __init__(self, pasta=None, dados=None, batch_size=32, augmentacao=True):
        self.batch_size = batch_size
        self.augmentacao = augmentacao
        self.imagens = []
        self.labels = []

        # Se dados já vierem prontos (usado para validação)
        if dados is not None:
            self.imagens, self.labels = zip(*dados)
        # Caso contrário, carregar da pasta
        elif pasta:
            for classe in ['boa', 'ruim']:
                caminho = os.path.join(pasta, classe)
                arquivos = sorted(os.listdir(caminho))
                if len(arquivos) % 2 != 0:
                    print(f"[AVISO] Número ímpar de imagens na pasta {caminho}. Ignorando a última.")
                    arquivos = arquivos[:-1]  # Remove a última imagem

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
        if self.augmentacao:
            imagens_processadas = [combinar_com_augmentacao(p1, p2) for p1, p2 in batch_x]
        else:
            imagens_processadas = [combinar_imagens(p1, p2) for p1, p2 in batch_x]
        imgs = np.array(imagens_processadas)
        return imgs / 255.0, np.array(batch_y)

    def on_epoch_end(self):
        zipped = list(zip(self.imagens, self.labels))
        random.shuffle(zipped)
        self.imagens, self.labels = zip(*zipped)

    def get_all_data(self):
        """Retorna uma lista de tuplas: [((p1, p2), label), ...]"""
        return list(zip(self.imagens, self.labels))
