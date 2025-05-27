# src/__init__.py
# (vazio ou só configuração global do projeto)

# src/core/__init__.py
from .modelo import criar_modelo
from .dataset import ParImageGenerator
from .processar_imagens import combinar_imagens
from .augmentar_imagens import combinar_com_augmentacao
from .gerar_graficos_cv import plot_metric as graficos
