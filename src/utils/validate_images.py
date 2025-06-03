# src/utils/validate_images.py

import os
import shutil
from PIL import Image # Pillow library
# import imghdr # Para verificar o tipo de arquivo de imagem

# --- Configurações ---
# Diretório principal onde suas imagens estão (ex: image/treinamento)
DATA_DIR = 'C:\\cod\\python\\AI\\image\\treinamento'

# Diretório para onde as imagens inválidas serão movidas
QUARANTINE_DIR = 'C:\\cod\\python\\AI\\image\\quarentena_imagens_invalidas'

# Tipos de arquivos de imagem esperados (minúsculas)
EXPECTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# --- Funções ---
def is_valid_image(filepath):
    """
    Verifica se um arquivo é uma imagem válida e de um tipo esperado,
    utilizando a biblioteca Pillow, com validação MENOS RESTRITA.
    Retorna True se for válido, False caso contrário, e uma mensagem.
    """
    try:
        # 1. Verifica se a extensão é esperada
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension not in EXPECTED_EXTENSIONS:
            return False, f"Extensão inesperada: {file_extension}. Esperado: {EXPECTED_EXTENSIONS}"

        # 2. Tenta abrir com PIL
        # Apenas abre o arquivo. Se esta linha não der erro, o Pillow conseguiu ler o cabeçalho.
        with Image.open(filepath) as img:
            # img.verify() # <--- ESTA LINHA FOI COMENTADA/REMOVIDA para relaxar a validação
            # img.load()   # Opcional: Descomente esta linha APENAS SE AINDA HOUVER PROBLEMAS
            #              # depois de comentar img.verify(). img.load() força a leitura de pixels
            #              # e pode revelar problemas mais profundos (mas é mais lento).
            pass # Mantém um pass se img.verify() for comentado/removido

        return True, "Imagem válida (aberta com Pillow sem verificação estrita)"
    except (IOError, SyntaxError, Image.UnidentifiedImageError, OSError, ValueError) as e:
        # Captura erros comuns de imagem corrompida, formato inválido, ou erro de IO/descompressão.
        return False, f"Erro ao abrir imagem com Pillow: {e}"
    except Exception as e:
        # Captura outros erros inesperados.
        return False, f"Erro inesperado durante validação: {e}"


def validate_and_quarantine_images(data_dir, quarantine_dir):
    """
    Percorre todas as imagens em data_dir, valida-as e move as inválidas
    para um diretório de quarentena.
    """
    os.makedirs(quarantine_dir, exist_ok=True)
    
    invalid_images_count = 0
    total_images_checked = 0
    
    print(f"Iniciando validação de imagens em: {os.path.abspath(data_dir)}")
    print(f"Imagens inválidas serão movidas para: {os.path.abspath(quarantine_dir)}")

    for root, _, files in os.walk(data_dir):
        for file_name in files:
            total_images_checked += 1
            filepath = os.path.join(root, file_name)
            
            is_valid, message = is_valid_image(filepath)
            
            if not is_valid:
                invalid_images_count += 1
                print(f"🚫 Imagem INVÁLIDA encontrada: {filepath} - Razão: {message}")
                
                # Mover para quarentena
                relative_path = os.path.relpath(filepath, data_dir)
                quarantine_filepath = os.path.join(quarantine_dir, relative_path)
                
                # Criar diretório de destino na quarentena se não existir
                os.makedirs(os.path.dirname(quarantine_filepath), exist_ok=True)
                
                try:
                    shutil.move(filepath, quarantine_filepath)
                    print(f"   -> Movida para quarentena: {quarantine_filepath}")
                except Exception as e:
                    print(f"   -> ERRO ao mover para quarentena {filepath}: {e}")

    print("\n--- Validação Concluída ---")
    print(f"Total de imagens verificadas: {total_images_checked}")
    print(f"Imagens inválidas encontradas e movidas para quarentena: {invalid_images_count}")
    if invalid_images_count > 0:
        print("Por favor, inspecione a pasta de quarentena para verificar as imagens problemáticas.")
    else:
        print("✅ Todas as imagens parecem válidas e foram mantidas no local.")

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    validate_and_quarantine_images(DATA_DIR, QUARANTINE_DIR)