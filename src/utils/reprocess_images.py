# src/utils/reprocess_images.py

import os
from PIL import Image
import shutil

# --- Configurações ---
# Diretório principal onde suas imagens estão (ex: image/treinamento)
DATA_DIR = 'C:\\cod\\python\\AI\\image\\treinamento'

# Diretório para onde os arquivos ORIGINAIS das imagens reprocessadas serão movidos
# Isso serve como backup e para garantir que você não reprocessa o mesmo arquivo infinitamente
ORIGINAL_BACKUP_DIR = 'C:\\cod\\python\\AI\\image\\reprocessadas_originais_backup'

# Extensões de imagem para procurar
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Opcional: Lista de padrões de nome de arquivo a serem especificamente reprocessados.
# Se você sabe que APENAS os arquivos 'SNAP' são problemáticos, pode focar neles.
# Caso contrário, o script tentará reprocessar TODOS os JPEGs/PNGs.
SPECIFIC_PATTERNS_TO_REPROCESS = ['SNAP'] # Exemplo: apenas arquivos que contêm 'SNAP' no nome


# --- Função para reprocessar e salvar a imagem ---
def reprocess_and_save_image(filepath, save_as_format='JPEG', quality=95):
    """
    Abre uma imagem e a salva novamente em um formato padrão,
    forçando uma nova codificação.

    Args:
        filepath (str): Caminho para o arquivo de imagem.
        save_as_format (str): Formato para salvar ('JPEG', 'PNG').
        quality (int): Qualidade para JPEG (0-100).
    Returns:
        bool: True se reprocessado com sucesso, False caso contrário.
    """
    try:
        with Image.open(filepath) as img:
            # Converte para RGB para compatibilidade universal ao salvar
            # e para garantir 3 canais se a imagem original for, por exemplo, CMYK ou L
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Salva a imagem novamente, forçando a codificação
            if save_as_format.upper() == 'JPEG':
                # Sobrescreve o arquivo original
                img.save(filepath, format='JPEG', quality=quality)
            elif save_as_format.upper() == 'PNG':
                # Sobrescreve o arquivo original
                img.save(filepath, format='PNG')
            else:
                print(f"AVISO: Formato '{save_as_format}' não suportado para reprocessamento.")
                return False
        return True
    except Exception as e:
        print(f"ERRO ao reprocessar '{filepath}': {e}")
        return False


# --- Bloco Principal ---
if __name__ == "__main__":
    os.makedirs(ORIGINAL_BACKUP_DIR, exist_ok=True)
    
    reprocessed_count = 0
    skipped_count = 0
    total_files_checked = 0

    print(f"Iniciando reprocessamento de imagens em: {os.path.abspath(DATA_DIR)}")
    print(f"Backups de originais serão movidos para: {os.path.abspath(ORIGINAL_BACKUP_DIR)}")
    if SPECIFIC_PATTERNS_TO_REPROCESS: # Linha correta
        print(f"Focando em arquivos com padrões: {SPECIFIC_PATTERNS_TO_REPROCESS}") # <--- CORREÇÃO AQUI: REPROCESS
    else:
        print("Reprocessando TODOS os arquivos de imagem encontrados.")

    for root, _, files in os.walk(DATA_DIR):
        for file_name in files:
            total_files_checked += 1
            filepath = os.path.join(root, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension not in IMAGE_EXTENSIONS:
                skipped_count += 1
                # print(f"Ignorando arquivo '{file_name}': extensão não é de imagem.")
                continue

            # Se há padrões específicos, verifica se o nome do arquivo corresponde
            if SPECIFIC_PATTERNS_TO_REPROCESS:
                if not any(pattern.lower() in file_name.lower() for pattern in SPECIFIC_PATTERNS_TO_REPROCESS):
                    skipped_count += 1
                    continue # Ignora arquivos que não correspondem aos padrões
            
            # Criar caminho para o backup original
            relative_path = os.path.relpath(filepath, DATA_DIR)
            backup_filepath = os.path.join(ORIGINAL_BACKUP_DIR, relative_path)
            os.makedirs(os.path.dirname(backup_filepath), exist_ok=True)

            print(f"Processando: {filepath}")

            # Mover original para backup ANTES de reprocessar para evitar perda se falhar
            try:
                shutil.move(filepath, backup_filepath)
            except Exception as e:
                print(f"ERRO: Não foi possível mover original para backup '{filepath}': {e}. Pulando este arquivo.")
                skipped_count += 1
                continue
            
            # Tentar reprocessar o arquivo do local de backup e salvá-lo no local original
            success = reprocess_and_save_image(backup_filepath, save_as_format='JPEG', quality=95) # Salvar como JPEG de volta no caminho original

            if success:
                print(f"   -> Reprocessado com sucesso: {file_name}")
                # Copiar o arquivo reprocessado do backup de volta para o local original
                try:
                    shutil.copy(backup_filepath, filepath)
                except Exception as e:
                    print(f"ERRO: Não foi possível copiar reprocessado para original '{filepath}': {e}. Reprocessamento falhou para este.")
                    reprocessed_count -= 1 # Desconta o reprocessamento
                    # O arquivo original está no backup_filepath
            else:
                # Se o reprocessamento falhou (mesmo abrindo), o original está no backup.
                print(f"   -> Falha ao reprocessar: {file_name}. Original mantido em backup.")
                
            reprocessed_count += 1


    print("\n--- Reprocessamento Concluído ---")
    print(f"Total de arquivos verificados: {total_files_checked}")
    print(f"Arquivos reprocessados e movidos para o local original: {reprocessed_count}")
    print(f"Arquivos ignorados/com erro de mover: {skipped_count}")
    print(f"Backups de arquivos originais em: {os.path.abspath(ORIGINAL_BACKUP_DIR)}")
    print("Por favor, verifique a pasta de backup para garantir que os originais estão lá.")