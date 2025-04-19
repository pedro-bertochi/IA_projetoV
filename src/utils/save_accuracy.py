import os

def save_accuracy(caminho_arquivo, acuracia, imagem1, imagem2):
    try:
        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)  # Garante que o diretório exista
        with open(caminho_arquivo, 'a') as arquivo:
            arquivo.write(f"{imagem1}, {imagem2} -> Confiança: {acuracia:.4f}\n")  # Salvando as informações
    except Exception as e:
        print(f"Ocorreu um erro ao salvar a acurácia: {e}")


