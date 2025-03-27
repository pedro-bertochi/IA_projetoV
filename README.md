# Manual de Funcionamento do Sistema de Classificação de Soldas com Redes Neurais

## Introdução

Este projeto utiliza uma rede neural convolucional (CNN) para classificar imagens de soldas em duas categorias: **boa** ou **ruim**. O objetivo é automatizar a inspeção de soldas, determinando a qualidade com base em características visuais das imagens.

O sistema é dividido em dois componentes principais:
1. **Treinamento do Modelo**: Onde o modelo é treinado usando imagens de soldas boas e ruins.
2. **Classificação das Imagens**: Onde o modelo classifica novas imagens e move as imagens para a pasta correspondente.

---

## Estrutura das Pastas

A estrutura de pastas foi organizada da seguinte forma:
```
image/
│── treinamento/      # Para treinar o modelo
│   ├── boa/         # Soldas boas para treinamento
│   ├── ruim/        # Soldas ruins para treinamento
│
│── classificacao/    # Para a IA classificar
│   ├── novas/       # Onde as novas imagens chegam
│   ├── boa/         # A IA move para cá as soldas boas
│   ├── ruim/        # A IA move para cá as soldas ruins
│   ├── incertas/    # Se quiser revisar imagens duvidosas
│
│── validacao/        # Opcional - Para testar o modelo
│   ├── boa/
│   ├── ruim/
```
---

## Descrição do Processo

### 1. Treinamento do Modelo

- **Objetivo**: Treinar a rede neural para classificar imagens de soldas como "boa" ou "ruim".
- **Arquitetura**: O modelo utiliza uma CNN (Rede Neural Convolucional), composta por camadas `Conv2D` e `MaxPooling2D`, seguidas de camadas densas (`Dense`), terminando em uma camada de saída com duas unidades para a classificação (boa ou ruim).
  
- **Dados**: O treinamento utiliza as imagens localizadas em `image/treinamento/boa` e `image/treinamento/ruim`, que são carregadas através do gerador de dados do Keras, `ImageDataGenerator`. O modelo é treinado por 10 épocas (`epochs`), um parâmetro que define o número de vezes que o modelo passará por todos os dados de treinamento.

- **Como funciona**: 
  - A função `train_generator` carrega as imagens de treinamento e as prepara (como normalização).
  - O modelo é treinado para minimizar o erro na classificação utilizando a função de perda `sparse_categorical_crossentropy`.
  - Após o treinamento, o modelo é salvo como um arquivo `.h5` para ser usado posteriormente para classificar novas imagens.

### 2. Classificação de Imagens

- **Objetivo**: Classificar novas imagens de soldas usando o modelo treinado e movê-las para a pasta correta.
  
- **Funcionamento**:
  - O script carrega o modelo treinado (`modelo_solda.h5`).
  - As imagens a serem classificadas são armazenadas na pasta `image/classificacao/novas`.
  - O modelo faz a previsão sobre a imagem, classificando-a como "boa" ou "ruim".
  - O usuário é então questionado se a classificação está correta. Se estiver errada, a imagem é movida para a pasta de revisão (`incertas`) ou para a pasta correta.
  
---

## Detalhes Técnicos

### Rede Neural Convolucional (CNN)

A CNN utilizada é composta por várias camadas:
1. **Conv2D**: Camada convolucional que aplica filtros às imagens para detectar padrões.
2. **MaxPooling2D**: Reduz a dimensionalidade das imagens, mantendo os principais elementos.
3. **Flatten**: Transforma os dados 2D em um vetor 1D.
4. **Dense**: Camadas densas para realizar a classificação com base nas características extraídas.

### Funções Importantes no Código

- **`processar_imagem(img_path)`**: Função que processa as imagens antes de passá-las para o modelo. A imagem é redimensionada para 128x128 pixels, normalizada e expandida para ter a forma que o modelo espera.
  
- **`train_generator.flow_from_directory()`**: Esta função carrega as imagens das pastas de treinamento e as prepara para o treinamento, aplicando a normalização.
  
- **`model.predict()`**: A função de previsão do modelo, que retorna a classe predita para a imagem fornecida.

---

## Como Rodar o Código

1. **Treinamento**:
   - Coloque as imagens de treinamento nas pastas `image/treinamento/boa` e `image/treinamento/ruim`.
   - Execute o script de treinamento para gerar ou continuar o treinamento do modelo.

2. **Classificação**:
   - Coloque as imagens a serem classificadas na pasta `image/classificacao/novas`.
   - Execute o script de classificação. O modelo irá classificar as imagens e perguntar ao usuário se a classificação está correta.
   - As imagens serão movidas para as pastas apropriadas (`boa`, `ruim` ou `incertas`).

---

## Observações

- **Épocas (Epochs)**: Durante o treinamento, cada época é uma iteração completa sobre o conjunto de dados de treinamento. Um número maior de épocas pode melhorar a precisão, mas também aumenta o risco de overfitting.
  
- **Desempenho do Modelo**: O modelo pode ser aprimorado ao aumentar a quantidade de dados de treinamento ou ajustar os parâmetros da rede.

---

Este manual fornece uma explicação detalhada de como o sistema funciona e como os dados são processados. Acesse os scripts para treinar e classificar as imagens conforme necessário e continue explorando as possibilidades de melhoria do sistema! Boa sorte no seu estudo de redes neurais!