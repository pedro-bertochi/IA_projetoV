# 🛠️ Projeto de Classificação de Soldas com IA

Este projeto classifica imagens de soldas como "boas" ou "ruins" utilizando um modelo de deep learning com Flask no backend, e um frontend web simples para upload e visualização do relatório em PDF.

---

## ✅ Pré-requisitos

- Python 3.8+
- pip
- Navegador web moderno
- Ambiente virtual recomendado (venv ou conda)

---

## 📦 Instalação

```
git clone <url-do-repositorio>
cd IA_projetoV

```

## ✅ Como rodar o projeto

### 1. Criar e ativar o ambiente virtual

```
python -m venv venv
```

Ativar:

```
Windows:

venv\Scripts\activate
```

```
Linux/MacOS:

source venv/bin/activate
```

2. Instalar as dependências

```
pip install -r requirements.txt
```

3. Rodar o Backend (Flask)
   Certifique-se de que você está na raíz do projeto.

```
python app.py
```

O Flask iniciará em: `http://localhost:5000`c.
O relatório PDF será gerado automaticamente na pasta: `static/reports/`.

4. Rodar o Frontend (Web)
   Opção 1 — abrir direto:
   Abra o arquivo frontend/index.html no navegador.
   ⚠️ Pode ter bloqueios de segurança (CORS).

Opção 2 — rodar um servidor local (recomendado):

No diretório frontend/:

```
python -m http.server 8000
```

O frontend estará em: `http://localhost:8000/index.html`

🖼️ Como gerar o relatório PDF
Acesse o frontend no navegador.

Selecione pelo menos dois arquivos de imagem.

Clique em "Enviar".

O backend irá:

Classificar as imagens.

Salvar as imagens nas pastas corretas (boa ou ruim).

Gerar um relatório PDF com:

Imagens.

Classificação.

Confiança.

O link de download do PDF aparecerá automaticamente na página.

✅ Dependências principais
```
Flask

Flask-CORS

TensorFlow

NumPy

Pillow

reportlab
```