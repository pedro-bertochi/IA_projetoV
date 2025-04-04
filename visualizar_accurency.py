#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


# Caminho do arquivo onde o script principal salva os resultados
resultado_path = "classificacao_resultados.txt"


# In[7]:


# Verifica se o arquivo existe
if not os.path.exists(resultado_path):
    print(f"Arquivo {resultado_path} não encontrado! Execute primeiro a classificação.")
    exit()


# In[8]:


# Carregar os dados no Pandas
df = pd.read_csv(resultado_path, names=["Imagem", "Certeza", "Correta"])


# In[9]:


df


# In[10]:


# Converter para os tipos corretos
df["Certeza"] = df["Certeza"].astype(float)
df["Correta"] = df["Correta"].astype(int)


# In[11]:


# Criar gráfico de dispersão com Seaborn
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="Imagem", y="Certeza", hue="Correta", palette={0: "red", 1: "green"})


# In[12]:


plt.axhline(y=50, color="gray", linestyle="--", label="Limite 50%")
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Imagens")
plt.ylabel("Certeza (%)")
plt.title("Dispersão da Precisão da IA")
plt.legend(["Correção Manual", "Classificação Certa"])
plt.grid(True, linestyle="--", alpha=0.5)


# In[13]:


# Exibir gráfico
plt.tight_layout()
plt.show()

