import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lendo os dados do arquivo CSV
df = pd.read_csv('parameter_tests/hyper.csv')

# Calculando a correlação
correlation_matrix = df.corr()

# Criando o heatmap de correlação
plt.figure(figsize=(10, 8))

# Aumentando o tamanho das anotações e rótulos
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f',
                      annot_kws={"size": 14},  # Tamanho da fonte das anotações
                      cbar_kws={'label': 'Correlation'})  # Rótulo da barra de cores

# Ajustando o rótulo da barra de cores (corrigido para 270 graus)
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation', rotation=270, fontsize=16, labelpad=25)
colorbar.ax.tick_params(labelsize=16)
# Aumentando o tamanho da fonte dos rótulos dos eixos
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Título e exibição
plt.title('Correlation between parameters and AEP', fontsize=18)  # Tamanho do título
plt.savefig("Mapa_de_correlacao.png", dpi=300)

plt.show()
