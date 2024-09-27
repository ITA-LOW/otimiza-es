import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Definindo os dados
data = {
    'cxpb': [0.7, 0.8, 0.85, 0.85, 0.8, 0.8, 0.5, 0.8, 0.8, 0.7, 0.8, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    'mutpb': [0.3, 0.4, 0.45, 0.45, 0.4, 0.4, 0.45, 0.4, 0.4, 0.3, 0.4, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
    'pop': [100, 100, 100, 250, 250, 250, 250, 250, 250, 250, 500, 250, 300, 500, 400, 300, 350, 300],
    'torneio': [2, 2, 2, 2, 4, 5, 6, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
    'alpha': [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.75, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'gen': [1000, 1000, 1000, 1500, 1500, 2500, 1500, 2000, 1500, 1500, 3500, 3500, 500, 500, 500, 3000, 2500, 1000],
    'indpb': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.35, 0.15, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    'sigma': [50, 50, 50, 50, 50, 50, 50, 10, 150, 150, 50, 100, 100, 100, 100, 100, 100, 100],
    'AEP': [411815, 411936, 410619, 411012, 412294, 410555, 400872, 407065, 409235, 404785, 410847, 412741, 415203, 411113, 412403, 415289, 410840, 408632]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Calculando a correlação
correlation_matrix = df.corr()

# Criando o heatmap de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

# Título e exibição
plt.title('Mapa de Correlação entre Parâmetros e AEP', fontsize=16)
plt.savefig("Mapa de correlação", dpi=300)

plt.show()
