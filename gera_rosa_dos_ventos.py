import matplotlib.pyplot as plt
import numpy as np

# Direções do vento em graus (de 0 a 360, padrão da rosa dos ventos)
direcoes = [0., 22.5, 45., 67.5,
            90., 112.5, 135., 157.5,
            180., 202.5, 225., 247.5,
            270., 292.5, 315., 337.5]

# Frequências ou intensidades do vento para cada direção (em % ou valores absolutos)
frequencias = [0.10, 0.25, 0.22, 0.15,
               0.08, 0.03, 0.02, 0.01,
               0.01, 0.02, 0.01, 0.01,
               0.01, 0.02, 0.01, 0.01]

# Converter as direções para radianos
direcoes_rad = np.radians(direcoes)

# Criar o gráfico polar
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Ajustar a direção do gráfico para a rosa dos ventos
ax.set_theta_zero_location("N")  # Norte (0°) no topo
ax.set_theta_direction(-1)       # Sentido horário

# Plotar as frequências como barras radiais
ax.bar(direcoes_rad, frequencias, width=0.4, bottom=0.0, color='blue', alpha=0.8)

# Definir título
ax.set_title("Distribuição de Frequência do Vento (Rosa dos Ventos)", va='bottom')

# Mostrar o gráfico
plt.show()
