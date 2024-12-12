import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para a turbina e o wake
turbine_diameter = 130  # Diâmetro do rotor da turbina em metros
k_y = 0.0324555  # Constante de expansão do wake
CT = 8 / 9  # Coeficiente de empuxo do modelo fornecido
turbine_positions = [(100, 0), (750, 0)]  # Posições das turbinas (x, y)

# Criar uma grade para o gráfico
x = np.linspace(-2000, 2000, 1000)  # Ajuste da faixa do eixo x
y = np.linspace(-2000, 2000, 1000)  # Ajuste da faixa do eixo y
X, Y = np.meshgrid(x, y)

# Calcular a perda de velocidade e o déficit de velocidade com base no modelo de Bastankhah
def bastankhah_wake(x, y, x0, y0, turb_diam, k_y, CT):
    """Calcular o déficit de velocidade com base no modelo de Bastankhah."""
    d = x - x0  # Distância da turbina na direção a montante
    sigma_y = k_y * d + turb_diam / np.sqrt(8)  # Largura do wake

    sigma_y = np.where(d > 0, sigma_y, np.inf)  # Evitar divisão por zero

    exponent = -0.5 * ((y - y0) / sigma_y)**2  # Expoente gaussiano
    radical = 1 - CT / (8 * (sigma_y**2) / (turb_diam**2))  # Termo radical

    velocity_deficit = (1 - np.sqrt(radical)) * np.exp(exponent)  # Cálculo do déficit de velocidade

    return velocity_deficit

# Criar o campo de déficit de velocidade considerando ambas as turbinas
velocity_deficit = np.zeros_like(X)

for x_turbine, y_turbine in turbine_positions:
    velocity_deficit += bastankhah_wake(X, Y, x_turbine, y_turbine, turbine_diameter, k_y, CT)

# Plotando
plt.figure(figsize=(8, 6))

# Plotar o rotor da primeira turbina como uma linha vertical
plt.plot([turbine_positions[0][0], turbine_positions[0][0]],
         [turbine_positions[0][1] - turbine_diameter, turbine_positions[0][1] + turbine_diameter],
         'k-', lw=3, label='Turbine Rotor')
plt.scatter([turbine_positions[0][0]], [turbine_positions[0][1]], color='black', zorder=5)

# Plotar o rotor da segunda turbina
plt.plot([turbine_positions[1][0], turbine_positions[1][0]],
         [turbine_positions[1][1] - turbine_diameter, turbine_positions[1][1] + turbine_diameter],
         'k-', lw=3)
plt.scatter([turbine_positions[1][0]], [turbine_positions[1][1]], color='black', zorder=5)

# Plotar a região do wake
contour = plt.contourf(X, Y, velocity_deficit, levels=100, cmap='coolwarm', alpha=1)
cbar = plt.colorbar(contour, label='Velocity Deficit', orientation='vertical')
cbar.ax.set_ylabel('Velocity Deficit', rotation=270, labelpad=25, fontsize=20)
cbar.ax.tick_params(labelsize=16)

# Adicionar rótulos e título
#plt.title('Simplified Bastankhah Gaussian Wake', fontsize=22)
""" plt.xlabel('Distance Downwind (x)', fontsize=18)
plt.ylabel('Lateral Distance (y)', fontsize=18) """
plt.xticks([])
plt.yticks([])
plt.xlim(-200, 1000)
plt.ylim(-500, 500)
plt.grid()
plt.savefig("2_turb.png", dpi=500)
plt.show()
