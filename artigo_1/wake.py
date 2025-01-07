import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para a turbina e o wake
turbine_diameter = 130  # Diâmetro do rotor da turbina em metros
k_y = 0.0324555  # Constante de expansão do wake
CT = 8 / 9  # Coeficiente de empuxo do modelo fornecido

x = [13.289264305759662, 763.7889178071316, 156.38040516032967, -703.9490033847319, -882.2421223406626, -115.77028599778183, 1273.1737463735944, 1095.8736372852568, 367.67779473517135, -601.2658431619044, -1122.130757030282, -1274.4119034277846, -1019.8196133713725, -249.15322637800398, 610.5975817167455, 947.0282013181233]
y = [-112.47778233099655, 64.11459801821951, 503.5457912143951, 362.2043045232052, -379.534161329817, -586.4727971428451, -262.7329662319257, 699.328943418747, 1228.7965043219301, 1025.0326317353088, 656.3707520347364, 226.26137689524182, -774.3132518261166, -1275.9007288129555, -1106.6553928729102, -858.3489017272991]

# Organizando os pontos em uma lista de tuplas
turbine_positions = list(zip(x, y))

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
plt.figure(figsize=(10, 8))

# Plotar a região do wake
contour = plt.contourf(X, Y, velocity_deficit, levels=100, cmap='coolwarm', alpha=1)
cbar = plt.colorbar(contour, label='Velocity Deficit', orientation='vertical')
cbar.ax.set_ylabel('Velocity Deficit', rotation=270, labelpad=25, fontsize=22)  # Aumentar o tamanho da fonte
cbar.ax.tick_params(labelsize=18)

# Adicionar rótulos e título com fonte maior
#plt.title("Simplified Bastankhah's Gaussian Wake", fontsize=26)
#plt.xlabel('Distance Downwind (x)', fontsize=24)
#plt.ylabel('Lateral Distance (y)', fontsize=24)

plt.grid()

# Aumentar o tamanho da fonte dos ticks dos eixos
plt.xticks([])
plt.yticks([])

plt.savefig("wake.png", dpi=500, bbox_inches='tight')

plt.xlim(-2000, 2000)
plt.ylim(-2000, 2000)

plt.show()
