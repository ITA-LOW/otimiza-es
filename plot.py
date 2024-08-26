import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(individual, generation_number, num_turbines=16):
    # Verifique a estrutura do indivíduo
    if len(individual) != num_turbines * 2:
        raise ValueError(f'O número de coordenadas ({len(individual)}) não corresponde ao número esperado de turbinas ({num_turbines * 2}).')

    # Extrai as coordenadas x e y
    x_coords = individual[:num_turbines]
    y_coords = individual[num_turbines:]

    # Configura a figura
    fig, ax = plt.subplots()

    # Plota as coordenadas das turbinas
    ax.plot(x_coords, y_coords, 'bo', label='Turbine Locations', markersize=8)

    # Adiciona um círculo representando a fronteira
    circle = plt.Circle((0, 0), 1300, color='r', fill=False, linestyle='--', label='Boundary', linewidth=2)
    ax.add_artist(circle)

    # Configura os limites e rótulos
    ax.set_xlim(-1400, 1400)
    ax.set_ylim(-1400, 1400)
    ax.set_aspect('equal', 'box')  # Mantém a proporção igual entre os eixos X e Y
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(f'Generation {generation_number}')

    # Adiciona uma grade e uma legenda
    ax.grid(True, linestyle='--', alpha=0.7)

    # Salva o gráfico em um arquivo
    plt.savefig(f'wind_farm_solution_gen_{generation_number}.png', dpi=300)
    plt.close()

def create_animation(generations, num_turbines, radius):
    fig, ax = plt.subplots()
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal', 'box')

    def update(frame):
        ax.clear()
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect('equal', 'box')
        best_individual = generations[frame]
        coords = np.array(best_individual).reshape((num_turbines, 2))
        ax.plot(coords[:, 0], coords[:, 1], 'bo', markersize=5)
        circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--', linewidth=2)
        ax.add_artist(circle)
        ax.set_title(f"Generation {frame + 1}")
        ax.grid(True, linestyle='--', alpha=0.7)

    ani = animation.FuncAnimation(fig, update, frames=len(generations), repeat=False)
    ani.save('wind_farm_optimization.gif', writer='imagemagick')
    plt.close(fig)
