import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

estilo='bmh'

def plot_solution(x, y):
    # Aplica o estilo ao gráfico
    plt.style.use(estilo)
    
    # Configura a figura
    fig, ax = plt.subplots()

    # Plota as coordenadas das turbinas
    ax.plot(x, y, 'bo', markersize=6)

    # Adiciona um círculo representando a fronteira
    circle = plt.Circle((0, 0), 1300, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle)

    # Configura os limites e rótulos
    ax.set_xlim(-1400, 1400)
    ax.set_ylim(-1400, 1400)
    ax.set_aspect('equal', 'box')  
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('Optimized turbines', fontsize=14)

    # Adiciona uma grade e uma legenda
    ax.grid(True, linestyle='--', alpha=0.7)

    # Salva o gráfico em um arquivo
    plt.savefig('wind_farm_solution.png', dpi=300, bbox_inches='tight')
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
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ani = animation.FuncAnimation(fig, update, frames=len(generations), repeat=False)
    ani.save('wind_farm_optimization.gif', writer='imagemagick')
    plt.close(fig)

def plot_fitness(x, y):
    plt.style.use(estilo)  # Aplicando um estilo mais suave com grade escura

    # Configura a figura e os eixos
    fig, ax = plt.subplots(figsize=(8, 6))  # Tamanho maior para melhor visualização

    # Plota os dados com marcadores menores e uma linha mais espessa
    ax.plot(x, y, label='Max Fitness', color='b',linewidth=1)

    # Configurações do título e eixos
    ax.set_title('Max Fitness x Generations', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)

    # Ajusta a grade para um visual mais elegante
    ax.grid(True, linestyle='--', alpha=0.7)

    # Salva o gráfico em um arquivo
    plt.savefig('max_fitness_vs_generations.png', dpi=300, bbox_inches='tight')  # Garante que tudo fique dentro do espaço salvo
    plt.close()

""" ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 
 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 
 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 
 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks',
 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'] """