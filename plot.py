import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(x,y):
    
    # Configura a figura
    fig, ax = plt.subplots()

    # Plota as coordenadas das turbinas
    ax.plot(x, y, 'bo', label='Turbine Locations', markersize=8)

    # Adiciona um círculo representando a fronteira
    circle = plt.Circle((0, 0), 1300, color='r', fill=False, linestyle='--', label='Boundary', linewidth=2)
    ax.add_artist(circle)

    # Configura os limites e rótulos
    ax.set_xlim(-1400, 1400)
    ax.set_ylim(-1400, 1400)
    ax.set_aspect('equal', 'box')  # Mantém a proporção igual entre os eixos X e Y
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('Last generation' )

    # Adiciona uma grade e uma legenda
    ax.grid(True, linestyle='--', alpha=0.7)
    

    # Salva o gráfico em um arquivo
    plt.savefig('wind_farm_solution.png', dpi=300)
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

def plot_max(x,y):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Geração")
    plt.ylabel("Aptidão Máxima")
    plt.title("Evolução da Aptidão Máxima ao Longo das Gerações")
    plt.grid()
    plt.show()