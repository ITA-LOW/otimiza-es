import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import csv

estilo='default'

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

def save_logbook_to_csv(logbook, filename):
    # Abre o arquivo CSV no modo de escrita
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escreve o cabeçalho
        writer.writerow(['Generation', 'MaxFitness'])
        # Escreve os dados do logbook
        for entry in logbook:
            writer.writerow([entry['gen'], entry['max']])


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

def plot_multiple_tests(file_list):
    plt.style.use(estilo)
    
    fig, ax = plt.subplots(figsize=(8, 6))  # Configura a figura
    
    # Itera sobre cada arquivo de dados
    for filename in file_list:
        generations = []
        max_fitness = []
        
        # Lê os dados de cada arquivo CSV
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                generations.append(int(row['Generation']))
                max_fitness.append(float(row['MaxFitness']))
        
        # Plota cada execução com uma cor diferente
        ax.plot(generations, max_fitness, label=f'Test: {filename}', linewidth=1)

    # Configurações do gráfico
    ax.set_title('Max Fitness x Generations for Multiple Tests', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Salva o gráfico em um arquivo
    plt.savefig('max_fitness_multiple_tests.png', dpi=300, bbox_inches='tight')
    plt.show()


""" ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 
 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 
 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 
 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks',
 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'] """