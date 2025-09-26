import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution_circle
import multiprocessing
import time
import matplotlib.pyplot as plt

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carregando coordenadas iniciais
config_dir = "config"
main_yaml_path = os.path.join(config_dir, 'iea37-ex16.yaml')
initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    x = np.asarray(x)
    y = np.asarray(y)
    return x**2 + y**2 <= radius**2

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

# Pré-carrega os dados fora da função evaluate:
full_path_turb = os.path.join(config_dir, fname_turb)
full_path_wr = os.path.join(config_dir, fname_wr)
TURB_LOC_DATA = (initial_coordinates, full_path_turb, full_path_wr)
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)

def evaluate_otimizado(individual, turb_loc_data=TURB_LOC_DATA,
             turb_atrbt_data=TURB_ATRBT_DATA,
             wind_rose_data=WIND_ROSE_DATA):
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6

    num_turb = len(turb_coords)
    if num_turb > 1:
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6

    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    return fitness,

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4) 
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_otimizado)

def main():
    random.seed(42)
    start_time = time.time()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # --- Configuração da Visualização em Tempo Real ---
    plt.ion() # Ligar modo interativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Otimização do Parque Eólico em Tempo Real', fontsize=20)

    # --- Parâmetros do Algoritmo Adaptativo ---
    NGEN = 500
    PATIENCE = 50
    MIN_DELTA = 10.0
    CXPB = 0.95
    MUTPB = 0.7
    SIGMA_NORMAL = 100
    SIGMA_AGGRESSIVE = 250
    AGGRESSIVE_DURATION = 15

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    stagnation_counter = 0
    aggressive_phase_triggered = False
    aggressive_phase_countdown = 0
    last_max_fitness = 0.0

    # Avaliação inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    last_max_fitness = hof[0].fitness.values[0]

    # --- Laço de Evolução Customizado ---
    for gen in range(1, NGEN + 1):
        
        # Lógica de estagnação e mutação adaptativa
        current_max_fitness = hof[0].fitness.values[0]
        if (current_max_fitness - last_max_fitness) < MIN_DELTA:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        last_max_fitness = current_max_fitness

        if stagnation_counter >= PATIENCE:
            if not aggressive_phase_triggered:
                print(f"--- Estagnação detectada na geração {gen}. Ativando mutação agressiva (Sigma={SIGMA_AGGRESSIVE}). ---")
                toolbox.register("mutate", mutate, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)
                aggressive_phase_triggered = True
                aggressive_phase_countdown = AGGRESSIVE_DURATION
                stagnation_counter = 0
            else:
                print(f"--- Estagnação persiste. Encerrando na geração {gen}. ---")
                break

        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                print(f"--- Fim da fase de mutação agressiva. Retornando para Sigma={SIGMA_NORMAL}. ---")
                toolbox.register("mutate", mutate, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)

        # Evolução
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # --- Atualização da Visualização ---
        ax1.clear()
        gens = logbook.select("gen")
        max_fitness = logbook.select("max")

        if len(gens) > 3:
            ax1.plot(gens[3:], max_fitness[3:], 'b-', label='Max Fitness')
        else:
            ax1.plot(gens, max_fitness, 'b-', label='Max Fitness')

        ax1.set_xlabel('Geração', fontsize=12)
        ax1.set_ylabel('Max Fitness (AEP)', fontsize=12)
        ax1.set_title('Evolução do Fitness', fontsize=14)
        ax1.grid(True, linestyle='--')
        ax1.legend()
        ax1.ticklabel_format(style='plain', axis='y')

        ax2.clear()
        best_ind_coords = np.array(hof[0]).reshape((IND_SIZE, 2))
        ax2.plot(best_ind_coords[:, 0], best_ind_coords[:, 1], 'bo', markersize=8)
        circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='r', fill=False, linestyle='--', linewidth=1.5)
        ax2.add_artist(circle)
        ax2.set_xlim(-CIRCLE_RADIUS - 100, CIRCLE_RADIUS + 100)
        ax2.set_ylim(-CIRCLE_RADIUS - 100, CIRCLE_RADIUS + 100)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title(f'Melhor Layout na Geração {gen}', fontsize=14)
        ax2.set_xlabel('Coordenada X (m)')
        ax2.set_ylabel('Coordenada Y (m)')
        ax2.grid(True, linestyle='--')

        fig.canvas.draw()
        plt.pause(0.05)

    pool.close()
    pool.join()

    print("\n--- Otimização Concluída ---")
    end_time = time.time()
    total_min = int((end_time - start_time)//60)
    total_sec = int((end_time - start_time)%60)
    print(f"Tempo de computação: {total_min}:{total_sec}")

    plt.ioff()
    print("Pressione Enter para fechar a janela do gráfico.")
    plt.show()

if __name__ == "__main__":
    main()