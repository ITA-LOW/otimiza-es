import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution_circle, plot_fitness, save_logbook_to_csv
import multiprocessing
import time
import os

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
initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    x = np.asarray(x)
    y = np.asarray(y)
    return x**2 + y**2 <= radius**2

def is_within_circle_otimizado(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle_otimizado(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("iea37-ex16.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("iea37-windrose.yaml")

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

# Configuração da otimização
def main():
    random.seed(42)
    start_time = time.time()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)  
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    generation_data = []
    max_fitness_data = []

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.95, mutpb=0.7, ngen=500, 
                                        stats=stats, halloffame=hof, verbose=True)
    
    pool.close()
    pool.join()

    for record in logbook:
        generation_data.append(record['gen'])
        max_fitness_data.append(record['max'])

    # ========================================================================
    # <<< MODIFICAÇÃO ABAIXO >>>
    # O loop de salvamento foi alterado para escrever no formato xc: [...] yc: [...]
    # ========================================================================
    
    output_dir = "best_layouts"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Salvando os {len(hof)} melhores layouts em '{output_dir}/' ---")
    for i, individual in enumerate(hof):
        rank = i + 1
        aep_fitness = individual.fitness.values[0]
        
        # Converte o indivíduo para o formato de coordenadas (16, 2)
        coords = np.array(individual).reshape((IND_SIZE, 2))
        
        # Define o nome do arquivo
        filename = os.path.join(output_dir, f"layout_rank_{rank}_coords.txt")
        
        # Abre o arquivo para escrita
        with open(filename, 'w') as f:
            # Extrai as listas de coordenadas X e Y
            x_coords_list = coords[:, 0]
            y_coords_list = coords[:, 1]
            
            # Formata as listas em strings com 4 casas decimais
            x_str = ", ".join([f"{val:.4f}" for val in x_coords_list])
            y_str = ", ".join([f"{val:.4f}" for val in y_coords_list])
            
            # Escreve no arquivo no formato solicitado
            f.write(f"xc: [{x_str}]\n")
            f.write(f"yc: [{y_str}]\n")
        
        print(f"Layout Rank {rank} (AEP: {aep_fitness:,.2f}) salvo em '{filename}' no formato xc/yc.")

    # --- Plotagem da melhor solução (Rank 1) ---
    print("\nPlotando a melhor solução (Rank 1)...")
    best_coords_for_plot = np.array(hof[0]).reshape((IND_SIZE, 2))
    x_coords = best_coords_for_plot[:, 0]
    y_coords = best_coords_for_plot[:, 1]
    
    plot_solution_circle(x_coords, y_coords, radius=CIRCLE_RADIUS)
    plot_fitness(generation_data[3:], max_fitness_data[3:])
    save_logbook_to_csv(logbook, "set_19")

    end_time = time.time()
    total_min = int((end_time - start_time)//60)
    total_sec = int((end_time - start_time)%60)
    print(f"Tempo de computação: {total_min}:{total_sec}")

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()