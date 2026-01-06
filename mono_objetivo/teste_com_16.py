# --- START OF FILE wind_farm_GA_FINAL_RECORDE.py ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML

# REPRODUTIBILIDADE TOTAL
random.seed(42)
np.random.seed(42)

# IDÊNTICO AO SEU
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

IND_SIZE = 16  
CIRCLE_RADIUS = 1300  
N_DIAMETERS = 260  

TURB_LOC_DATA = getTurbLocYAML("config/iea37-ex16.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("config/iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("config/iea37-windrose.yaml")
initial_coordinates, _, _ = TURB_LOC_DATA
BASE_COORDS_FLAT = np.array(initial_coordinates).flatten().tolist()

# ==========================================
# A SUA FUNÇÃO DE AVALIAÇÃO (NÃO MEXI EM NADA)
# ==========================================
def evaluate_otimizado(individual):
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    # Círculo
    dists_sq = np.sum(turb_coords**2, axis=1)
    penalty_out_of_circle = np.sum(dists_sq > CIRCLE_RADIUS**2) * 1e6

    # Distância
    num_turb = len(turb_coords)
    diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i_upper, j_upper = np.triu_indices(num_turb, k=1)
    close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
    penalty_close_turbines = np.sum(close_mask) * 1e6

    aep = calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], WIND_ROSE_DATA[0],
                  TURB_ATRBT_DATA[4], TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], 
                  TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3])
    
    return np.sum(aep) - penalty_out_of_circle - penalty_close_turbines,

# ==========================================
# MUTAÇÃO ADAPTATIVA (BOMBA DE SHAO 2025)
# ==========================================
def mutate_bombado(individual, hof_best, sigma, indpb):
    """
    Se estiver estagnado, movemos o indivíduo 10% na direção do melhor (HOF).
    Isso é o 'Learning-based Optimization' da sua revisão.
    """
    ind_arr = np.array(individual)
    best_arr = np.array(hof_best)
    
    # Aplica um 'puxão' em direção ao melhor layout do mundo
    if random.random() < 0.2: # 20% de chance de usar a inteligência do HOF
        ind_arr = ind_arr + 0.1 * (best_arr - ind_arr)
    
    # Aplica a sua mutação Gaussiana original por cima
    for i in range(len(ind_arr)):
        if random.random() < indpb:
            ind_arr[i] += random.gauss(0, sigma)
            
    # Sua função de restrição original
    for i in range(IND_SIZE):
        x, y = ind_arr[2*i], ind_arr[2*i + 1]
        if (x**2 + y**2) > CIRCLE_RADIUS**2:
            angle = np.arctan2(y, x)
            ind_arr[2*i] = CIRCLE_RADIUS * np.cos(angle)
            ind_arr[2*i+1] = CIRCLE_RADIUS * np.sin(angle)
            
    individual[:] = ind_arr.tolist()
    return individual,

# SETUP (IDÊNTICO AO SEU)
toolbox.register("individual", lambda: creator.Individual(BASE_COORDS_FLAT))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_otimizado)

def main():
    start_time = time.time()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Aumentamos a população para 400 (mais diversidade conforme Liu 2024)
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)
    
    NGEN = 600
    CXPB, MUTPB = 0.95, 0.7
    
    # Avaliação Inicial
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    print(f"--- GA BOMBADO FINAL (YAML Seed: {hof[0].fitness.values[0]/1e6:.5f} GWh) ---")

    last_max = hof[0].fitness.values[0]
    stagnation = 0

    for gen in range(1, NGEN + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values

        # MUTAÇÃO ADAPTATIVA
        sigma = 250 if stagnation > 30 else 100
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                # Passamos o HOF para guiar a mutação
                mutate_bombado(offspring[i], hof[0], sigma, 0.4)
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        if hof[0].fitness.values[0] > last_max:
            last_max = hof[0].fitness.values[0]
            stagnation = 0
        else:
            stagnation += 1

        if gen % 10 == 0:
            print(f"Gen {gen} | Record: {hof[0].fitness.values[0]/1e6:.6f} GWh | Stagnation: {stagnation}")

    pool.close()
    pool.join()
    print(f"\nRESULTADO FINAL: {hof[0].fitness.values[0]/1e6:.6f} GWh")
    return hof[0]

if __name__ == "__main__":
    main()