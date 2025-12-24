# multi_objetivo/otimizacao_multi.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
import matplotlib.pyplot as plt

# Importações dos módulos do projeto
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from multi_objetivo.cabling import analisar_layout_completo

# --- 1. CONFIGURAÇÃO DA DEAP PARA MULTI-OBJETIVO ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# --- 2. CONFIGURAÇÃO DOS PARÂMETROS GLOBAIS ---
IND_SIZE = 16
CIRCLE_RADIUS = 1300
MIN_TURB_DIST_IN_DIAMETERS = 2.0
N_GRUPOS_CABEAMENTO = int(np.sqrt(IND_SIZE))

# Limites para a posição da subestação (retângulo de busca)
SUBSTATION_BOUNDS = [-1500, -1300, -100, 100]  # [xmin, xmax, ymin, ymax]

# --- 3. PRÉ-CARREGAMENTO DE DADOS ---
config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
main_yaml_path = os.path.join(config_dir, f'iea37-ex{IND_SIZE}.yaml')
try:
    initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)
    full_path_turb = os.path.join(config_dir, fname_turb)
    full_path_wr = os.path.join(config_dir, fname_wr)
    
    TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
    WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)
    TURB_DIAM = TURB_ATRBT_DATA[-1]
    MIN_TURB_DIST_METERS = MIN_TURB_DIST_IN_DIAMETERS * TURB_DIAM
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos de configuração: {e}")
    sys.exit(1)

# --- 4. FUNÇÃO DE AVALIAÇÃO MULTI-OBJETIVO ---
def evaluate_multi_objective(individual):
    # Desempacota o indivíduo: 32 para turbinas, 2 para subestação
    turb_coords = np.array(individual[:-2]).reshape((IND_SIZE, 2))
    sub_coords = np.array(individual[-2:])
    
    # --- Penalidades ---
    # Penalidade por turbinas fora do círculo
    distances_from_center = np.linalg.norm(turb_coords, axis=1)
    penalty_out_of_circle = np.sum(np.maximum(0, distances_from_center - CIRCLE_RADIUS)) * 1e6

    # Penalidade por turbinas muito próximas
    penalty_close_turbines = 0
    if IND_SIZE > 1:
        dist_matrix = np.linalg.norm(turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        close_distances = dist_matrix[dist_matrix < MIN_TURB_DIST_METERS]
        penalty_close_turbines = np.sum(MIN_TURB_DIST_METERS - close_distances) * 1e6

    # Penalidade para subestação fora dos limites
    penalty_sub_bounds = 0
    if not (SUBSTATION_BOUNDS[0] <= sub_coords[0] <= SUBSTATION_BOUNDS[1] and
            SUBSTATION_BOUNDS[2] <= sub_coords[1] <= SUBSTATION_BOUNDS[3]):
        penalty_sub_bounds = 1e9 # Penalidade alta e fixa

    total_penalty = penalty_out_of_circle + penalty_close_turbines + penalty_sub_bounds

    # --- Objetivo 1: AEP ---
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = TURB_ATRBT_DATA
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    
    aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                         turb_diam, turb_ci, turb_co, rated_ws, rated_pwr))
    
    # --- Objetivo 2: Custo de Cabeamento e Perda de Energia no Cabeamento ---
    layout_com_sub = np.vstack([sub_coords.reshape(1, 2), turb_coords])
    substation_idx = 0
    
    cabling_cost = 0
    perda_anual_cabeamento = 0
    try:
        _, resultados_cabeamento = analisar_layout_completo(
            coordenadas=layout_com_sub, substation_idx=substation_idx,
            n_grupos=N_GRUPOS_CABEAMENTO, Vn=33e3, P_turbina=rated_pwr
        )
        cabling_cost = resultados_cabeamento['custo_total_usd']
        perda_anual_cabeamento = resultados_cabeamento['perda_anual_mwh']
    except Exception as e:
        cabling_cost = 2e9 # Penalidade muito alta se o cabeamento falhar
        perda_anual_cabeamento = 1e9 # Penalidade alta para perda de energia

    aep_liquido = aep_bruto - perda_anual_cabeamento
    aep_fitness = aep_liquido - total_penalty
        
    return aep_fitness, cabling_cost


# --- 5. CONFIGURAÇÃO DA TOOLBOX DA DEAP ---
toolbox = base.Toolbox()

def create_individual():
    # Cria a parte das turbinas a partir do layout inicial
    turb_part = np.array(initial_coordinates).flatten().tolist()
    # Cria a parte da subestação com uma posição aleatória dentro dos limites
    sub_x = random.uniform(SUBSTATION_BOUNDS[0], SUBSTATION_BOUNDS[1])
    sub_y = random.uniform(SUBSTATION_BOUNDS[2], SUBSTATION_BOUNDS[3])
    return creator.Individual(turb_part + [sub_x, sub_y])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def custom_mutate(individual, mu, sigma, indpb):
    # Muta apenas as coordenadas das turbinas
    size_turbines = len(individual) - 2
    for i in range(size_turbines):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
    
    # Muta as coordenadas da subestação com limites
    if random.random() < indpb:
        individual[-2] += random.gauss(mu, sigma)
        individual[-1] += random.gauss(mu, sigma)
        # Garante que a subestação fique nos limites após a mutação
        individual[-2] = np.clip(individual[-2], SUBSTATION_BOUNDS[0], SUBSTATION_BOUNDS[1])
        individual[-1] = np.clip(individual[-1], SUBSTATION_BOUNDS[2], SUBSTATION_BOUNDS[3])

    return individual,

toolbox.register("evaluate", evaluate_multi_objective)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", custom_mutate, mu=0, sigma=100, indpb=0.4)
toolbox.register("select", tools.selNSGA2)

# --- 6. FUNÇÃO PRINCIPAL (MAIN) ---
def main():
    random.seed(42)
    NGEN = 300
    POP = 300
    CXPB = 0.95
    MUTPB = 0.7
    
    # Configuração do multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP)
    pareto_front = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg", "min", "max"

    # Avaliação inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pareto_front.update(pop)
    
    print("--- Início da Evolução Multi-Objetivo com Subestação Otimizada ---")

    for gen in range(1, NGEN + 1):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop = toolbox.select(offspring + pop, MU)
        pareto_front.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(f"Geração {gen}: AEP Max={record['max'][0]:.2f}, Custo Min={record['min'][1]:.2f}, #Sol={len(pareto_front)}")

    pool.close()
    pool.join()

    print("\n--- Fim da Evolução ---")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'multi_objective_sub')
    os.makedirs(output_dir, exist_ok=True)
    
    pareto_solutions = []
    for i, ind in enumerate(pareto_front):
        solution = {
            'rank': i,
            'aep': ind.fitness.values[0],
            'cost_usd': ind.fitness.values[1],
            'layout_turbines': np.array(ind[:-2]).reshape((IND_SIZE, 2)).tolist(),
            'layout_substation': np.array(ind[-2:]).tolist()
        }
        pareto_solutions.append(solution)

    results_path = os.path.join(output_dir, 'pareto_solutions.json')
    with open(results_path, 'w') as f:
        import json
        json.dump(pareto_solutions, f, indent=2)

    print(f"Resultados da fronteira de Pareto salvos em: {results_path}")

if __name__ == "__main__":
    main()
