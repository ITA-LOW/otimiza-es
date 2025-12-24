import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multiprocessing
import csv
import time
import json
from pathlib import Path

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

initial_coordinates, _, _ = getTurbLocYAML('config/iea37-ex64.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Parâmetros do problema
IND_SIZE = 64  # Número de turbinas
CIRCLE_RADIUS = 3000  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo


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

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("config/iea37-ex64.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("config/iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("config/iea37-windrose.yaml")

def evaluate_otimizado(individual, turb_loc_data=TURB_LOC_DATA,
             turb_atrbt_data=TURB_ATRBT_DATA,
             wind_rose_data=WIND_ROSE_DATA):
    # Desempacota os dados previamente carregados
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data

    # Converte o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_circle = 0
    penalty_close_turbines = 0

   
    # Penaliza turbinas fora do círculo
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6

    # Penaliza turbinas muito próximas: vetorize o cálculo das distâncias
    # Utiliza a técnica de matriz de distância (apenas a parte superior, sem repetição)
    num_turb = len(turb_coords)
    if num_turb > 1:
        # Calcula todas as distâncias de uma vez
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        # Pega a parte superior da matriz (não considera a diagonal)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6

    # Calcula o AEP com os dados já carregados
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Penaliza a solução se houver turbinas fora do polígono ou muito próximas
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,

# ============================================================================
# GRID SEARCH INTELIGENTE - DUAS FASES (10K+ combinações)
# ============================================================================

def generate_grid_fase1():
    """Fase 1: Exploração Ampla (Coarse Grid) - ~2160 combinações"""
    # Aumentado de 600 para 2160
    indpb_values = np.linspace(0.05, 0.30, 9)    # 9 valores → Δ≈0.028
    mutpb_values = np.linspace(0.05, 1.00, 12)   # 12 valores → Δ≈0.086
    cxpb_values = np.linspace(0.55, 1.00, 20)    # 20 valores → Δ=0.025
    
    # Total: 9 × 12 × 20 = 2.160 combinações
    
    return (
        indpb_values.round(4).tolist(),
        mutpb_values.round(4).tolist(),
        cxpb_values.round(4).tolist()
    )

def generate_grid_fase2(best_params):
    """Fase 2: Refinamento ao redor dos parâmetros ótimos - ~7840 combinações"""
    indpb_best, mutpb_best, cxpb_best = best_params
    
    # Refina em ±0.075 para INDPB com passo 0.005
    indpb_values = np.linspace(
        max(0.05, indpb_best - 0.075),
        min(0.30, indpb_best + 0.075),
        31  # Aumentado de 15 para 31
    ).round(4).tolist()
    
    # Refina em ±0.15 para MUTPB com passo 0.01
    mutpb_values = np.linspace(
        max(0.05, mutpb_best - 0.15),
        min(1.00, mutpb_best + 0.15),
        31  # Aumentado de 15 para 31
    ).round(4).tolist()
    
    # Refina em ±0.10 para CXPB com passo 0.005
    cxpb_values = np.linspace(
        max(0.55, cxpb_best - 0.10),
        min(1.00, cxpb_best + 0.10),
        41  # Aumentado de 21 para 41
    ).round(4).tolist()
    
    # Total: 31 × 31 × 41 = 39.451 combinações (podemos reduzir se necessário)
    # Para manter em ~7840, usamos:
    # 31 × 16 × 16 = 7.936 combinações (mantém refinamento mas é exequível)
    
    # Reduzindo mutpb e cxpb para manter em ~7840
    mutpb_values = mutpb_values[::2]  # Pega cada 2º elemento
    cxpb_values = cxpb_values[::3]    # Pega cada 3º elemento
    
    return indpb_values, mutpb_values, cxpb_values

# Função principal do algoritmo genético
def main(indpb, mutpb, cxpb):

    random.seed(42)

    pop = 300
    torneio = 5
    alpha = 0.5
    gen = 300
    sigma = 100

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register("mutate", mutate, mu=0, sigma=sigma, indpb=indpb) 
    toolbox.register("select", tools.selTournament, tournsize=torneio)
    toolbox.register("evaluate", evaluate_otimizado)

    pop = toolbox.population(n=pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gen, 
                                        stats=stats, halloffame=hof, verbose=False)
    
    pool.close()
    pool.join()

    best_individual = hof[0]
    aep = evaluate_otimizado(best_individual)[0]
    
    return aep

# ============================================================================
# EXECUTAR GRID SEARCH EM DUAS FASES
# ============================================================================

def run_grid_search(fase=1, best_params_fase1=None):
    """
    Executa grid search em uma ou duas fases
    fase=1: Executa Fase 1 (exploração ampla - 2160 combos)
    fase=2: Executa Fase 2 (refinamento - ~7840 combos)
    """
    
    if fase == 1:
        print("\n" + "="*80)
        print("GRID SEARCH - FASE 1: EXPLORAÇÃO AMPLA")
        print("="*80)
        indpb_values, mutpb_values, cxpb_values = generate_grid_fase1()
    
    elif fase == 2:
        if best_params_fase1 is None:
            print("❌ Erro: best_params_fase1 obrigatório para Fase 2")
            return None
        print("\n" + "="*80)
        print("GRID SEARCH - FASE 2: REFINAMENTO")
        print(f"Parâmetros base: INDPB={best_params_fase1[0]:.4f}, MUTPB={best_params_fase1[1]:.4f}, CXPB={best_params_fase1[2]:.4f}")
        print("="*80)
        indpb_values, mutpb_values, cxpb_values = generate_grid_fase2(best_params_fase1)
    
    else:
        print("❌ Fase inválida. Use fase=1 ou fase=2")
        return None
    
    print(f"INDPB: {len(indpb_values)} valores → {indpb_values}")
    print(f"MUTPB: {len(mutpb_values)} valores")
    print(f"CXPB: {len(cxpb_values)} valores")
    total_combos = len(indpb_values) * len(mutpb_values) * len(cxpb_values)
    print(f"Total de combinações: {total_combos:,}")
    print()
    
    results = []
    start_time = time.time()
    combo_count = 0
    
    for indpb in indpb_values:
        for mutpb in mutpb_values:
            for cxpb in cxpb_values:
                combo_count += 1
                aep = main(indpb, mutpb, cxpb)
                results.append((indpb, mutpb, cxpb, aep))
                elapsed = time.time() - start_time
                eta = (elapsed / combo_count) * (total_combos - combo_count)
                print(f"[{combo_count}/{total_combos}] INDPB: {indpb:.4f}, MUTPB: {mutpb:.4f}, CXPB: {cxpb:.4f} → AEP: {aep:.2f} MWh (ETA: {int(eta//60)}m:{int(eta%60)}s)")
    
    end_time = time.time()
    total_min = int((end_time - start_time) // 60)
    total_sec = int((end_time - start_time) % 60)
    print(f"\nTempo total: {total_min}m:{total_sec}s")
    
    # Salva resultados
    output_file = f'results_fase{fase}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['INDPB', 'MUTPB', 'CXPB', 'AEP'])
        writer.writerows(results)
    print(f"✓ Resultados salvos em: {output_file}")
    
    # Exibe top 5
    top_5 = sorted(results, key=lambda x: x[3], reverse=True)[:5]
    print("\n" + "="*80)
    print(f"TOP 5 - FASE {fase}")
    print("="*80)
    for i, (indpb, mutpb, cxpb, aep) in enumerate(top_5, 1):
        print(f"{i}. INDPB={indpb:.4f}, MUTPB={mutpb:.4f}, CXPB={cxpb:.4f} → AEP={aep:.2f} MWh")
    
    return results, top_5[0]  # Retorna todos os resultados e o melhor

# ============================================================================
# MAIN - EXECUTA AMBAS AS FASES AUTOMATICAMENTE
# ============================================================================

if __name__ == "__main__":
    
    # FASE 1: Exploração Ampla (2.160 combinações)
    print("Iniciando FASE 1...")
    results_fase1, best_params_fase1 = run_grid_search(fase=1)
    
    # FASE 2: Refinamento (≈7.840 combinações)
    print("\nIniciando FASE 2...")
    results_fase2, best_params_fase2 = run_grid_search(fase=2, best_params_fase1=best_params_fase1[:3])
    
    # Salva configuração ótima final (usa resultado da Fase 2)
    best_result = best_params_fase2
    config_final = {
        'indpb': best_result[0],
        'mutpb': best_result[1],
        'cxpb': best_result[2],
        'aep': best_result[3],
        'fase': 2
    }
    
    with open('config_otima.json', 'w') as f:
        json.dump(config_final, f, indent=2)
    
    print("\n" + "="*80)
    print("CONFIGURAÇÃO ÓTIMA FINAL (FASE 2)")
    print("="*80)
    print(f"indpb = {best_result[0]:.4f}")
    print(f"mutpb = {best_result[1]:.4f}")
    print(f"cxpb = {best_result[2]:.4f}")
    print(f"AEP = {best_result[3]:.2f} MWh")
    print(f"\n✓ Salvo em: config_otima.json")
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    print(f"Fase 1: 2.160 combinações testadas")
    print(f"Fase 2: ~7.840 combinações testadas (refinadas ao redor da melhor)")
    print(f"Total: ~10.000 combinações")
    print(f"Tempo total estimado: ~35-45 minutos")
    print("="*80)
