"""
Estratégia Híbrida em Duas Fases para Otimização de Parques Eólicos
Baseado em: ESTRATEGIA_HIBRIDA_ANALISE.md

Fase 1: Otimização de Layout (AEP Bruto apenas) - Muito rápida, exploração intensa
Fase 2: Otimização Multiobjetivo (AEP Líquido + Custo) - Refinamento partindo dos melhores layouts da Fase 1

Vantagens:
- Fase 1: 10-50x mais rápido (sem cálculo de cabeamento), mais avaliações
- Fase 2: Parte de soluções com AEP bruto alto, foca em refinamento
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools

# Módulos customizados do projeto
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multi_objetivo.cabling_v3 as cabling_v3

# =============================================================================
# CONFIGURAÇÃO DO AMBIENTE DEAP E CONSTANTES
# =============================================================================

# Limpa tipos anteriores se existirem (para evitar conflitos)
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "IndividualPhase1"):
    del creator.IndividualPhase1
if hasattr(creator, "IndividualPhase2"):
    del creator.IndividualPhase2

# Fase 1: Single-objective (apenas AEP bruto)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

# Fase 2: Multi-objective (AEP líquido + Custo)
creator.create("FitnessMulti", base.Fitness, weights=(100.0, -1.0))  # AEP tem peso 100x maior!
creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)

# Cria toolboxes separadas para cada fase
toolbox_phase1 = base.Toolbox()
toolbox_phase2 = base.Toolbox()

# Parâmetros do Parque Eólico e da Otimização
IND_SIZE = 16
CIRCLE_RADIUS = 1300
N_DIAMETERS = 260
SUBSTATION_CONTINENT = np.array([[-1.0, -1350.0]])

# Limites para número de grupos de cabeamento (será otimizado pelo AG na Fase 2)
MIN_GRUPOS = 2   # Mínimo: 2 grupos
MAX_GRUPOS = 16  # Máximo: 16 grupos (uma turbina por grupo)
N_GRUPOS_INICIAL = 4  # Valor inicial

# =============================================================================
# PRÉ-CARREGAMENTO DE DADOS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = "config"
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)

full_path_wr = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
full_path_turb = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)

# =============================================================================
# FUNÇÕES DE INICIALIZAÇÃO, RESTRIÇÃO E MUTAÇÃO - FASE 1
# COPIADO EXATAMENTE DE wind_farm_GA_16.py
# =============================================================================

def create_individual_from_coordinates(coords):
    """Cria indivíduo a partir de coordenadas - EXATO de wind_farm_GA_16.py"""
    individual = creator.IndividualPhase1(np.array(coords).flatten().tolist())
    return individual

toolbox_phase1.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox_phase1.register("population", tools.initRepeat, list, toolbox_phase1.individual)

def is_within_circle(x, y, radius):
    """Verifica se coordenadas estão dentro do círculo - EXATO de wind_farm_GA_16.py"""
    x = np.asarray(x)
    y = np.asarray(y)
    return x**2 + y**2 <= radius**2

def enforce_circle(individual):
    """Enforce circle constraint - EXATO de wind_farm_GA_16.py"""
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

def mutate_phase1(individual, mu, sigma, indpb):
    """Mutação - EXATA de wind_farm_GA_16.py (cria novo indivíduo)"""
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.IndividualPhase1(individual.tolist()),

def create_individual_phase2_from_coords(coords):
    """Cria um indivíduo da Fase 2 a partir de coordenadas da Fase 1."""
    # Garante que coords é um array numpy
    if isinstance(coords, list):
        coords = np.array(coords)
    
    # Se for 2D (IND_SIZE, 2), achata para 1D
    if coords.ndim == 2:
        coords_flat = coords.flatten().tolist()
    else:
        coords_flat = coords.tolist() if isinstance(coords, np.ndarray) else list(coords)
    
    # Garante que temos exatamente IND_SIZE * 2 elementos
    if len(coords_flat) != IND_SIZE * 2:
        raise ValueError(f"Coordenadas devem ter {IND_SIZE * 2} elementos, recebido {len(coords_flat)}")
    
    # Adiciona número de grupos normalizado
    n_grupos_normalizado = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    return creator.IndividualPhase2(coords_flat + [n_grupos_normalizado])

# Não registramos individual diretamente pois precisamos passar coords dinamicamente
# Usaremos create_individual_phase2_from_coords diretamente quando necessário

def mutate_phase2(individual, mu, sigma, indpb):
    """Mutação para Fase 2: coordenadas + número de grupos."""
    individual_arr = np.array(individual)
    n_coords = IND_SIZE * 2
    
    if random.random() < indpb:
        # Muta coordenadas das turbinas
        for i in range(n_coords):
            individual_arr[i] += random.gauss(mu, sigma)
        
        # Muta número de grupos (último elemento)
        if random.random() < 0.3:  # 30% de chance de mutar número de grupos
            individual_arr[-1] += random.gauss(0, 0.1)
            individual_arr[-1] = max(0.0, min(1.0, individual_arr[-1]))
        
        mutated_list = individual_arr.tolist()
        enforce_circle(mutated_list[:n_coords])
        
        for i in range(len(individual)):
            individual[i] = mutated_list[i]
            
    return individual,

# =============================================================================
# FUNÇÕES DE AVALIAÇÃO - FASE 1 (APENAS AEP BRUTO)
# COPIADA EXATAMENTE DE wind_farm_GA_16.py (evaluate_otimizado)
# =============================================================================

# Pré-carrega dados como em wind_farm_GA_16.py
TURB_LOC_DATA_P1 = getTurbLocYAML(main_yaml_path)
TURB_ATRBT_DATA_P1 = TURB_ATRBT_DATA
WIND_ROSE_DATA_P1 = WIND_ROSE_DATA

def evaluate_phase1(individual, turb_loc_data=TURB_LOC_DATA_P1,
                    turb_atrbt_data=TURB_ATRBT_DATA_P1,
                    wind_rose_data=WIND_ROSE_DATA_P1):
    """
    Avaliação EXATA de wind_farm_GA_16.py (evaluate_otimizado).
    """
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

# =============================================================================
# FUNÇÕES DE AVALIAÇÃO - FASE 2 (AEP LÍQUIDO + CUSTO)
# =============================================================================

def detectar_sobreposicao_cabos(paths, coords):
    """Detecta sobreposições entre segmentos de cabos."""
    def segmentos_intersectam(p1, p2, q1, q2):
        def orientacao(o, a, b):
            val = (a[1] - o[1]) * (b[0] - a[0]) - (a[0] - o[0]) * (b[1] - a[1])
            if abs(val) < 1e-9:
                return 0
            return 1 if val > 0 else 2
        
        def no_segmento(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientacao(p1, p2, q1)
        o2 = orientacao(p1, p2, q2)
        o3 = orientacao(q1, q2, p1)
        o4 = orientacao(q1, q2, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and no_segmento(p1, q1, p2):
            return True
        if o2 == 0 and no_segmento(p1, q2, p2):
            return True
        if o3 == 0 and no_segmento(q1, p1, q2):
            return True
        if o4 == 0 and no_segmento(q1, p2, q2):
            return True
        return False
    
    n_overlaps = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_i = paths[i]
            path_j = paths[j]
            
            for k in range(len(path_i) - 1):
                p1 = coords[path_i[k]]
                p2 = coords[path_i[k + 1]]
                
                for l in range(len(path_j) - 1):
                    q1 = coords[path_j[l]]
                    q2 = coords[path_j[l + 1]]
                    
                    if (path_i[k] == path_j[l] or path_i[k] == path_j[l + 1] or
                        path_i[k + 1] == path_j[l] or path_i[k + 1] == path_j[l + 1]):
                        continue
                    
                    if segmentos_intersectam(p1, p2, q1, q2):
                        n_overlaps += 1
    
    return n_overlaps * 1e6

def evaluate_phase2(individual):
    """
    Fase 2: Avalia AEP líquido + Custo (com cabeamento completo).
    O número de grupos de cabeamento é extraído do genoma e otimizado pelo AG.
    """
    try:
        n_coords = IND_SIZE * 2
        # Extrai coordenadas e número de grupos do indivíduo
        coords_flat = individual[:n_coords]
        n_grupos_normalizado = individual[-1]
        
        # Converte número de grupos normalizado para valor real (sempre int)
        n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
        n_grupos = int(np.round(n_grupos_float))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        
        turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # Penalidades
        dist_from_center = np.linalg.norm(turb_coords, axis=1)
        penalty_out_of_circle = np.sum(np.maximum(0, dist_from_center - CIRCLE_RADIUS)) * 1e6

        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(IND_SIZE, k=1)
        close_distances = dist_matrix[i_upper, j_upper]
        violations_close = close_distances < N_DIAMETERS
        penalty_close_turbines = np.sum(np.maximum(0, N_DIAMETERS - close_distances[violations_close])) * 1e6
        
        # AEP Bruto
        _, _, _, _, turb_diam = TURB_ATRBT_DATA
        aep_bruto = np.sum(calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], 
                                   WIND_ROSE_DATA[0], turb_diam, *TURB_ATRBT_DATA[0:2], 
                                   *TURB_ATRBT_DATA[2:4]))

        # Cabeamento usando número de grupos do genoma
        distancias_ao_continente = np.linalg.norm(turb_coords - SUBSTATION_CONTINENT, axis=1)
        ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
        
        try:
            planta, resultados = cabling_v3.analisar_layout_completo(
                turb_coords, sub=ponto_de_coleta_idx, n_grupos=n_grupos)
            
            custo_total = resultados['custo_total_usd']
            perdas_joule_mwh = resultados['perda_anual_mwh']
            penalty_overlap = detectar_sobreposicao_cabos(planta.paths, turb_coords)
            
            aep_liquido = aep_bruto - perdas_joule_mwh - penalty_out_of_circle - penalty_close_turbines - penalty_overlap
            custo_penalizado = custo_total + penalty_out_of_circle + penalty_close_turbines + penalty_overlap
            
        except Exception as e:
            # Se houver erro no cabeamento, penaliza fortemente
            print(f"Erro no cabeamento com {n_grupos} grupos: {e}")
            return -1e6, 1e12
        
        if aep_liquido <= 0:
            return -1e6, 1e12
        
        return aep_liquido, custo_penalizado
        
    except Exception as e:
        print(f"Erro na avaliação Fase 2: {e}. Penalizando indivíduo.")
        return -1e6, 1e12

# =============================================================================
# CONFIGURAÇÃO DAS TOOLBOXES
# =============================================================================

# Toolbox Fase 1 (Single-objective) - EXATO de wind_farm_GA_16.py
toolbox_phase1.register("mate", tools.cxBlend, alpha=0.5)
toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=100, indpb=0.4)
toolbox_phase1.register("select", tools.selTournament, tournsize=5)  # EXATO: tournsize=5
toolbox_phase1.register("evaluate", evaluate_phase1)

# Toolbox Fase 2 (Multi-objective)
def mate_phase2(ind1, ind2):
    """Crossover que trata coordenadas e número de grupos separadamente."""
    # Crossover blend para coordenadas
    n_coords = IND_SIZE * 2
    tools.cxBlend(ind1[:n_coords], ind2[:n_coords], alpha=0.5)
    
    # Crossover aritmético para número de grupos (último elemento)
    alpha = random.random()
    temp = alpha * ind1[-1] + (1 - alpha) * ind2[-1]
    ind2[-1] = alpha * ind2[-1] + (1 - alpha) * ind1[-1]
    ind1[-1] = temp
    
    # Garante limites
    ind1[-1] = max(0.0, min(1.0, ind1[-1]))
    ind2[-1] = max(0.0, min(1.0, ind2[-1]))
    
    return ind1, ind2

toolbox_phase2.register("mate", mate_phase2)
toolbox_phase2.register("mutate", mutate_phase2, mu=0, sigma=100, indpb=0.4)
toolbox_phase2.register("select", tools.selNSGA2)
toolbox_phase2.register("evaluate", evaluate_phase2)

# =============================================================================
# FUNÇÃO PRINCIPAL - ESTRATÉGIA HÍBRIDA EM DUAS FASES
# =============================================================================

def optimize_phase1(POP_SIZE, NGEN, CXPB, MUTPB):
    """
    Fase 1: Otimização de Layout (AEP Bruto apenas).
    CÓDIGO COPIADO EXATAMENTE DE wind_farm_GA_16.py main()
    """
    print("=" * 80)
    print("FASE 1: OTIMIZAÇÃO DE LAYOUT (AEP BRUTO)")
    print("Algoritmo EXATO de wind_farm_GA_16.py")
    print("=" * 80)
    
    pool = multiprocessing.Pool()
    toolbox_phase1.register("map", pool.map)
    
    # Cria diretório para salvar evolução da Fase 1
    evolution_dir_phase1 = "pareto_front_results/evolution_phase1"
    os.makedirs(evolution_dir_phase1, exist_ok=True)
    
    pop = toolbox_phase1.population(n=POP_SIZE)
    hof = tools.HallOfFame(maxsize=50)  # Guarda os melhores layouts
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # --- Parameters for Adaptive GA and Early Stopping - EXATOS de wind_farm_GA_16.py ---
    PATIENCE = 50
    MIN_DELTA = 10.0
    SIGMA_NORMAL = 100
    SIGMA_AGGRESSIVE = 250
    AGGRESSIVE_DURATION = 15

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    stagnation_counter = 0
    aggressive_phase_triggered = False
    aggressive_phase_countdown = 0
    last_max_fitness = 0.0
    gen = 0  # Inicializa gen para caso o loop não execute

    # Avalia população inicial - EXATO de wind_farm_GA_16.py
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox_phase1.map(toolbox_phase1.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)
    last_max_fitness = hof[0].fitness.values[0]
    
    # Salva melhor indivíduo da geração inicial (gen 0) da Fase 1
    if len(hof) > 0:
        best_ind_gen0 = hof[0]
        coords = np.array(best_ind_gen0).reshape((IND_SIZE, 2))
        evolution_file = os.path.join(evolution_dir_phase1, f"gen_0000_best.txt")
        with open(evolution_file, 'w') as f:
            x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
            y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
            f.write(f"xc: [{x_str}]\n")
            f.write(f"yc: [{y_str}]\n")
            f.write(f"aep: {best_ind_gen0.fitness.values[0]:.2f}\n")
            f.write(f"phase: 1\n")

    for gen in range(1, NGEN + 1):
        # EXATO de wind_farm_GA_16.py
        current_max_fitness = hof[0].fitness.values[0]

        if (current_max_fitness - last_max_fitness) < MIN_DELTA:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        last_max_fitness = current_max_fitness

        if stagnation_counter >= PATIENCE:
            if not aggressive_phase_triggered:
                print(f"--- Stagnation detected at gen {gen}. Increasing sigma to {SIGMA_AGGRESSIVE} for {AGGRESSIVE_DURATION} generations. ---")
                toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)
                aggressive_phase_triggered = True
                aggressive_phase_countdown = AGGRESSIVE_DURATION
                stagnation_counter = 0
            else:
                print(f"--- Stagnation persists after aggressive mutation. Stopping early at generation {gen}. ---")
                break

        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                print(f"--- End of aggressive mutation phase at generation {gen}. Reverting sigma to {SIGMA_NORMAL}. ---")
                toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)

        # EXATO de wind_farm_GA_16.py - sem elitismo, substitui população completamente
        offspring = toolbox_phase1.select(pop, len(pop))
        offspring = [toolbox_phase1.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i - 1], offspring[i] = toolbox_phase1.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_phase1.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox_phase1.map(toolbox_phase1.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # EXATO: pop[:] = offspring (sem elitismo)
        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # Salva melhor indivíduo da geração (Fase 1)
        if len(hof) > 0:
            best_ind_gen = hof[0]
            coords = np.array(best_ind_gen).reshape((IND_SIZE, 2))
            evolution_file = os.path.join(evolution_dir_phase1, f"gen_{gen:04d}_best.txt")
            with open(evolution_file, 'w') as f:
                x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
                y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
                f.write(f"xc: [{x_str}]\n")
                f.write(f"yc: [{y_str}]\n")
                f.write(f"aep: {best_ind_gen.fitness.values[0]:.2f}\n")
                f.write(f"phase: 1\n")

    pool.close()
    pool.join()
    
    # Seleciona top layouts para Fase 2
    top_layouts = sorted(hof, key=lambda x: x.fitness.values[0], reverse=True)
    n_top = min(30, len(top_layouts))  # Top 10% ou 30 layouts
    best_layouts = top_layouts[:n_top]
    
    print(f"\nFase 1 concluída. Melhor AEP bruto: {hof[0].fitness.values[0]/1000:.2f} GWh")
    print(f"Selecionados {len(best_layouts)} melhores layouts para Fase 2")
    
    # Retorna layouts e número da última geração
    last_gen_phase1 = gen  # Última geração executada (pode ser menor que NGEN se parou cedo)
    return best_layouts, last_gen_phase1

def optimize_phase2(best_layouts_phase1, POP_SIZE, NGEN, CXPB, MUTPB, start_gen_number=0):
    """
    Fase 2: Otimização Multiobjetivo (AEP Líquido + Custo).
    Parte dos melhores layouts da Fase 1 e refina considerando cabeamento.
    
    Args:
        start_gen_number: Número da geração inicial (continua da Fase 1)
    """
    print("\n" + "=" * 80)
    print("FASE 2: OTIMIZAÇÃO MULTIOBJETIVO (AEP LÍQUIDO + CUSTO)")
    print("=" * 80)
    
    pool = multiprocessing.Pool()
    toolbox_phase2.register("map", pool.map)
    
    # Cria diretório para salvar evolução por geração
    evolution_dir = "pareto_front_results/evolution"
    os.makedirs(evolution_dir, exist_ok=True)
    
    # Inicializa população: top layouts da Fase 1 + alguns aleatórios
    pop = []
    for layout in best_layouts_phase1:
        # Converte layout da Fase 1 para formato da Fase 2
        # layout é um IndividualPhase1 (lista de coordenadas)
        # Garantimos que é uma lista plana de 32 elementos (16 turbinas * 2 coordenadas)
        coords_list = list(layout)  # Converte para lista Python pura
        if len(coords_list) != IND_SIZE * 2:
            print(f"AVISO: Layout tem {len(coords_list)} elementos, esperado {IND_SIZE * 2}. Pulando...")
            continue
        
        # Converte para numpy array e reshape
        coords_array = np.array(coords_list, dtype=float)
        coords = coords_array.reshape((IND_SIZE, 2))
        
        # Cria indivíduo da Fase 2
        ind_phase2 = create_individual_phase2_from_coords(coords)
        pop.append(ind_phase2)
    
    # Adiciona indivíduos aleatórios para diversidade
    n_random = POP_SIZE - len(pop)
    if n_random > 0:
        for _ in range(n_random):
            # Cria indivíduo aleatório a partir das coordenadas iniciais
            coords_flat = np.array(initial_coordinates).flatten()
            # Adiciona pequena perturbação aleatória
            coords_perturbed = [c + random.gauss(0, 50) for c in coords_flat]
            enforce_circle(coords_perturbed)
            # Converte para formato 2D e cria indivíduo
            coords_2d = np.array(coords_perturbed).reshape((IND_SIZE, 2))
            ind_phase2 = create_individual_phase2_from_coords(coords_2d)
            pop.append(ind_phase2)
    
    hof = tools.ParetoFront()
    
    stats_aep = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_cost = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats = tools.MultiStatistics(aep=stats_aep, cost=stats_cost)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Parâmetros para detecção de estagnação (similar à Fase 1)
    PATIENCE = 50
    MIN_DELTA_AEP = 10.0  # MWh - mudança mínima em AEP
    MIN_DELTA_COST = 100.0  # USD - mudança mínima em custo (ajustado para detectar melhorias pequenas)
    
    stagnation_counter = 0
    last_best_aep = 0.0
    last_best_cost = float('inf')
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox_phase2.map(toolbox_phase2.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pop = [ind for ind in pop if ind.fitness.values[0] > 0]
    hof.update(pop)
    
    # Inicializa valores de referência
    if len(hof) > 0:
        hof_valid = [ind for ind in hof if ind.fitness.values[0] > 0]
        if len(hof_valid) > 0:
            last_best_aep = max(ind.fitness.values[0] for ind in hof_valid)
            last_best_cost = min(ind.fitness.values[1] for ind in hof_valid)
            
            # Salva melhor indivíduo da geração inicial (gen 0)
            best_ind_gen0 = max(hof_valid, key=lambda x: x.fitness.values[0])
            n_coords = IND_SIZE * 2
            coords_flat = best_ind_gen0[:n_coords]
            coords = np.array(coords_flat).reshape((IND_SIZE, 2))
            
            # Numeração continua da Fase 1
            gen_number = start_gen_number + 1  # +1 porque gen 0 da Fase 2 é após última gen da Fase 1
            # Extrai número de grupos
            n_grupos_normalizado = best_ind_gen0[-1]
            n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
            n_grupos = int(np.round(n_grupos_float))
            n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
            
            evolution_file = os.path.join(evolution_dir, f"gen_{gen_number:04d}_best.txt")
            with open(evolution_file, 'w') as f:
                x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
                y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
                f.write(f"xc: [{x_str}]\n")
                f.write(f"yc: [{y_str}]\n")
                f.write(f"aep: {best_ind_gen0.fitness.values[0]:.2f}\n")
                f.write(f"cost: {best_ind_gen0.fitness.values[1]:.2f}\n")
                f.write(f"n_grupos: {n_grupos}\n")
                f.write(f"phase: 2\n")
    
    for gen in range(1, NGEN + 1):
        offspring = toolbox_phase2.select(pop, len(pop))
        offspring = [toolbox_phase2.clone(ind) for ind in offspring]
        
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox_phase2.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_phase2.mutate(offspring[i])
                del offspring[i].fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox_phase2.map(toolbox_phase2.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox_phase2.select(pop + offspring, POP_SIZE)
        hof.update(pop)
        
        hof_valid = [ind for ind in hof if ind.fitness.values[0] > 0]
        hof.clear()
        hof.update(hof_valid)
        
        record = stats.compile(pop)
        n_valid = sum(1 for ind in pop if ind.fitness.values[0] > 0)
        
        aep_max_mwh = record['aep']['max']
        cost_min_usd = record['cost']['min']
        
        # Detecção de estagnação (verifica melhorias nos extremos da frente de Pareto)
        current_best_aep = aep_max_mwh
        current_best_cost = cost_min_usd
        
        # Verifica se houve melhoria significativa em AEP ou custo
        # Compara com o melhor histórico, não apenas com a última geração
        aep_improved = (current_best_aep - last_best_aep) >= MIN_DELTA_AEP
        cost_improved = (last_best_cost - current_best_cost) >= MIN_DELTA_COST
        
        # Se houve melhoria, reseta contador
        if aep_improved or cost_improved:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Atualiza valores de referência (sempre mantém o melhor histórico)
        # Isso permite detectar melhorias acumuladas ao longo de várias gerações
        if current_best_aep > last_best_aep:
            last_best_aep = current_best_aep
        if current_best_cost < last_best_cost:
            last_best_cost = current_best_cost
        
        # Parada precoce se estagnado por muitas gerações
        if stagnation_counter >= PATIENCE:
            print(f"\n--- Gen {gen}: Estagnação detectada na Fase 2. Parando precocemente. ---")
            print(f"   Melhor AEP: {last_best_aep/1000:.2f} GWh, Melhor Custo: {last_best_cost:.2f} USD")
            print(f"   Sem melhoria significativa por {PATIENCE} gerações.")
            break
        
        # Calcula número médio de grupos
        if n_valid > 0:
            grupos_list = [
                int(np.round(MIN_GRUPOS + ind[-1] * (MAX_GRUPOS - MIN_GRUPOS))) 
                for ind in pop if ind.fitness.values[0] > 0
            ]
            n_grupos_medio = int(np.round(np.mean(grupos_list))) if grupos_list else 0
        else:
            n_grupos_medio = 0
        
        # Salva melhor indivíduo da geração (maior AEP)
        if len(hof_valid) > 0:
            best_ind_gen = max(hof_valid, key=lambda x: x.fitness.values[0])
            n_coords = IND_SIZE * 2
            coords_flat = best_ind_gen[:n_coords]
            coords = np.array(coords_flat).reshape((IND_SIZE, 2))
            
            # Salva coordenadas do melhor indivíduo desta geração
            # Numeração continua da Fase 1
            gen_number = start_gen_number + gen
            
            # Extrai número de grupos
            n_grupos_normalizado = best_ind_gen[-1]
            n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
            n_grupos = int(np.round(n_grupos_float))
            n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
            
            evolution_file = os.path.join(evolution_dir, f"gen_{gen_number:04d}_best.txt")
            with open(evolution_file, 'w') as f:
                x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
                y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
                f.write(f"xc: [{x_str}]\n")
                f.write(f"yc: [{y_str}]\n")
                f.write(f"aep: {best_ind_gen.fitness.values[0]:.2f}\n")
                f.write(f"cost: {best_ind_gen.fitness.values[1]:.2f}\n")
                f.write(f"n_grupos: {n_grupos}\n")
                f.write(f"phase: 2\n")
        
        if gen % 1 == 0 or gen == NGEN:
            print(f"Gen {gen}: AEP Max={aep_max_mwh:.2f} MWh ({aep_max_mwh/1000:.2f} GWh), "
                  f"Cost Min={cost_min_usd:.2f} USD, "
                  f"Grupos Médio={n_grupos_medio}, "
                  f"Valid={n_valid}/{len(pop)}, Pareto={len(hof)}, Stagnation={stagnation_counter}/{PATIENCE}")
    
    pool.close()
    pool.join()
    
    return hof

def save_results(hof_final):
    """Salva os resultados da Fase 2."""
    output_dir = "pareto_front_results"
    os.makedirs(output_dir, exist_ok=True)
    
    hof_valid = [ind for ind in hof_final if ind.fitness.values[0] > 0]
    print(f"\n--- Otimização concluída. {len(hof_valid)} soluções válidas na Frente de Pareto. ---")
    
    if len(hof_valid) == 0:
        print("AVISO: Nenhuma solução válida encontrada!")
        return
    
    results = []
    for i, individual in enumerate(hof_valid):
        aep_liq, cost = individual.fitness.values
        
        # Extrai coordenadas e número de grupos
        n_coords = IND_SIZE * 2
        coords_flat = individual[:n_coords]
        n_grupos_normalizado = individual[-1]
        n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
        n_grupos = int(np.round(n_grupos_float))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        
        coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # Recalcula perdas para salvar
        distancias_ao_continente = np.linalg.norm(coords - SUBSTATION_CONTINENT, axis=1)
        ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
        
        try:
            _, resultados_cabeamento = cabling_v3.analisar_layout_completo(
                coords, sub=ponto_de_coleta_idx, n_grupos=n_grupos)
            perdas = resultados_cabeamento['perda_anual_mwh']
        except:
            perdas = 0.0
        
        filename = os.path.join(output_dir, f"solution_{i+1}_coords.txt")
        
        with open(filename, 'w') as f:
            x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
            y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
            f.write(f"xc: [{x_str}]\n")
            f.write(f"yc: [{y_str}]\n")
            f.write(f"n_grupos: {n_grupos}\n")
        
        results.append({
            'Solution': i+1, 
            'AEP_Liquido_MWh': aep_liq, 
            'Custo_USD': cost,
            'Perdas_Joule_MWh': perdas,
            'N_Grupos': n_grupos,
            'File': filename
        })
    
    df_pareto = pd.DataFrame(results)
    df_pareto_sorted = df_pareto.sort_values(by='AEP_Liquido_MWh', ascending=False)
    csv_path = os.path.join(output_dir, "pareto_summary.csv")
    df_pareto_sorted.to_csv(csv_path, index=False, float_format='%.2f')
    
    melhor_aep = df_pareto_sorted['AEP_Liquido_MWh'].max()
    print(f"\nMelhor AEP encontrado: {melhor_aep:.2f} MWh ({melhor_aep/1000:.2f} GWh)")
    print(f"Resumo salvo em: {csv_path}")

def main():
    random.seed(42)
    start_time = time.time()
    
    # Parâmetros Fase 1 - EXATOS de wind_farm_GA_16.py
    POP_SIZE_P1 = 300
    NGEN_P1 = 500   # EXATO: 500 gerações
    CXPB_P1 = 0.95
    MUTPB_P1 = 0.7
    
    # Parâmetros Fase 2 (refinamento)
    POP_SIZE_P2 = 300
    NGEN_P2 = 500
    CXPB_P2 = 0.95
    MUTPB_P2 = 0.7
    
    print("=" * 80)
    print("ESTRATÉGIA HÍBRIDA EM DUAS FASES")
    print("Fase 1: Otimização de Layout (AEP Bruto) - Exploração")
    print("Fase 2: Otimização Multiobjetivo (AEP Líquido + Custo) - Refinamento")
    print("=" * 80)
    
    # Fase 1: Otimização de Layout
    best_layouts, last_gen_phase1 = optimize_phase1(POP_SIZE_P1, NGEN_P1, CXPB_P1, MUTPB_P1)
    
    # Fase 2: Otimização Multiobjetivo (continua numeração da Fase 1)
    hof_final = optimize_phase2(best_layouts, POP_SIZE_P2, NGEN_P2, CXPB_P2, MUTPB_P2, start_gen_number=last_gen_phase1)
    
    # Salva resultados
    save_results(hof_final)
    
    total_time = time.time() - start_time
    print(f"\nTempo total: {total_time/60:.2f} minutos")

if __name__ == "__main__":
    main()

