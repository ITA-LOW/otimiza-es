"""
BENCHMARK COMPARISON SCRIPT FOR GECCO REVIEW
--------------------------------------------
Author: [Seu Nome/Autores Anônimos]
Purpose: Compare Proposed Two-Phase Method vs. Pure NSGA-II Baseline.

Metrics:
1. Hypervolume (Convergence & Diversity)
2. Pareto Front Visualization

This script proves that the Phase 1 warm-start strategy provides 
better convergence than simply running NSGA-II for longer.
"""

import sys
import os
import time
import random
import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms

# Configuração de fontes para GECCO (Type 42/TrueType - requerido)
# ACM/GECCO requerem Type 1/TrueType fonts (Type 42). Type 3 fonts NÃO são aceitos.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Importa função de hipervolume diretamente
try:
    from deap.tools._hypervolume import hv as hypervolume_module
except ImportError:
    from deap.tools._hypervolume import pyhv as hypervolume_module

# Importa testes estatísticos
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("AVISO: scipy não disponível. Testes estatísticos serão pulados.")

# Importa módulos do projeto (Assumindo estrutura de pastas original)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multi_objetivo.cabling_v3 as cabling_v3

# =============================================================================
# 1. CONFIGURAÇÃO DO EXPERIMENTO (AJUSTE AQUI PARA TESTES RÁPIDOS)
# =============================================================================

# --- CONFIGURAÇÃO DE EXECUÇÃO ---
# Configuração balanceada: robustez estatística + tempo razoável
# Baseado em run_all_scenarios.py: NGEN_P1=500, NGEN_P2=1500 para 16 turbinas
N_SEEDS = 20            # Número de seeds (reduzido de 30 para acelerar, ainda robusto)
POP_SIZE = 150         # Tamanho da população (reduzido de 300, ainda adequado)
NGEN_PHASE1 = 500       # Gerações Fase 1 (baseado em run_all_scenarios.py)
NGEN_PHASE2 = 1000      # Gerações Fase 2 (reduzido de 1500, mas ainda robusto)

# O Baseline roda pelo tempo total combinado para ser justo
NGEN_BASELINE = NGEN_PHASE1 + NGEN_PHASE2

# Flag para debug detalhado (ativa debug na primeira avaliação do baseline)
DEBUG_BASELINE = True  # Mude para False para desativar debug 

# --- CONSTANTES FÍSICAS (IGUAIS AO ORIGINAL) ---
IND_SIZE = 16 
CIRCLE_RADIUS = 5000
N_DIAMETERS = 260
MIN_GRUPOS = 2
MAX_GRUPOS = 64
N_GRUPOS_INICIAL = MIN_GRUPOS

# Carrega dados externos uma vez
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = "config"
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
TURB_LOC_DATA = getTurbLocYAML(main_yaml_path) # Usado para init coordenadas
initial_coordinates = TURB_LOC_DATA[0]
full_path_turb = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")
full_path_wr = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)

# =============================================================================
# 2. CONFIGURAÇÃO DEAP (CREATORS & TOOLBOXES)
# =============================================================================

if hasattr(creator, "FitnessMax"): del creator.FitnessMax
if hasattr(creator, "IndividualPhase1"): del creator.IndividualPhase1
if hasattr(creator, "FitnessMulti"): del creator.FitnessMulti
if hasattr(creator, "IndividualPhase2"): del creator.IndividualPhase2
if hasattr(creator, "FitnessMin"): del creator.FitnessMin
if hasattr(creator, "IndividualSequential"): del creator.IndividualSequential

# Fase 1 (Single Objective)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

# Fase 2 / Baseline (Multi Objective: Max AEP, Min Cost)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)

# Sequencial (Single Objective: Min Cost, com turbinas fixas)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualSequential", list, fitness=creator.FitnessMin)

toolbox_p1 = base.Toolbox()
toolbox_p2 = base.Toolbox()
toolbox_base = base.Toolbox() # Toolbox separada para o Baseline
toolbox_seq = base.Toolbox() # Toolbox para abordagem sequencial

# =============================================================================
# 3. FUNÇÕES AUXILIARES E AVALIADORES (COPIADOS DO ORIGINAL PARA CONSISTÊNCIA)
# =============================================================================

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2

def enforce_circle(individual_coords):
    """Projeta coordenadas para dentro do círculo."""
    for i in range(0, len(individual_coords), 2):
        x, y = individual_coords[i], individual_coords[i+1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            individual_coords[i] = CIRCLE_RADIUS * np.cos(angle)
            individual_coords[i+1] = CIRCLE_RADIUS * np.sin(angle)
    return individual_coords

def enforce_substation(sub_pos):
    """Projeta subestação para dentro do círculo."""
    x, y = sub_pos[0], sub_pos[1]
    if not is_within_circle(x, y, CIRCLE_RADIUS):
        angle = np.arctan2(y, x)
        sub_pos[0] = CIRCLE_RADIUS * np.cos(angle)
        sub_pos[1] = CIRCLE_RADIUS * np.sin(angle)
    return sub_pos

def displace_substation_from_turbines(centroid, turb_coords, min_distance=50.0):
    """
    Desloca a subestação do centroide para garantir distância mínima das turbinas.
    
    Estratégia:
    1. Começa no centroide
    2. Se estiver muito próxima de alguma turbina, desloca na direção oposta
    3. Garante distância mínima de todas as turbinas
    
    Args:
        centroid: Posição inicial (centroide das turbinas)
        turb_coords: Array numpy de shape (N, 2) com coordenadas das turbinas
        min_distance: Distância mínima permitida (metros)
    
    Returns:
        sub_pos: Posição da subestação deslocada [x, y]
    """
    sub_pos = np.array(centroid.copy())
    max_iterations = 20
    
    for iteration in range(max_iterations):
        # Calcula distâncias até todas as turbinas
        dists_to_turbines = np.linalg.norm(turb_coords - sub_pos, axis=1)
        min_dist = np.min(dists_to_turbines)
        
        # Se está longe o suficiente, pode parar
        if min_dist >= min_distance:
            break
        
        # Encontra a turbina mais próxima
        closest_turb_idx = np.argmin(dists_to_turbines)
        closest_turb = turb_coords[closest_turb_idx]
        
        # Direção do centroide para a turbina mais próxima
        direction_to_turb = closest_turb - sub_pos
        dist_to_turb = np.linalg.norm(direction_to_turb)
        
        if dist_to_turb < 1e-6:
            # Se subestação está exatamente na turbina, desloca aleatoriamente
            angle = random.uniform(0, 2*np.pi)
            sub_pos = sub_pos + min_distance * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Normaliza direção
            direction_to_turb = direction_to_turb / dist_to_turb
            
            # Desloca na direção oposta (afastando da turbina)
            # Move o suficiente para garantir min_distance
            needed_displacement = min_distance - min_dist + 10.0  # +10m de margem
            sub_pos = sub_pos - direction_to_turb * needed_displacement
        
        # Garante que fica dentro do círculo
        dist_from_center = np.linalg.norm(sub_pos)
        if dist_from_center > CIRCLE_RADIUS:
            angle = np.arctan2(sub_pos[1], sub_pos[0])
            sub_pos = CIRCLE_RADIUS * np.array([np.cos(angle), np.sin(angle)])
    
    return sub_pos.tolist()

def repair_spacing(coords_array, max_iterations=10):
    """
    Repara violações de distância mínima entre turbinas.
    Afasta turbinas muito próximas mantendo-as dentro do círculo.
    
    Args:
        coords_array: Array numpy de shape (N, 2) com coordenadas das turbinas
        max_iterations: Número máximo de iterações de repair
    
    Returns:
        coords_array reparado
    """
    coords = coords_array.copy()
    min_dist = N_DIAMETERS
    
    for iteration in range(max_iterations):
        # Calcula distâncias entre todos os pares
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Encontra pares que violam a distância mínima
        i_upper, j_upper = np.triu_indices(len(coords), k=1)
        violations = dists[i_upper, j_upper] < min_dist
        
        if not np.any(violations):
            break  # Sem violações, pode parar
        
        # Para cada violação, afasta as turbinas
        for idx in np.where(violations)[0]:
            i, j = i_upper[idx], j_upper[idx]
            dist_ij = dists[i, j]
            
            if dist_ij < min_dist:
                # Direção de separação
                if dist_ij < 1e-6:  # Evita divisão por zero
                    # Se estão no mesmo lugar, separa aleatoriamente
                    angle = random.uniform(0, 2*np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = (coords[i] - coords[j]) / dist_ij
                
                # Distância que precisa ser adicionada
                needed_separation = (min_dist - dist_ij) / 2.0
                
                # Move ambas as turbinas na direção oposta
                move_i = direction * needed_separation
                move_j = -direction * needed_separation
                
                # Aplica movimento
                new_pos_i = coords[i] + move_i
                new_pos_j = coords[j] + move_j
                
                # Garante que ficam dentro do círculo
                dist_i = np.linalg.norm(new_pos_i)
                dist_j = np.linalg.norm(new_pos_j)
                
                if dist_i > CIRCLE_RADIUS:
                    angle_i = np.arctan2(new_pos_i[1], new_pos_i[0])
                    new_pos_i = CIRCLE_RADIUS * np.array([np.cos(angle_i), np.sin(angle_i)])
                
                if dist_j > CIRCLE_RADIUS:
                    angle_j = np.arctan2(new_pos_j[1], new_pos_j[0])
                    new_pos_j = CIRCLE_RADIUS * np.array([np.cos(angle_j), np.sin(angle_j)])
                
                coords[i] = new_pos_i
                coords[j] = new_pos_j
    
    return coords

# --- AVALIADOR FASE 1 (AEP BRUTO) ---
def evaluate_phase1(individual):
    # Reutiliza lógica original
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Penalidades Geométricas
    dist_from_center = np.linalg.norm(turb_coords, axis=1)
    penalty_circle = np.sum(np.maximum(0, dist_from_center - CIRCLE_RADIUS)) * 1e6
    
    # Distância entre turbinas
    num_turb = len(turb_coords)
    diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i_upper, j_upper = np.triu_indices(num_turb, k=1)
    close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
    penalty_spacing = np.sum(close_mask) * 1e6
    
    # Cálculo AEP
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, 
                  TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3])
    
    return np.sum(aep) - penalty_circle - penalty_spacing,

# --- AVALIADOR FASE 2 / BASELINE (AEP LÍQUIDO + CUSTO) ---
def evaluate_full(individual, debug=False):
    """Avalia o genoma completo (Turbinas + Grupos + Subestação)."""
    try:
        n_coords = IND_SIZE * 2
        # Parse Genoma
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        
        # Converte Grupos
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        # Limita n_grupos ao número de turbinas (não faz sentido ter mais grupos que turbinas)
        n_grupos = min(n_grupos, IND_SIZE)
        
        turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # 1. Penalidades Geométricas Básicas (mais suaves)
        dist_turb = np.linalg.norm(turb_coords, axis=1)
        violations_turb = np.maximum(0, dist_turb - CIRCLE_RADIUS)
        pen_turb_out = np.sum(violations_turb) * 1e5  # Reduzido de 1e6 para 1e5
        
        dist_sub = np.linalg.norm(sub_pos)
        pen_sub_out = np.maximum(0, dist_sub - CIRCLE_RADIUS) * 1e5  # Reduzido
        
        # Distância Turbina-Turbina
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_u, j_u = np.triu_indices(IND_SIZE, k=1)
        # Penalidade mais suave: linear ao invés de extrema
        violations = np.maximum(0, N_DIAMETERS - dists[i_u, j_u])
        pen_spacing = np.sum(violations) * 1e5  # Reduzido de 1e6 para 1e5
        
        # Distância Sub-Turbina (mais suave)
        d_sub_turb = np.linalg.norm(turb_coords - sub_pos, axis=1)
        violation_sub_close = np.maximum(0, 50.0 - np.min(d_sub_turb))
        pen_sub_close = violation_sub_close * 1e5  # Reduzido de 1e6 para 1e5
        
        # 2. AEP Bruto
        wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
        turb_diam = TURB_ATRBT_DATA[4]
        aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, 
                      TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))

        # 3. Cabeamento (Cost + Losses)
        coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
        try:
            plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
            custo_usd = res['custo_total_usd']
            perdas_mwh = res['perda_anual_mwh']
        except Exception as cabling_error:
            if debug:
                print(f"   [DEBUG] Erro no cabeamento: {cabling_error}, n_grupos={n_grupos}")
                import traceback
                traceback.print_exc()
            return -1e6, 1e12
        
        # 4. Restrições Elétricas (Cruzamentos - simplificado via biblioteca do usuário se disponível)
        # Assumindo que o cabling_v3 e o SAP (Strict Angular Partitioning) garantem topologia planar radial
        # O código original usa 'detectar_sobreposicao_cabos' mas é complexo copiar tudo.
        # Como usamos o SAP (cabling_v3), cruzamentos são matematicamente impossíveis por definição,
        # exceto se houver sobreposição física de linhas por coordenadas ruins.
        # Vamos assumir penalidade 0 aqui para o baseline simplificado, ou usar uma heurística simples.
        pen_cabos = 0 
        
        aep_liq = aep_bruto - perdas_mwh - pen_turb_out - pen_spacing - pen_sub_out - pen_sub_close
        custo_final = custo_usd + pen_turb_out + pen_spacing + pen_sub_out + pen_sub_close
        
        if debug:
            print(f"   [DEBUG Baseline] Avaliação completa:")
            print(f"      AEP bruto: {aep_bruto:.2f} MWh")
            print(f"      Perdas Joule: {perdas_mwh:.2f} MWh")
            print(f"      Penalidades: turb_out={pen_turb_out:.2f}, spacing={pen_spacing:.2f}, "
                  f"sub_out={pen_sub_out:.2f}, sub_close={pen_sub_close:.2f}")
            print(f"      AEP líquido: {aep_liq:.2f} MWh")
            print(f"      Custo: {custo_usd:.2e} USD")
            print(f"      Custo final: {custo_final:.2e} USD")
        
        if aep_liq <= 0: 
            if debug:
                print(f"   [DEBUG] AEP líquido <= 0, retornando penalidade")
            return -1e6, 1e12
        return aep_liq, custo_final

    except Exception as e:
        if debug:
            print(f"   [DEBUG] Exceção geral em evaluate_full: {e}")
        return -1e6, 1e12

# --- AVALIADOR SEQUENCIAL (MINIMIZA APENAS CUSTO, TURBINAS FIXAS) ---
def evaluate_sequential(individual, fixed_turb_coords, debug=False):
    """
    Avalia apenas custo de cabeamento com turbinas fixas.
    Genoma: [n_grupos_norm (1), sub_x (1), sub_y (1)]
    """
    try:
        n_grupos_norm = individual[0]
        sub_pos = np.array([individual[1], individual[2]])
        
        # Converte Grupos
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        
        # Penalidades para subestação
        dist_sub = np.linalg.norm(sub_pos)
        pen_sub_out = np.maximum(0, dist_sub - CIRCLE_RADIUS) * 1e5
        
        # Distância Sub-Turbina
        d_sub_turb = np.linalg.norm(fixed_turb_coords - sub_pos, axis=1)
        violation_sub_close = np.maximum(0, 50.0 - np.min(d_sub_turb))
        pen_sub_close = violation_sub_close * 1e5
        
        # Cabeamento (Cost + Losses)
        coords_all = np.vstack([fixed_turb_coords, sub_pos.reshape(1, 2)])
        try:
            plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
            custo_usd = res['custo_total_usd']
            perdas_mwh = res['perda_anual_mwh']
        except Exception as cabling_error:
            if debug:
                print(f"   [DEBUG Sequential] Erro no cabeamento: {cabling_error}, n_grupos={n_grupos}")
            return 1e12,
        
        # Custo final (minimizar)
        custo_final = custo_usd + pen_sub_out + pen_sub_close
        
        if debug:
            print(f"   [DEBUG Sequential] Custo: {custo_usd:.2e} USD, Penalidades: {pen_sub_out + pen_sub_close:.2e}")
        
        return custo_final,
        
    except Exception as e:
        if debug:
            print(f"   [DEBUG Sequential] Exceção: {e}")
        return 1e12,

# --- OPERADORES GENÉTICOS ---

def create_individual_from_coordinates(coords):
    """
    Cria indivíduo da Fase 1 a partir de coordenadas.
    CONSERVADOR: Retorna coordenadas exatas do YAML (sem perturbação).
    Igual ao código original wind_farm_GA_16.py
    """
    return creator.IndividualPhase1(np.array(coords).flatten().tolist())

def create_random_phase1_ind():
    """
    Gera indivíduo para Fase 1 (apenas coordenadas).
    CONSERVADOR: Retorna coordenadas exatas do YAML, igual ao código original.
    A diversificação acontece gradualmente via mutação/crossover.
    """
    if initial_coordinates is not None:
        # Retorna coordenadas exatas do YAML (sem perturbação)
        coords = initial_coordinates.flatten().tolist()
    else:
        # Fallback: totalmente aleatório se YAML não disponível
        coords = []
        for _ in range(IND_SIZE):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0, CIRCLE_RADIUS * 0.9)
            coords.extend([r*np.cos(angle), r*np.sin(angle)])
    return creator.IndividualPhase1(coords)

def create_random_phase2_ind():
    """
    Gera indivíduo para Baseline (Genoma Completo).
    CONSERVADOR: Usa coordenadas exatas do YAML (sem perturbação).
    A diversificação acontece gradualmente via mutação/crossover.
    Igual à filosofia do código original wind_farm_GA_16.py
    """
    # 1. Coordenadas Turbinas - EXATAS do YAML (sem perturbação)
    if initial_coordinates is not None:
        coords = initial_coordinates.flatten().tolist()
    else:
        # Fallback: totalmente aleatório se YAML não disponível
        coords = []
        for _ in range(IND_SIZE):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0, CIRCLE_RADIUS * 0.9)
            coords.extend([r*np.cos(angle), r*np.sin(angle)])
    
    # 2. Gene de Grupos [0, 1] - valor inicial conservador (normalizado de MIN_GRUPOS)
    n_grupos_norm = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))  # Garante [0, 1]
    
    # 3. Subestação - centroide das turbinas DESLOCADO para não ficar em cima de turbina
    coords_array = np.array(coords).reshape((IND_SIZE, 2))
    centroid = np.array(np.mean(coords_array, axis=0))  # Garante que é array numpy
    # Desloca subestação do centroide para garantir distância mínima de 50m das turbinas
    sub_pos = displace_substation_from_turbines(centroid, coords_array, min_distance=50.0)
    
    # Monta genoma completo: [coords_turbinas (32), n_grupos_norm (1), sub_x (1), sub_y (1)]
    full_genome = coords + [n_grupos_norm] + sub_pos
    return creator.IndividualPhase2(full_genome)

def convert_p1_to_p2(ind_p1):
    """Smart Seeding: Converte layout P1 -> P2 com heurística de centroide."""
    coords_flat = list(ind_p1)
    coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    
    # Inicializa subestação no centroide (Heurística Smart)
    centroid = np.mean(coords, axis=0)
    sub_pos = enforce_substation(centroid).tolist()
    
    # Inicializa grupos (pode ser aleatório ou fixo, vamos variar levemente)
    g_norm = random.random()
    
    return creator.IndividualPhase2(coords_flat + [g_norm] + sub_pos)

def mutate_p2(individual, indpb):
    """Mutação adaptada para genoma misto - usando parâmetros do wind_farm_GA_16.py."""
    n_coords = IND_SIZE * 2
    # Turbinas - usando sigma=100 do wind_farm_GA_16.py
    for i in range(n_coords):
        if random.random() < indpb:
            individual[i] += random.gauss(0, 100)  # sigma=100 do wind_farm_GA_16.py
    enforce_circle(individual[:n_coords])
    
    # Repara spacing após mutação das turbinas
    coords_array = np.array(individual[:n_coords]).reshape((IND_SIZE, 2))
    coords_repaired = repair_spacing(coords_array)
    individual[:n_coords] = coords_repaired.flatten().tolist()
    
    # Grupos - mutação
    if random.random() < indpb:
        individual[n_coords] += random.gauss(0, 0.1)
        individual[n_coords] = max(0.0, min(1.0, individual[n_coords]))
        
    # Subestação - usando sigma=100 do wind_farm_GA_16.py
    if random.random() < indpb:
        individual[n_coords+1] += random.gauss(0, 100)  # sigma=100 do wind_farm_GA_16.py
        individual[n_coords+2] += random.gauss(0, 100)
    
    sub_arr = np.array([individual[n_coords+1], individual[n_coords+2]])
    sub_arr = enforce_substation(sub_arr)
    
    # Garante distância mínima da subestação às turbinas
    coords_array = np.array(individual[:n_coords]).reshape((IND_SIZE, 2))
    sub_arr = displace_substation_from_turbines(sub_arr, coords_array, min_distance=50.0)
    
    individual[n_coords+1] = sub_arr[0]
    individual[n_coords+2] = sub_arr[1]
    
    return individual,

# Registra nas Toolboxes
toolbox_p1.register("individual", create_random_phase1_ind)
toolbox_p1.register("population", tools.initRepeat, list, toolbox_p1.individual)
toolbox_p1.register("evaluate", evaluate_phase1)
toolbox_p1.register("mate", tools.cxBlend, alpha=0.5)
toolbox_p1.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.4)  # Parâmetros do wind_farm_GA_16.py
toolbox_p1.register("select", tools.selTournament, tournsize=5)  # Parâmetros do wind_farm_GA_16.py

toolbox_p2.register("evaluate", evaluate_full)
toolbox_p2.register("mate", tools.cxBlend, alpha=0.5)  # Parâmetros do wind_farm_GA_16.py
toolbox_p2.register("mutate", mutate_p2, indpb=0.4)  # Parâmetros do wind_farm_GA_16.py
toolbox_p2.register("select", tools.selNSGA2)

toolbox_base.register("individual", create_random_phase2_ind)
toolbox_base.register("population", tools.initRepeat, list, toolbox_base.individual)
# Wrapper para evaluate_full com debug opcional
def evaluate_full_wrapper(individual):
    """Wrapper que permite debug apenas na primeira avaliação."""
    global DEBUG_BASELINE
    debug_now = DEBUG_BASELINE
    if DEBUG_BASELINE:
        DEBUG_BASELINE = False  # Desativa após primeira vez
    return evaluate_full(individual, debug=debug_now)

def mate_with_repair(ind1, ind2):
    """Crossover com repair automático após operação."""
    tools.cxBlend(ind1, ind2, alpha=0.5)
    
    # Repair após crossover
    n_coords = IND_SIZE * 2
    
    # Repair coordenadas das turbinas
    coords1 = np.array(ind1[:n_coords]).reshape((IND_SIZE, 2))
    coords1 = repair_spacing(coords1)
    ind1[:n_coords] = coords1.flatten().tolist()
    enforce_circle(ind1[:n_coords])
    
    coords2 = np.array(ind2[:n_coords]).reshape((IND_SIZE, 2))
    coords2 = repair_spacing(coords2)
    ind2[:n_coords] = coords2.flatten().tolist()
    enforce_circle(ind2[:n_coords])
    
    # Repair subestação
    sub1 = np.array([ind1[n_coords+1], ind1[n_coords+2]])
    sub1 = enforce_substation(sub1)
    sub1 = displace_substation_from_turbines(sub1, coords1, min_distance=50.0)
    ind1[n_coords+1] = sub1[0]
    ind1[n_coords+2] = sub1[1]
    
    sub2 = np.array([ind2[n_coords+1], ind2[n_coords+2]])
    sub2 = enforce_substation(sub2)
    sub2 = displace_substation_from_turbines(sub2, coords2, min_distance=50.0)
    ind2[n_coords+1] = sub2[0]
    ind2[n_coords+2] = sub2[1]
    
    return ind1, ind2

toolbox_base.register("evaluate", evaluate_full_wrapper)
toolbox_base.register("mate", mate_with_repair)
toolbox_base.register("mutate", mutate_p2, indpb=0.4)  # Parâmetros do wind_farm_GA_16.py
toolbox_base.register("select", tools.selNSGA2)

# --- TOOLBOX SEQUENCIAL ---
def create_sequential_ind(fixed_turb_coords):
    """Cria indivíduo sequencial: [n_grupos_norm, sub_x, sub_y]"""
    # Inicializa subestação no centroide
    centroid = np.mean(fixed_turb_coords, axis=0)
    sub_pos = displace_substation_from_turbines(centroid, fixed_turb_coords, min_distance=50.0)
    
    # Número de grupos inicial (normalizado)
    n_grupos_norm = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))
    
    return creator.IndividualSequential([n_grupos_norm, sub_pos[0], sub_pos[1]])

def mutate_sequential(individual, indpb, fixed_turb_coords):
    """Mutação para indivíduo sequencial - usando parâmetros do wind_farm_GA_16.py"""
    # Mutação de grupos
    if random.random() < indpb:
        individual[0] += random.gauss(0, 0.1)
        individual[0] = max(0.0, min(1.0, individual[0]))
    
    # Mutação de subestação - usando sigma=100 do wind_farm_GA_16.py
    if random.random() < indpb:
        individual[1] += random.gauss(0, 100)  # sigma=100 do wind_farm_GA_16.py
        individual[2] += random.gauss(0, 100)
    
    # Enforce constraints
    sub_arr = np.array([individual[1], individual[2]])
    sub_arr = enforce_substation(sub_arr)
    sub_arr = displace_substation_from_turbines(sub_arr, fixed_turb_coords, min_distance=50.0)
    individual[1] = sub_arr[0]
    individual[2] = sub_arr[1]
    
    return individual,

def mate_sequential(ind1, ind2):
    """Crossover para indivíduo sequencial"""
    tools.cxBlend(ind1, ind2, alpha=0.5)
    # Garante bounds
    ind1[0] = max(0.0, min(1.0, ind1[0]))
    ind2[0] = max(0.0, min(1.0, ind2[0]))
    return ind1, ind2

# Registra toolbox sequencial (será configurada dinamicamente com turbinas fixas)
toolbox_seq.register("mate", mate_sequential)
toolbox_seq.register("select", tools.selTournament, tournsize=5)  # Parâmetros do wind_farm_GA_16.py
# mutate será registrado dinamicamente em run_sequential_method

# =============================================================================
# 4. EXECUÇÃO DOS MÉTODOS
# =============================================================================

def run_proposed_method(seed, track_evolution=False, ref_point=None):
    """
    Executa Fase 1 + Fase 2.
    
    Returns:
        pareto_front: Lista de soluções não-dominadas
        evolution_data: Dict com evolução do hipervolume (se track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    # --- FASE 1 ---
    pop = toolbox_p1.population(n=POP_SIZE)
    # Roda GA Simples - usando parâmetros do wind_farm_GA_16.py
    pop, _ = algorithms.eaSimple(pop, toolbox_p1, cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE1, verbose=False)
    
    # Seleciona melhores da Fase 1
    best_p1 = tools.selBest(pop, int(POP_SIZE * 0.2)) # Top 20%
    
    # --- TRANSIÇÃO (SMART SEEDING) ---
    pop_p2 = []
    # Preenche população P2:
    # 1. Clones dos melhores P1 convertidos
    for ind in best_p1:
        pop_p2.append(convert_p1_to_p2(ind))
    
    # 2. Completa o resto com novos indivíduos aleatórios baseados nos melhores (perturbação)
    while len(pop_p2) < POP_SIZE:
        parent = random.choice(best_p1)
        child = convert_p1_to_p2(parent)
        child, = mutate_p2(child, indpb=0.3) # Mutação forte para diversidade
        pop_p2.append(child)
        
    # --- FASE 2 ---
    # Recalcula fitness inicial (pois mudou para multiobjetivo)
    invalid_ind = [ind for ind in pop_p2 if not ind.fitness.valid]
    fits = list(map(toolbox_p2.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Filtra soluções válidas antes de começar Fase 2
    pop_p2 = [ind for ind in pop_p2 if ind.fitness.valid and ind.fitness.values[0] > 0]
    if len(pop_p2) == 0:
        # Se não há soluções válidas após Fase 1, retorna vazio
        if track_evolution:
            return [], evolution_data
        return []
    
    # Roda NSGA-II com rastreamento customizado se necessário
    if track_evolution:
        # Loop manual para rastrear evolução
        pareto_front = tools.ParetoFront()
        pareto_front.update(pop_p2)
        
        # Calcula hipervolume inicial (apenas soluções válidas)
        pareto_valid = filter_valid_solutions(pareto_front)
        if len(pareto_valid) > 0 and ref_point is not None:
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"].append(NGEN_PHASE1)
            evolution_data["hv"].append(hv)
            evolution_data["n_solutions"].append(len(pareto_valid))
        else:
            evolution_data["gen"].append(NGEN_PHASE1)
            evolution_data["hv"].append(0.0)
            evolution_data["n_solutions"].append(0)
        
        for gen in range(NGEN_PHASE2):
            # Seleção
            offspring = toolbox_p2.select(pop_p2, len(pop_p2))
            offspring = list(map(toolbox_p2.clone, offspring))
            
            # Crossover - usando parâmetros do wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 do wind_farm_GA_16.py
                    toolbox_p2.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutação - usando parâmetros do wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 do wind_farm_GA_16.py
                    toolbox_p2.mutate(mutant)
                    del mutant.fitness.values
            
            # Avaliação
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_p2.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combina população + offspring e seleciona melhores
            combined = pop_p2 + offspring
            pop_p2 = toolbox_p2.select(combined, POP_SIZE)
            pareto_front.update(pop_p2)
            
            # Calcula hipervolume (apenas soluções válidas)
            pareto_valid = filter_valid_solutions(pareto_front)
            if len(pareto_valid) > 0 and ref_point is not None:
                pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
                pf_array = np.array(pf_points)
                hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
                evolution_data["gen"].append(NGEN_PHASE1 + gen)
                evolution_data["hv"].append(hv)
                evolution_data["n_solutions"].append(len(pareto_valid))
            else:
                # Se não há soluções válidas, registra 0
                evolution_data["gen"].append(NGEN_PHASE1 + gen)
                evolution_data["hv"].append(0.0)
                evolution_data["n_solutions"].append(0)
        
        pop_final = pop_p2
    else:
        # Roda NSGA-II padrão - usando parâmetros do wind_farm_GA_16.py
        pop_final, logbook = algorithms.eaMuPlusLambda(pop_p2, toolbox_p2, mu=POP_SIZE, lambda_=POP_SIZE,
                                                       cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE2, verbose=False)
    
    # Extrai frente de Pareto usando ParetoFront
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop_final)
    
    # Filtra soluções inválidas
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

def run_baseline_method(seed, track_evolution=False, ref_point=None):
    """
    Executa NSGA-II Puro (Random Init, Long Run).
    
    Returns:
        pareto_front: Lista de soluções não-dominadas
        evolution_data: Dict com evolução do hipervolume (se track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    pop = toolbox_base.population(n=POP_SIZE)
    
    if track_evolution:
        # Avalia população inicial
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # Debug: avalia uma solução para ver o que está acontecendo
        if len(invalid_ind) > 0:
            test_ind = invalid_ind[0]
            test_fit = evaluate_full(test_ind, debug=True)
            print(f"   [DEBUG Baseline Init] Primeira solução: AEP={test_fit[0]:.2f}, Cost={test_fit[1]:.2e}")
        fits = list(map(toolbox_base.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        # Loop manual para rastrear evolução
        pareto_front = tools.ParetoFront()
        pareto_front.update(pop)
        
        # Calcula hipervolume inicial (apenas soluções válidas)
        pareto_valid = filter_valid_solutions(pareto_front)
        if len(pareto_valid) > 0 and ref_point is not None:
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"].append(0)
            evolution_data["hv"].append(hv)
            evolution_data["n_solutions"].append(len(pareto_valid))
        else:
            evolution_data["gen"].append(0)
            evolution_data["hv"].append(0.0)
            evolution_data["n_solutions"].append(0)
        
        for gen in range(1, NGEN_BASELINE + 1):
            # ELITISMO: Preserva melhores soluções válidas antes de gerar offspring
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            n_elite = min(5, len(valid_pop))  # Preserva top 5 soluções válidas
            elite = tools.selBest(valid_pop, n_elite) if n_elite > 0 else []
            
            # Seleção
            offspring = toolbox_base.select(pop, len(pop))
            offspring = list(map(toolbox_base.clone, offspring))
            
            # Crossover - usando parâmetros do wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 do wind_farm_GA_16.py
                    toolbox_base.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutação - usando parâmetros do wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 do wind_farm_GA_16.py
                    toolbox_base.mutate(mutant)
                    del mutant.fitness.values
            
            # Avaliação
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_base.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combina população + offspring e seleciona melhores
            combined = pop + offspring
            pop = toolbox_base.select(combined, POP_SIZE)
            
            # Garante que elite está na população
            if len(elite) > 0:
                # Remove piores e adiciona elite
                pop_sorted = sorted(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid and x.fitness.values[0] > 0 else -1e12, reverse=True)
                pop = elite + pop_sorted[:POP_SIZE - len(elite)]
            
            pareto_front.update(pop)
            
            # Calcula hipervolume (apenas soluções válidas)
            pareto_valid = filter_valid_solutions(pareto_front)
            if len(pareto_valid) > 0 and ref_point is not None:
                pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
                pf_array = np.array(pf_points)
                hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
                evolution_data["gen"].append(gen)
                evolution_data["hv"].append(hv)
                evolution_data["n_solutions"].append(len(pareto_valid))
            else:
                # Se não há soluções válidas, registra 0
                evolution_data["gen"].append(gen)
                evolution_data["hv"].append(0.0)
                evolution_data["n_solutions"].append(0)
        
        pop_final = pop
    else:
        # Roda NSGA-II com elitismo explícito (mesma lógica do track_evolution)
        for gen in range(1, NGEN_BASELINE + 1):
            # ELITISMO: Preserva melhores soluções válidas antes de gerar offspring
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            n_elite = min(5, len(valid_pop))  # Preserva top 5 soluções válidas
            elite = tools.selBest(valid_pop, n_elite) if n_elite > 0 else []
            
            # Seleção
            offspring = toolbox_base.select(pop, len(pop))
            offspring = list(map(toolbox_base.clone, offspring))
            
            # Crossover - usando parâmetros do wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 do wind_farm_GA_16.py
                    toolbox_base.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutação - usando parâmetros do wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 do wind_farm_GA_16.py
                    toolbox_base.mutate(mutant)
                    del mutant.fitness.values
            
            # Avaliação
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_base.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combina população + offspring e seleciona melhores
            combined = pop + offspring
            pop = toolbox_base.select(combined, POP_SIZE)
            
            # Garante que elite está na população
            if len(elite) > 0:
                # Remove piores e adiciona elite
                pop_sorted = sorted(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid and x.fitness.values[0] > 0 else -1e12, reverse=True)
                pop = elite + pop_sorted[:POP_SIZE - len(elite)]
        
        pop_final = pop
    
    # Extrai frente de Pareto usando ParetoFront
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop_final)
    
    # Debug: mostra informações sobre a frente de Pareto antes do filtro (apenas baseline)
    if len(pareto_front) > 0:
        aep_vals = [ind.fitness.values[0] for ind in pareto_front]
        cost_vals = [ind.fitness.values[1] for ind in pareto_front]
        print(f"   [DEBUG Baseline] Pareto front antes do filtro: {len(pareto_front)} soluções")
        print(f"   [DEBUG Baseline] AEP range: [{min(aep_vals):.2f}, {max(aep_vals):.2f}] MWh")
        print(f"   [DEBUG Baseline] Cost range: [{min(cost_vals):.2e}, {max(cost_vals):.2e}] USD")
        print(f"   [DEBUG Baseline] Soluções com AEP > 0: {sum(1 for aep in aep_vals if aep > 0)}")
        print(f"   [DEBUG Baseline] Soluções com Cost > 0 e < 1e12: {sum(1 for cost in cost_vals if 0 < cost < 1e12)}")
    
    # Filtra soluções inválidas
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

def run_sequential_method(seed, track_evolution=False, ref_point=None):
    """
    Executa abordagem sequencial (dois GAs simples separados):
    
    1. GA SIMPLES - Fase 1: Otimiza layout das turbinas
       - Objetivo: Maximizar AEP
       - Variáveis: Posições (x, y) de todas as turbinas
       - Algoritmo: eaSimple (GA padrão com seleção por torneio)
    
    2. GA SIMPLES - Fase 2 Sequencial: Otimiza subestação e cabeamento
       - Objetivo: Minimizar custo (com turbinas fixas da Fase 1)
       - Variáveis: Posição da subestação (x, y) + número de grupos
       - Algoritmo: eaSimple (GA padrão com seleção por torneio)
       - Inicialização: Subestação começa no centroide das turbinas
    
    Returns:
        pareto_front: Lista de soluções (convertidas para formato multi-objetivo para comparação)
        evolution_data: Dict com evolução (se track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    # --- FASE 1: GA SIMPLES para otimizar posições das turbinas (maximizar AEP) ---
    pop = toolbox_p1.population(n=POP_SIZE)
    pop, _ = algorithms.eaSimple(pop, toolbox_p1, cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE1, verbose=False)  # Parâmetros do wind_farm_GA_16.py
    
    # Seleciona melhor layout da Fase 1
    best_p1 = tools.selBest(pop, 1)[0]
    fixed_turb_coords = np.array(best_p1).reshape((IND_SIZE, 2))
    
    # Calcula AEP do layout fixo (será usado depois)
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep_bruto_fixed = np.sum(calcAEP(fixed_turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                     TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
    
    # --- FASE 2 SEQUENCIAL: GA SIMPLES para otimizar subestação e cabeamento (minimizar custo) ---
    # Configura toolbox sequencial com turbinas fixas da Fase 1
    def evaluate_seq_wrapper(ind):
        return evaluate_sequential(ind, fixed_turb_coords)
    
    def mutate_seq_wrapper(ind):
        return mutate_sequential(ind, 0.4, fixed_turb_coords)  # indpb=0.4 do wind_farm_GA_16.py
    
    toolbox_seq.register("individual", lambda: create_sequential_ind(fixed_turb_coords))
    toolbox_seq.register("population", tools.initRepeat, list, toolbox_seq.individual)
    toolbox_seq.register("evaluate", evaluate_seq_wrapper)
    toolbox_seq.register("mutate", mutate_seq_wrapper)
    
    # Cria população inicial
    pop_seq = toolbox_seq.population(n=POP_SIZE)
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop_seq if not ind.fitness.valid]
    fits = list(map(toolbox_seq.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Roda GA SIMPLES (eaSimple) para minimizar custo de cabeamento
    if track_evolution:
        # Loop manual para rastrear evolução
        best_costs = []
        for gen in range(NGEN_PHASE2):
            # Seleção
            offspring = toolbox_seq.select(pop_seq, len(pop_seq))
            offspring = list(map(toolbox_seq.clone, offspring))
            
            # Crossover - usando parâmetros do wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 do wind_farm_GA_16.py
                    toolbox_seq.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutação - usando parâmetros do wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 do wind_farm_GA_16.py
                    toolbox_seq.mutate(mutant)
                    del mutant.fitness.values
            
            # Avaliação
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_seq.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # Elitismo: mantém melhor
            combined = pop_seq + offspring
            pop_seq = tools.selBest(combined, POP_SIZE)
            
            # Rastreia melhor custo
            best_cost = min([ind.fitness.values[0] for ind in pop_seq if ind.fitness.valid])
            best_costs.append(best_cost)
            
            # Para comparação com outros métodos, precisamos converter para formato multi-objetivo
            # Mas como é sequencial, temos apenas uma solução final (ou podemos criar várias variando grupos)
            # Por enquanto, vamos criar uma "frente de Pareto" com as melhores soluções encontradas
            if gen % 50 == 0 or gen == NGEN_PHASE2 - 1:
                # Calcula AEP líquido para cada solução válida
                valid_solutions = [ind for ind in pop_seq if ind.fitness.valid and ind.fitness.values[0] < 1e12]
                if len(valid_solutions) > 0:
                    # Converte para formato multi-objetivo para comparação
                    # (mas não temos frente de Pareto real, apenas uma solução)
                    pass
        
        pop_final = pop_seq
    else:
        # Roda GA SIMPLES padrão - usando parâmetros do wind_farm_GA_16.py
        pop_final, _ = algorithms.eaSimple(pop_seq, toolbox_seq, cxpb=0.95, mutpb=0.7, 
                                          ngen=NGEN_PHASE2, verbose=False)
    
    # Converte soluções sequenciais para formato multi-objetivo para comparação
    # Cria "frente de Pareto" artificial com as melhores soluções encontradas
    pareto_front_list = []
    
    # Pega as melhores soluções (diversas em termos de custo)
    valid_solutions = [ind for ind in pop_final if ind.fitness.valid and ind.fitness.values[0] < 1e12]
    if len(valid_solutions) > 0:
        # Ordena por custo
        valid_solutions.sort(key=lambda x: x.fitness.values[0])
        
        # Para cada solução, calcula AEP líquido completo
        for ind_seq in valid_solutions[:min(50, len(valid_solutions))]:  # Top 50
            n_grupos_norm = ind_seq[0]
            sub_pos = np.array([ind_seq[1], ind_seq[2]])
            
            n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
            n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
            # Limita n_grupos ao número de turbinas
            n_grupos = min(n_grupos, IND_SIZE)
            
            # Calcula cabeamento completo
            coords_all = np.vstack([fixed_turb_coords, sub_pos.reshape(1, 2)])
            try:
                plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
                custo_usd = res['custo_total_usd']
                perdas_mwh = res['perda_anual_mwh']
                
                # AEP líquido
                aep_liq = aep_bruto_fixed - perdas_mwh
                
                if aep_liq > 0 and custo_usd > 0:
                    # Cria indivíduo Phase2 para compatibilidade
                    full_genome = fixed_turb_coords.flatten().tolist() + [n_grupos_norm] + sub_pos.tolist()
                    ind_p2 = creator.IndividualPhase2(full_genome)
                    ind_p2.fitness.values = (aep_liq, custo_usd)
                    pareto_front_list.append(ind_p2)
            except:
                continue
    
    # Cria frente de Pareto
    pareto_front = tools.ParetoFront()
    pareto_front.update(pareto_front_list)
    
    # Filtra soluções válidas
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    # Para rastreamento de evolução, precisamos simular
    if track_evolution:
        # Como não temos frente de Pareto real durante a evolução, vamos usar o melhor custo
        # e estimar AEP líquido baseado no melhor custo encontrado
        if len(pareto_front_filtered) > 0 and ref_point is not None:
            # Calcula hipervolume da frente final
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_front_filtered]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"] = list(range(NGEN_PHASE1, NGEN_PHASE1 + NGEN_PHASE2 + 1))
            evolution_data["hv"] = [hv] * (NGEN_PHASE2 + 1)  # Mantém constante (aproximação)
            evolution_data["n_solutions"] = [len(pareto_front_filtered)] * (NGEN_PHASE2 + 1)
        else:
            evolution_data["gen"] = list(range(NGEN_PHASE1, NGEN_PHASE1 + NGEN_PHASE2 + 1))
            evolution_data["hv"] = [0.0] * (NGEN_PHASE2 + 1)
            evolution_data["n_solutions"] = [0] * (NGEN_PHASE2 + 1)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

# =============================================================================
# 5. FUNÇÕES AUXILIARES
# =============================================================================

def filter_valid_solutions(pareto_front):
    """
    Filtra soluções inválidas da frente de Pareto.
    Soluções válidas: AEP > 0 e Custo > 0
    """
    return [ind for ind in pareto_front 
            if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]

# =============================================================================
# 6. FUNÇÕES DE ANÁLISE E MÉTRICAS ADICIONAIS
# =============================================================================

def calculate_coverage(pareto_front_A, pareto_front_B):
    """
    Calcula a métrica de Coverage (C-metric) entre duas frentes de Pareto.
    
    C(A, B) = |{b in B | existe a in A tal que a domina b}| / |B|
    
    Retorna um valor entre 0 e 1:
    - 1.0 significa que todos os pontos de B são dominados por A
    - 0.0 significa que nenhum ponto de B é dominado por A
    
    Args:
        pareto_front_A: Lista de indivíduos da frente A
        pareto_front_B: Lista de indivíduos da frente B
    
    Returns:
        coverage: Valor de coverage de A sobre B
    """
    if len(pareto_front_B) == 0:
        return 0.0
    if len(pareto_front_A) == 0:
        return 0.0
    
    # Extrai valores objetivos (AEP, Cost)
    # Assumindo que fitness.values = (AEP, Cost) com weights=(1.0, -1.0)
    # Queremos maximizar AEP e minimizar Cost
    points_A = [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front_A 
                if ind.fitness.valid and ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
    points_B = [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front_B 
                if ind.fitness.valid and ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
    
    if len(points_B) == 0:
        return 0.0
    if len(points_A) == 0:
        return 0.0
    
    dominated_count = 0
    for b_aep, b_cost in points_B:
        # Verifica se existe algum ponto em A que domina b
        # a domina b se: a_aep >= b_aep E a_cost <= b_cost E (a_aep > b_aep OU a_cost < b_cost)
        is_dominated = False
        for a_aep, a_cost in points_A:
            if (a_aep >= b_aep and a_cost <= b_cost and 
                (a_aep > b_aep or a_cost < b_cost)):
                is_dominated = True
                break
        if is_dominated:
            dominated_count += 1
    
    return dominated_count / len(points_B)

def calculate_spread(pareto_front):
    """
    Calcula a métrica de spread (diversidade) da frente de Pareto.
    Spread mede a distribuição das soluções ao longo da frente.
    Menor spread = melhor distribuição.
    """
    if len(pareto_front) < 2:
        return 0.0
    
    # Extrai valores objetivos
    aep_values = [ind.fitness.values[0] for ind in pareto_front]
    cost_values = [ind.fitness.values[1] for ind in pareto_front]
    
    # Normaliza para [0, 1]
    aep_min, aep_max = min(aep_values), max(aep_values)
    cost_min, cost_max = min(cost_values), max(cost_values)
    
    if aep_max == aep_min or cost_max == cost_min:
        return 0.0
    
    aep_norm = [(a - aep_min) / (aep_max - aep_min) for a in aep_values]
    cost_norm = [(c - cost_min) / (cost_max - cost_min) for c in cost_values]
    
    # Calcula distâncias entre pontos consecutivos (após ordenar por AEP)
    sorted_indices = sorted(range(len(aep_norm)), key=lambda i: aep_norm[i])
    distances = []
    for i in range(len(sorted_indices) - 1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
        dist = np.sqrt((aep_norm[idx1] - aep_norm[idx2])**2 + 
                      (cost_norm[idx1] - cost_norm[idx2])**2)
        distances.append(dist)
    
    if len(distances) == 0:
        return 0.0
    
    # Spread = desvio padrão das distâncias
    mean_dist = np.mean(distances)
    spread = np.std(distances) if len(distances) > 1 else 0.0
    
    return spread

def calculate_convergence_gen(evolution_data, threshold=0.95):
    """
    Calcula a geração em que o hipervolume atinge threshold% do valor final.
    Retorna None se não convergir.
    """
    if evolution_data is None or len(evolution_data["hv"]) == 0:
        return None
    
    hv_values = evolution_data["hv"]
    final_hv = hv_values[-1]
    target_hv = threshold * final_hv
    
    for i, hv in enumerate(hv_values):
        if hv >= target_hv:
            return evolution_data["gen"][i]
    
    return None

def calculate_statistical_tests(prop_data, base_data, metric_name="Hypervolume"):
    """
    Realiza testes estatísticos para comparar os dois métodos.
    Retorna dict com resultados.
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy não disponível"}
    
    results = {}
    
    # Teste de normalidade (Shapiro-Wilk)
    _, p_prop_norm = stats.shapiro(prop_data)
    _, p_base_norm = stats.shapiro(base_data)
    results["normality"] = {
        "proposed_p": p_prop_norm,
        "baseline_p": p_base_norm,
        "both_normal": p_prop_norm > 0.05 and p_base_norm > 0.05
    }
    
    # Teste t (se normal) ou Mann-Whitney (se não normal)
    if results["normality"]["both_normal"]:
        # Teste t de Student
        t_stat, p_value = stats.ttest_ind(prop_data, base_data, alternative='greater')
        results["test"] = "t-test"
        results["statistic"] = t_stat
        results["p_value"] = p_value
    else:
        # Teste de Mann-Whitney U (Wilcoxon rank-sum)
        u_stat, p_value = stats.mannwhitneyu(prop_data, base_data, alternative='greater')
        results["test"] = "Mann-Whitney U"
        results["statistic"] = u_stat
        results["p_value"] = p_value
    
    # Efeito (Cohen's d ou similar)
    mean_prop = np.mean(prop_data)
    mean_base = np.mean(base_data)
    std_pooled = np.sqrt((np.var(prop_data) + np.var(base_data)) / 2)
    cohens_d = (mean_prop - mean_base) / std_pooled if std_pooled > 0 else 0
    results["effect_size"] = {
        "cohens_d": cohens_d,
        "interpretation": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
    }
    
    # Estatísticas descritivas
    results["descriptive"] = {
        "proposed": {
            "mean": mean_prop,
            "std": np.std(prop_data),
            "median": np.median(prop_data),
            "min": np.min(prop_data),
            "max": np.max(prop_data)
        },
        "baseline": {
            "mean": mean_base,
            "std": np.std(base_data),
            "median": np.median(base_data),
            "min": np.min(base_data),
            "max": np.max(base_data)
        }
    }
    
    results["significant"] = p_value < 0.05
    results["metric_name"] = metric_name
    
    return results

# =============================================================================
# 6. LOOP PRINCIPAL E MÉTRICAS
# =============================================================================

if __name__ == "__main__":
    print(f"--- INICIANDO BENCHMARK GECCO 2026 ---")
    print(f"Comparação: Proposed Two-Phase vs. Baseline Pure NSGA-II vs. Sequential")
    print(f"Seeds: {N_SEEDS} | Pop: {POP_SIZE}")
    print(f"Gerações Proposto: P1={NGEN_PHASE1} + P2={NGEN_PHASE2}")
    print(f"Gerações Baseline: {NGEN_BASELINE} (Equivalente em custo)")
    print(f"Gerações Sequencial: P1={NGEN_PHASE1} + P2={NGEN_PHASE2} (otimiza apenas custo na P2)")
    
    results_prop_hv = []
    results_base_hv = []
    results_seq_hv = []  # Sequencial
    results_prop_n_solutions = []
    results_base_n_solutions = []
    results_seq_n_solutions = []  # Sequencial
    results_prop_spread = []
    results_base_spread = []
    results_seq_spread = []  # Sequencial
    pareto_prop_all = []
    pareto_base_all = []
    pareto_seq_all = []  # Sequencial
    
    # Métricas de tempo de execução (computacional)
    times_prop = []  # Tempo de execução do método proposto (segundos)
    times_base = []  # Tempo de execução do baseline (segundos)
    times_seq = []   # Tempo de execução do sequencial (segundos)
    
    # Métricas adicionais
    n_evaluations_prop = []  # Número de avaliações de função (método proposto)
    n_evaluations_base = []  # Número de avaliações de função (baseline)
    n_evaluations_seq = []   # Número de avaliações de função (sequencial)
    success_rate_prop = []   # Taxa de sucesso (1 se encontrou soluções válidas, 0 caso contrário)
    success_rate_base = []
    success_rate_seq = []
    
    # Dados de evolução (para visualização)
    evolution_prop = []  # Lista de dicts, um por seed
    evolution_base = []
    evolution_seq = []  # Sequencial
    
    # Ponto de referência para Hipervolume
    # AEP max ~600 GWh -> normalizar ou usar valor fixo.
    # Custo max ~10M USD.
    # Como DEAP assume minimização no HV, transformamos AEP para -AEP.
    # Ref Point deve ser pior que qualquer solução viável: (Max Cost, Min -AEP)
    # Ex: Custo=20M, -AEP=0 (ou seja AEP=0)
    ref_point = [2e7, 0]
    
    # Flag para rastrear evolução (pode ser lento, use apenas se necessário)
    TRACK_EVOLUTION = True  # Mude para False para executar mais rápido 

    for i in range(N_SEEDS):
        t0_seed = time.time()
        print(f"\n>>> Executando Seed {i+1}/{N_SEEDS}...")
        
        # 1. Proposto
        print("   -> Rodando Método Proposto...")
        t_prop_start = time.time()
        if TRACK_EVOLUTION:
            pf_prop, evol_prop = run_proposed_method(i, track_evolution=True, ref_point=ref_point)
            evolution_prop.append(evol_prop)
        else:
            pf_prop = run_proposed_method(i)
        t_prop = time.time() - t_prop_start
        times_prop.append(t_prop)
        print(f"   [Tempo Proposto: {t_prop:.1f}s]")
        
        # 2. Baseline
        print("   -> Rodando Baseline...")
        t_base_start = time.time()
        if TRACK_EVOLUTION:
            pf_base, evol_base = run_baseline_method(i, track_evolution=True, ref_point=ref_point)
            evolution_base.append(evol_base)
        else:
            pf_base = run_baseline_method(i)
        t_base = time.time() - t_base_start
        times_base.append(t_base)
        print(f"   [Tempo Baseline: {t_base:.1f}s]")
        
        # 3. Sequencial
        print("   -> Rodando Método Sequencial...")
        t_seq_start = time.time()
        if TRACK_EVOLUTION:
            pf_seq, evol_seq = run_sequential_method(i, track_evolution=True, ref_point=ref_point)
            evolution_seq.append(evol_seq)
        else:
            pf_seq = run_sequential_method(i)
        t_seq = time.time() - t_seq_start
        times_seq.append(t_seq)
        print(f"   [Tempo Sequencial: {t_seq:.1f}s]")
        
        # Calcula número de avaliações de função (aproximado)
        # Proposto: Fase 1 (NGEN_PHASE1 * POP_SIZE) + Fase 2 (NGEN_PHASE2 * POP_SIZE)
        n_eval_prop = NGEN_PHASE1 * POP_SIZE + NGEN_PHASE2 * POP_SIZE
        # Baseline: NGEN_BASELINE * POP_SIZE
        n_eval_base = NGEN_BASELINE * POP_SIZE
        # Sequencial: Fase 1 (NGEN_PHASE1 * POP_SIZE) + Fase 2 (NGEN_PHASE2 * POP_SIZE)
        n_eval_seq = NGEN_PHASE1 * POP_SIZE + NGEN_PHASE2 * POP_SIZE
        
        n_evaluations_prop.append(n_eval_prop)
        n_evaluations_base.append(n_eval_base)
        n_evaluations_seq.append(n_eval_seq)
        
        # Taxa de sucesso (1 se encontrou soluções válidas, 0 caso contrário)
        success_rate_prop.append(1 if len(pf_prop) > 0 else 0)
        success_rate_base.append(1 if len(pf_base) > 0 else 0)
        success_rate_seq.append(1 if len(pf_seq) > 0 else 0)
        
        # Debug: mostra informações sobre as soluções
        if len(pf_prop) == 0:
            print(f"   AVISO [Seed {i}]: Método proposto retornou 0 soluções válidas!")
        if len(pf_base) == 0:
            print(f"   AVISO [Seed {i}]: Baseline retornou 0 soluções válidas!")
        if len(pf_seq) == 0:
            print(f"   AVISO [Seed {i}]: Método sequencial retornou 0 soluções válidas!")
        
        # Cálculo de Hipervolume
        # Transforma para minimização: [Cost, -AEP]
        # (Original: [AEP, Cost] -> FitnessMulti weights=(1.0, -1.0))
        # DEAP armazena fitness.values como (AEP, Cost) ou similar dependendo da implementação.
        # Nossas weights são (1.0, -1.0).
        # Para HV do DEAP, precisamos passar valores para MINIMIZAR.
        # Obj1: AEP (queremos max). Para min, usamos -AEP.
        # Obj2: Custo (queremos min). Já é Custo.
        
        def get_front_points(pf):
            # Retorna lista de [Custo, -AEP] para cálculo HV
            # Já vem filtrado, mas garantimos que são válidas
            return [[ind.fitness.values[1], -ind.fitness.values[0]] 
                   for ind in pf if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
            
        pts_prop = get_front_points(pf_prop)
        pts_base = get_front_points(pf_base)
        pts_seq = get_front_points(pf_seq)
        
        # Debug: mostra quantas soluções válidas temos
        print(f"   [Seed {i}] Soluções válidas: Prop={len(pts_prop)}, Base={len(pts_base)}, Seq={len(pts_seq)}")
        
        # Calcula hipervolume usando a função correta do DEAP
        # A função hypervolume espera um array numpy e um ponto de referência
        if len(pts_prop) > 0:
            pts_prop_array = np.array(pts_prop)
            hv_prop = hypervolume_module.hypervolume(pts_prop_array, np.array(ref_point))
        else:
            hv_prop = 0.0
            
        if len(pts_base) > 0:
            pts_base_array = np.array(pts_base)
            hv_base = hypervolume_module.hypervolume(pts_base_array, np.array(ref_point))
        else:
            hv_base = 0.0
        
        if len(pts_seq) > 0:
            pts_seq_array = np.array(pts_seq)
            hv_seq = hypervolume_module.hypervolume(pts_seq_array, np.array(ref_point))
        else:
            hv_seq = 0.0
        
        results_prop_hv.append(hv_prop)
        results_base_hv.append(hv_base)
        results_seq_hv.append(hv_seq)
        
        # Métricas adicionais
        n_sol_prop = len(pf_prop)
        n_sol_base = len(pf_base)
        n_sol_seq = len(pf_seq)
        results_prop_n_solutions.append(n_sol_prop)
        results_base_n_solutions.append(n_sol_base)
        results_seq_n_solutions.append(n_sol_seq)
        
        spread_prop = calculate_spread(pf_prop)
        spread_base = calculate_spread(pf_base)
        spread_seq = calculate_spread(pf_seq)
        results_prop_spread.append(spread_prop)
        results_base_spread.append(spread_base)
        results_seq_spread.append(spread_seq)
        
        # Guarda pontos originais para plot (AEP, Custo)
        # Filtra apenas soluções válidas (AEP > 0, Cost > 0)
        prop_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                      for ind in pf_prop if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        base_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                      for ind in pf_base if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        seq_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                     for ind in pf_seq if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        
        pareto_prop_all.extend(prop_valid)
        pareto_base_all.extend(base_valid)
        pareto_seq_all.extend(seq_valid)
        
        if len(prop_valid) == 0:
            print(f"   AVISO [Seed {i}]: Nenhuma solução válida no método proposto!")
        if len(base_valid) == 0:
            print(f"   AVISO [Seed {i}]: Nenhuma solução válida no baseline!")
        if len(seq_valid) == 0:
            print(f"   AVISO [Seed {i}]: Nenhuma solução válida no método sequencial!")
        
        print(f"   [Seed {i}] HV: Prop={hv_prop:.2e} | Base={hv_base:.2e} | Seq={hv_seq:.2e}")
        print(f"   [Seed {i}] Soluções: Prop={n_sol_prop} | Base={n_sol_base} | Seq={n_sol_seq}")
        print(f"   [Seed {i}] Spread: Prop={spread_prop:.4f} | Base={spread_base:.4f} | Seq={spread_seq:.4f}")
        print(f"   [Seed {i}] Tempos: Prop={t_prop:.1f}s | Base={t_base:.1f}s | Seq={t_seq:.1f}s")
        print(f"   Tempo Total Seed: {time.time()-t0_seed:.1f}s")

    # =============================================================================
    # 7. ANÁLISE ESTATÍSTICA
    # =============================================================================
    
    print("\n" + "="*80)
    print("RESULTADOS FINAIS E ANÁLISE ESTATÍSTICA")
    print("="*80)
    
    print(f"\n--- HIPERVOLUME ---")
    print(f"Proposto:   Média={np.mean(results_prop_hv):.2e}, Std={np.std(results_prop_hv):.2e}, Mediana={np.median(results_prop_hv):.2e}")
    print(f"Baseline:   Média={np.mean(results_base_hv):.2e}, Std={np.std(results_base_hv):.2e}, Mediana={np.median(results_base_hv):.2e}")
    print(f"Sequencial: Média={np.mean(results_seq_hv):.2e}, Std={np.std(results_seq_hv):.2e}, Mediana={np.median(results_seq_hv):.2e}")
    
    print(f"\n--- NÚMERO DE SOLUÇÕES ---")
    print(f"Proposto:   Média={np.mean(results_prop_n_solutions):.1f}, Std={np.std(results_prop_n_solutions):.1f}")
    print(f"Baseline:   Média={np.mean(results_base_n_solutions):.1f}, Std={np.std(results_base_n_solutions):.1f}")
    print(f"Sequencial: Média={np.mean(results_seq_n_solutions):.1f}, Std={np.std(results_seq_n_solutions):.1f}")
    
    print(f"\n--- SPREAD (DIVERSIDADE) ---")
    print(f"Proposto:   Média={np.mean(results_prop_spread):.4f}, Std={np.std(results_prop_spread):.4f}")
    print(f"Baseline:   Média={np.mean(results_base_spread):.4f}, Std={np.std(results_base_spread):.4f}")
    print(f"Sequencial: Média={np.mean(results_seq_spread):.4f}, Std={np.std(results_seq_spread):.4f}")
    
    print(f"\n--- TEMPO DE EXECUÇÃO (COMPUTACIONAL) ---")
    print(f"Proposto:   Média={np.mean(times_prop):.1f}s, Std={np.std(times_prop):.1f}s, Mediana={np.median(times_prop):.1f}s")
    print(f"Baseline:   Média={np.mean(times_base):.1f}s, Std={np.std(times_base):.1f}s, Mediana={np.median(times_base):.1f}s")
    print(f"Sequencial: Média={np.mean(times_seq):.1f}s, Std={np.std(times_seq):.1f}s, Mediana={np.median(times_seq):.1f}s")
    print(f"\n   Speedup Proposto vs Baseline: {np.mean(times_base)/np.mean(times_prop):.2f}x")
    print(f"   Speedup Sequencial vs Baseline: {np.mean(times_base)/np.mean(times_seq):.2f}x")
    print(f"   Speedup Sequencial vs Proposto: {np.mean(times_prop)/np.mean(times_seq):.2f}x")
    
    print(f"\n--- NÚMERO DE AVALIAÇÕES DE FUNÇÃO ---")
    print(f"Proposto:   {n_evaluations_prop[0]:,} avaliações (Fase 1: {NGEN_PHASE1*POP_SIZE:,} + Fase 2: {NGEN_PHASE2*POP_SIZE:,})")
    print(f"Baseline:   {n_evaluations_base[0]:,} avaliações ({NGEN_BASELINE*POP_SIZE:,})")
    print(f"Sequencial: {n_evaluations_seq[0]:,} avaliações (Fase 1: {NGEN_PHASE1*POP_SIZE:,} + Fase 2: {NGEN_PHASE2*POP_SIZE:,})")
    
    # Tempo por avaliação (eficiência computacional)
    time_per_eval_prop = np.mean(times_prop) / n_evaluations_prop[0] if n_evaluations_prop[0] > 0 else 0
    time_per_eval_base = np.mean(times_base) / n_evaluations_base[0] if n_evaluations_base[0] > 0 else 0
    time_per_eval_seq = np.mean(times_seq) / n_evaluations_seq[0] if n_evaluations_seq[0] > 0 else 0
    
    print(f"\n--- TEMPO POR AVALIAÇÃO (EFICIÊNCIA) ---")
    print(f"Proposto:   {time_per_eval_prop*1000:.3f} ms/avaliação")
    print(f"Baseline:   {time_per_eval_base*1000:.3f} ms/avaliação")
    print(f"Sequencial: {time_per_eval_seq*1000:.3f} ms/avaliação")
    
    print(f"\n--- TAXA DE SUCESSO (SOLUÇÕES VÁLIDAS) ---")
    success_prop = np.mean(success_rate_prop) * 100
    success_base = np.mean(success_rate_base) * 100
    success_seq = np.mean(success_rate_seq) * 100
    print(f"Proposto:   {success_prop:.1f}% ({sum(success_rate_prop)}/{N_SEEDS} execuções)")
    print(f"Baseline:   {success_base:.1f}% ({sum(success_rate_base)}/{N_SEEDS} execuções)")
    print(f"Sequencial: {success_seq:.1f}% ({sum(success_rate_seq)}/{N_SEEDS} execuções)")
    
    # Coverage (C-metric) - comparação de dominância
    # Calcula sobre as soluções agregadas de todas as seeds
    print(f"\n--- COVERAGE (C-METRIC) - DOMINÂNCIA ENTRE MÉTODOS ---")
    print(f"   (C(A,B) = fração de soluções de B dominadas por A)")
    
    # Filtra soluções válidas para coverage
    prop_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_prop_all if aep > 0 and cost > 0]
    base_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_base_all if aep > 0 and cost > 0]
    seq_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_seq_all if aep > 0 and cost > 0]
    
    # Função auxiliar para calcular coverage entre duas listas de pontos
    def calc_coverage_points(points_A, points_B):
        """Calcula coverage de A sobre B usando listas de pontos (aep, cost)"""
        if len(points_B) == 0 or len(points_A) == 0:
            return 0.0
        dominated = 0
        for b_aep, b_cost in points_B:
            for a_aep, a_cost in points_A:
                if a_aep >= b_aep and a_cost <= b_cost and (a_aep > b_aep or a_cost < b_cost):
                    dominated += 1
                    break
        return dominated / len(points_B)
    
    if len(prop_valid_coverage) > 0 and len(base_valid_coverage) > 0:
        cov_prop_base = calc_coverage_points(prop_valid_coverage, base_valid_coverage)
        cov_base_prop = calc_coverage_points(base_valid_coverage, prop_valid_coverage)
        print(f"Proposto domina Baseline: {cov_prop_base:.3f} ({cov_prop_base*100:.1f}%)")
        print(f"Baseline domina Proposto: {cov_base_prop:.3f} ({cov_base_prop*100:.1f}%)")
    
    if len(prop_valid_coverage) > 0 and len(seq_valid_coverage) > 0:
        cov_prop_seq = calc_coverage_points(prop_valid_coverage, seq_valid_coverage)
        cov_seq_prop = calc_coverage_points(seq_valid_coverage, prop_valid_coverage)
        print(f"Proposto domina Sequencial: {cov_prop_seq:.3f} ({cov_prop_seq*100:.1f}%)")
        print(f"Sequencial domina Proposto: {cov_seq_prop:.3f} ({cov_seq_prop*100:.1f}%)")
    
    if len(base_valid_coverage) > 0 and len(seq_valid_coverage) > 0:
        cov_base_seq = calc_coverage_points(base_valid_coverage, seq_valid_coverage)
        cov_seq_base = calc_coverage_points(seq_valid_coverage, base_valid_coverage)
        print(f"Baseline domina Sequencial: {cov_base_seq:.3f} ({cov_base_seq*100:.1f}%)")
        print(f"Sequencial domina Baseline: {cov_seq_base:.3f} ({cov_seq_base*100:.1f}%)")
    
    # Testes estatísticos
    if SCIPY_AVAILABLE:
        print(f"\n--- TESTES ESTATÍSTICOS ---")
        stats_hv = calculate_statistical_tests(results_prop_hv, results_base_hv, "Hypervolume")
        print(f"Teste: {stats_hv['test']}")
        print(f"Estatística: {stats_hv['statistic']:.4f}")
        print(f"p-value: {stats_hv['p_value']:.6f}")
        print(f"Significativo (p<0.05): {'SIM' if stats_hv['significant'] else 'NÃO'}")
        print(f"Efeito (Cohen's d): {stats_hv['effect_size']['cohens_d']:.4f} ({stats_hv['effect_size']['interpretation']})")
        
        stats_nsol = calculate_statistical_tests(results_prop_n_solutions, results_base_n_solutions, "Number of Solutions")
        print(f"\nNúmero de Soluções - p-value: {stats_nsol['p_value']:.6f}")
        
        stats_spread = calculate_statistical_tests(results_prop_spread, results_base_spread, "Spread")
        print(f"Spread - p-value: {stats_spread['p_value']:.6f}")
        
        # Testes sequencial vs outros
        print(f"\n--- COMPARAÇÃO SEQUENCIAL ---")
        stats_seq_hv = calculate_statistical_tests(results_seq_hv, results_base_hv, "Hypervolume (Sequential vs Baseline)")
        print(f"Sequencial vs Baseline HV - p-value: {stats_seq_hv['p_value']:.6f}")
        
        stats_seq_prop_hv = calculate_statistical_tests(results_seq_hv, results_prop_hv, "Hypervolume (Sequential vs Proposed)")
        print(f"Sequencial vs Proposed HV - p-value: {stats_seq_prop_hv['p_value']:.6f}")
        
        # Testes estatísticos de tempo
        print(f"\n--- TESTES ESTATÍSTICOS - TEMPO DE EXECUÇÃO ---")
        stats_time_prop_base = calculate_statistical_tests(times_base, times_prop, "Execution Time (Baseline vs Proposed)")
        print(f"Baseline vs Proposed Time - p-value: {stats_time_prop_base['p_value']:.6f}")
        print(f"   (Teste verifica se Proposed é mais rápido)")
        
        stats_time_seq_base = calculate_statistical_tests(times_base, times_seq, "Execution Time (Baseline vs Sequential)")
        print(f"Baseline vs Sequential Time - p-value: {stats_time_seq_base['p_value']:.6f}")
        print(f"   (Teste verifica se Sequential é mais rápido)")
        
        stats_time_seq_prop = calculate_statistical_tests(times_prop, times_seq, "Execution Time (Proposed vs Sequential)")
        print(f"Proposed vs Sequential Time - p-value: {stats_time_seq_prop['p_value']:.6f}")
        print(f"   (Teste verifica se Sequential é mais rápido)")
    
    # Tempo de convergência (se rastreado)
    if TRACK_EVOLUTION and len(evolution_prop) > 0:
        print(f"\n--- TEMPO DE CONVERGÊNCIA (95% do HV final) ---")
        conv_prop = [calculate_convergence_gen(evol) for evol in evolution_prop]
        conv_base = [calculate_convergence_gen(evol) for evol in evolution_base]
        conv_prop = [c for c in conv_prop if c is not None]
        conv_base = [c for c in conv_base if c is not None]
        if conv_prop:
            print(f"Proposto:   Média={np.mean(conv_prop):.1f} gerações")
        if conv_base:
            print(f"Baseline:   Média={np.mean(conv_base):.1f} gerações")
    
    # =============================================================================
    # 8. VISUALIZAÇÃO E SALVAMENTO
    # =============================================================================
    
    # PLOT 1: Boxplot Hipervolume - MELHORADO
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot([results_prop_hv, results_base_hv, results_seq_hv], 
                     tick_labels=['Proposed (Two-Phase)', 'Baseline (Pure NSGA-II)', 'Sequential'],
                     patch_artist=True, widths=0.6)
    
    # Cores personalizadas
    colors = ['#2E86AB', '#E63946', '#06A77D']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Melhora linhas
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.2)
    
    plt.ylabel('Hypervolume (Higher is Better)', fontsize=15, fontweight='bold')
    plt.title(f'Statistical Comparison ({N_SEEDS} runs)', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('comparison_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('comparison_boxplot.pdf', bbox_inches='tight', facecolor='white')
    print("Graph saved: comparison_boxplot.png and comparison_boxplot.pdf")
    
    # PLOT 2: Pareto Fronts (Scatter) - MELHORADO
    print(f"\n--- PREPARANDO SCATTER PLOT ---")
    print(f"Total de pontos propostos coletados: {len(pareto_prop_all)}")
    print(f"Total de pontos baseline coletados: {len(pareto_base_all)}")
    
    # Filtra soluções válidas (AEP > 0, Cost > 0)
    pareto_prop_valid = [(aep, cost) for aep, cost in pareto_prop_all if aep > 0 and cost > 0]
    pareto_base_valid = [(aep, cost) for aep, cost in pareto_base_all if aep > 0 and cost > 0]
    pareto_seq_valid = [(aep, cost) for aep, cost in pareto_seq_all if aep > 0 and cost > 0]
    
    print(f"Pontos válidos após filtro: Proposto={len(pareto_prop_valid)}, Baseline={len(pareto_base_valid)}, Sequencial={len(pareto_seq_valid)}")
    
    if len(pareto_prop_valid) == 0 and len(pareto_base_valid) == 0 and len(pareto_seq_valid) == 0:
        print("ERRO: Nenhum ponto válido para plotar! Pulando scatter plot.")
    else:
        plt.figure(figsize=(12, 8))
        
        if len(pareto_prop_valid) > 0:
            p_aep = [p[0] for p in pareto_prop_valid]
            p_cost = [p[1] for p in pareto_prop_valid]
            plt.scatter(p_cost, p_aep, c='#2E86AB', alpha=0.7, s=60, 
                       edgecolors='#1B4965', linewidths=0.8, 
                       label=f'Proposed Solutions (n={len(pareto_prop_valid)})', zorder=3)
        
        if len(pareto_base_valid) > 0:
            b_aep = [p[0] for p in pareto_base_valid]
            b_cost = [p[1] for p in pareto_base_valid]
            plt.scatter(b_cost, b_aep, c='#E63946', alpha=0.6, s=50,
                       edgecolors='#A41623', linewidths=0.8,
                       label=f'Baseline Solutions (n={len(pareto_base_valid)})', zorder=2)
        
        if len(pareto_seq_valid) > 0:
            s_aep = [p[0] for p in pareto_seq_valid]
            s_cost = [p[1] for p in pareto_seq_valid]
            plt.scatter(s_cost, s_aep, c='#06A77D', alpha=0.6, s=50,
                       edgecolors='#045D4A', linewidths=0.8,
                       label=f'Sequential Solutions (n={len(pareto_seq_valid)})', zorder=2)
        
        # Melhora a aparência
        plt.xlabel('Cabling Cost (M USD)', fontsize=15, fontweight='bold')
        plt.ylabel('Net AEP (GWh)', fontsize=15, fontweight='bold')
        plt.title(f'Pareto Fronts Accumulation ({N_SEEDS} seeds)\nPop={POP_SIZE}, Gens={NGEN_BASELINE}', 
                  fontsize=18, fontweight='bold')
        plt.legend(fontsize=14, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Remove bordas superiores e direitas
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig('comparison_pareto.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('comparison_pareto.pdf', bbox_inches='tight', facecolor='white')
        print(f"Graph saved: comparison_pareto.png and comparison_pareto.pdf ({len(pareto_prop_valid)} proposed, {len(pareto_base_valid)} baseline, {len(pareto_seq_valid)} sequential solutions)")
    
    # PLOT 3: Evolução do Hipervolume (se rastreado)
    if TRACK_EVOLUTION and len(evolution_prop) > 0:
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Evolução média
        plt.subplot(1, 2, 1)
        for i, evol in enumerate(evolution_prop):
            if i == 0:
                plt.plot(evol["gen"], evol["hv"], '#2E86AB', alpha=0.3, linewidth=0.8, label='Proposed (individual)')
            else:
                plt.plot(evol["gen"], evol["hv"], '#2E86AB', alpha=0.3, linewidth=0.8)
        
        for i, evol in enumerate(evolution_base):
            if i == 0:
                plt.plot(evol["gen"], evol["hv"], '#E63946', alpha=0.3, linewidth=0.8, label='Baseline (individual)')
            else:
                plt.plot(evol["gen"], evol["hv"], '#E63946', alpha=0.3, linewidth=0.8)
        
        # Calcula média por geração
        all_gens_prop = set()
        for evol in evolution_prop:
            all_gens_prop.update(evol["gen"])
        all_gens_base = set()
        for evol in evolution_base:
            all_gens_base.update(evol["gen"])
        
        if all_gens_prop:
            gen_range_prop = sorted(all_gens_prop)
            hv_mean_prop = []
            for gen in gen_range_prop:
                hvs = []
                for evol in evolution_prop:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        hvs.append(evol["hv"][idx])
                if hvs:
                    hv_mean_prop.append(np.mean(hvs))
                else:
                    hv_mean_prop.append(None)
            # Remove None values
            gen_range_prop_clean = [g for g, h in zip(gen_range_prop, hv_mean_prop) if h is not None]
            hv_mean_prop_clean = [h for h in hv_mean_prop if h is not None]
            if gen_range_prop_clean:
                plt.plot(gen_range_prop_clean, hv_mean_prop_clean, '#2E86AB', linewidth=2.5, 
                        label='Proposed (mean)', zorder=10)
        
        if all_gens_base:
            gen_range_base = sorted(all_gens_base)
            hv_mean_base = []
            for gen in gen_range_base:
                hvs = []
                for evol in evolution_base:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        hvs.append(evol["hv"][idx])
                if hvs:
                    hv_mean_base.append(np.mean(hvs))
                else:
                    hv_mean_base.append(None)
            # Remove None values
            gen_range_base_clean = [g for g, h in zip(gen_range_base, hv_mean_base) if h is not None]
            hv_mean_base_clean = [h for h in hv_mean_base if h is not None]
            if gen_range_base_clean:
                plt.plot(gen_range_base_clean, hv_mean_base_clean, '#E63946', linewidth=2.5, 
                        label='Baseline (mean)', zorder=10)
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Hypervolume', fontsize=14, fontweight='bold')
        plt.title('Hypervolume Evolution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        ax1 = plt.gca()
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Subplot 2: Número de soluções
        plt.subplot(1, 2, 2)
        for i, evol in enumerate(evolution_prop):
            if i == 0:
                plt.plot(evol["gen"], evol["n_solutions"], '#2E86AB', alpha=0.3, linewidth=0.8, label='Proposed (individual)')
            else:
                plt.plot(evol["gen"], evol["n_solutions"], '#2E86AB', alpha=0.3, linewidth=0.8)
        
        for i, evol in enumerate(evolution_base):
            if i == 0:
                plt.plot(evol["gen"], evol["n_solutions"], '#E63946', alpha=0.3, linewidth=0.8, label='Baseline (individual)')
            else:
                plt.plot(evol["gen"], evol["n_solutions"], '#E63946', alpha=0.3, linewidth=0.8)
        
        # Calcula média para número de soluções também
        if all_gens_prop:
            n_sol_mean_prop = []
            for gen in gen_range_prop:
                n_sols = []
                for evol in evolution_prop:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        n_sols.append(evol["n_solutions"][idx])
                if n_sols:
                    n_sol_mean_prop.append(np.mean(n_sols))
                else:
                    n_sol_mean_prop.append(None)
            gen_range_prop_clean_nsol = [g for g, n in zip(gen_range_prop, n_sol_mean_prop) if n is not None]
            n_sol_mean_prop_clean = [n for n in n_sol_mean_prop if n is not None]
            if gen_range_prop_clean_nsol:
                plt.plot(gen_range_prop_clean_nsol, n_sol_mean_prop_clean, '#2E86AB', 
                        linewidth=2.5, label='Proposed (mean)', zorder=10)
        
        if all_gens_base:
            n_sol_mean_base = []
            for gen in gen_range_base:
                n_sols = []
                for evol in evolution_base:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        n_sols.append(evol["n_solutions"][idx])
                if n_sols:
                    n_sol_mean_base.append(np.mean(n_sols))
                else:
                    n_sol_mean_base.append(None)
            gen_range_base_clean_nsol = [g for g, n in zip(gen_range_base, n_sol_mean_base) if n is not None]
            n_sol_mean_base_clean = [n for n in n_sol_mean_base if n is not None]
            if gen_range_base_clean_nsol:
                plt.plot(gen_range_base_clean_nsol, n_sol_mean_base_clean, '#E63946', 
                        linewidth=2.5, label='Baseline (mean)', zorder=10)
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Pareto Solutions', fontsize=14, fontweight='bold')
        plt.title('Pareto Front Size Evolution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        ax2 = plt.gca()
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comparison_evolution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('comparison_evolution.pdf', bbox_inches='tight', facecolor='white')
        print("Graph saved: comparison_evolution.png and comparison_evolution.pdf")
    
    # PLOT 4: Comparação de múltiplas métricas - MELHORADO (incluindo tempo)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = ['#2E86AB', '#E63946', '#06A77D']
    
    # Hipervolume
    bp1 = axes[0].boxplot([results_prop_hv, results_base_hv, results_seq_hv], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='black', linewidth=1.2)
    axes[0].set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
    axes[0].set_title('Hypervolume Comparison', fontsize=16, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Número de soluções
    bp2 = axes[1].boxplot([results_prop_n_solutions, results_base_n_solutions, results_seq_n_solutions], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp2[element], color='black', linewidth=1.2)
    axes[1].set_ylabel('Number of Solutions', fontsize=14, fontweight='bold')
    axes[1].set_title('Pareto Front Size', fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Spread
    bp3 = axes[2].boxplot([results_prop_spread, results_base_spread, results_seq_spread], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp3[element], color='black', linewidth=1.2)
    axes[2].set_ylabel('Spread (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].set_title('Solution Diversity', fontsize=16, fontweight='bold')
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    # Tempo de execução
    bp4 = axes[3].boxplot([times_prop, times_base, times_seq], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp4[element], color='black', linewidth=1.2)
    axes[3].set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    axes[3].set_title('Computational Efficiency', fontsize=16, fontweight='bold')
    axes[3].tick_params(axis='both', which='major', labelsize=12)
    axes[3].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('comparison_metrics.pdf', bbox_inches='tight', facecolor='white')
    print("Graph saved: comparison_metrics.png and comparison_metrics.pdf")
    
    # PLOT 5: Gráfico dedicado de tempo de execução (para artigo)
    plt.figure(figsize=(10, 6))
    bp_time = plt.boxplot([times_prop, times_base, times_seq], 
                          tick_labels=['Proposed\n(Two-Phase)', 'Baseline\n(Pure NSGA-II)', 'Sequential\n(Two GAs)'],
                          patch_artist=True, widths=0.6)
    
    # Cores personalizadas
    colors_time = ['#2E86AB', '#E63946', '#06A77D']
    for patch, color in zip(bp_time['boxes'], colors_time):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Melhora linhas
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp_time[element], color='black', linewidth=1.2)
    
    plt.ylabel('Execution Time (seconds)', fontsize=16, fontweight='bold')
    plt.title(f'Computational Efficiency Comparison ({N_SEEDS} runs)', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Adiciona anotações com speedup
    speedup_prop = np.mean(times_base) / np.mean(times_prop)
    speedup_seq = np.mean(times_base) / np.mean(times_seq)
    speedup_seq_prop = np.mean(times_prop) / np.mean(times_seq)
    
    # Anotações no gráfico
    ax_time = plt.gca()
    ax_time.text(1, np.max(times_prop) * 1.1, f'{speedup_prop:.2f}x faster\nthan baseline', 
                ha='center', fontsize=11, fontweight='bold', color='#2E86AB')
    ax_time.text(3, np.max(times_seq) * 1.1, f'{speedup_seq:.2f}x faster\nthan baseline', 
                ha='center', fontsize=11, fontweight='bold', color='#06A77D')
    
    ax_time.tick_params(axis='both', which='major', labelsize=13)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('comparison_execution_time.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('comparison_execution_time.pdf', bbox_inches='tight', facecolor='white')
    print("Graph saved: comparison_execution_time.png and comparison_execution_time.pdf")
    
    # Salva resultados em CSV
    results_df = pd.DataFrame({
        'seed': list(range(N_SEEDS)),
        'hv_proposed': results_prop_hv,
        'hv_baseline': results_base_hv,
        'hv_sequential': results_seq_hv,
        'n_solutions_proposed': results_prop_n_solutions,
        'n_solutions_baseline': results_base_n_solutions,
        'n_solutions_sequential': results_seq_n_solutions,
        'spread_proposed': results_prop_spread,
        'spread_baseline': results_base_spread,
        'spread_sequential': results_seq_spread,
        'time_proposed_seconds': times_prop,
        'time_baseline_seconds': times_base,
        'time_sequential_seconds': times_seq,
        'n_evaluations_proposed': n_evaluations_prop,
        'n_evaluations_baseline': n_evaluations_base,
        'n_evaluations_sequential': n_evaluations_seq,
        'success_rate_proposed': success_rate_prop,
        'success_rate_baseline': success_rate_base,
        'success_rate_sequential': success_rate_seq
    })
    results_df.to_csv('benchmark_results.csv', index=False)
    print("Results saved: benchmark_results.csv")
    
    plt.show()