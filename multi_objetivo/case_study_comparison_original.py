"""
ESTUDO DE CASO COMPARATIVO
==========================
Compara 3 métodos de otimização para parques eólicos offshore:
1. Baseline: NSGA-II puro (evolui turbinas + subestação + cabeamento)
2. Proposed: Método proposto (a ser implementado)
3. Sequential: Método sequencial (a ser implementado)

Todos os métodos começam com a mesma população inicial (coordenadas do YAML).
"""

"""
CÓDIGO PRINCIPAL DE EXECUÇÃO DO ESTUDO DE CASO
==============================================
Executa os 3 métodos de otimização e salva resultados em 3 CSVs:
1. summary_results.csv - Uma linha por execução/método (melhor solução)
2. all_pareto_fronts.csv - Múltiplas linhas por execução (todas soluções Pareto)
3. convergence_history.csv - Uma linha a cada X gerações (dinâmica do algoritmo)

NOTA: As funções de plotagem foram removidas e devem ser executadas separadamente
usando o arquivo case_study_plot.py que lê os CSVs gerados.
"""

import sys
import os
import time
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools
import multiprocessing
from functools import partial

# Importa função de hipervolume
try:
    from deap.tools._hypervolume import hv as hypervolume_module
except ImportError:
    from deap.tools._hypervolume import pyhv as hypervolume_module

# Importa módulos do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multi_objetivo.cabling_v3 as cabling_v3

# Importa módulo de coleta de métricas
from multi_objetivo.metrics_collector import (
    collect_summary_metrics,
    collect_pareto_front_metrics,
    collect_convergence_metrics,
    collect_representative_solution_coords,
    save_summary_results,
    save_pareto_fronts,
    save_convergence_history,
    save_representative_solutions
)

# =============================================================================
# CONFIGURAÇÃO DO ESTUDO DE CASO
# =============================================================================

# SEED será gerada aleatoriamente no início da execução e salva para reprodutibilidade
SEED = None  # Será definida no início da execução
N_RUNS = 5  # Número de execuções (cada uma com seed diferente)
POP_SIZE = 15  # Tamanho da população
generations = 500
NGEN_BASELINE = 2*generations  # Número de gerações para Baseline
NGEN_SEQUENTIAL_P1 = generations  # Gerações Fase 1 Sequential (igual ao wind_farm_GA_16.py)
NGEN_SEQUENTIAL_P2 = generations  # Gerações Fase 2 Sequential (otimização de cabeamento)
NGEN_PROPOSED_P1 = generations  # Gerações Fase 1 Proposed (igual ao Sequential Fase 1)
NGEN_PROPOSED_P2 = generations  # Gerações Fase 2 Proposed (NSGA-II multiobjetivo)

# Parâmetros do Algoritmo Genético
CXPB = 0.95  # Probabilidade de crossover
MUTPB = 0.7  # Probabilidade de mutação
INDPB = 0.4  # Probabilidade de mutar cada gene
SIGMA = 100  # Desvio padrão para mutação gaussiana (metros)
TOURNSIZE = 5  # Tamanho do torneio

# Constantes físicas
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 5000  # Raio do círculo de restrição (metros)
N_DIAMETERS = 260  # Distância mínima entre turbinas (diâmetros)
MIN_SUB_TURB_DIST = 50.0  # Distância mínima entre subestação e turbinas (metros)

# Limites para número de grupos de cabeamento
MIN_GRUPOS = np.sqrt(IND_SIZE)
MAX_GRUPOS = IND_SIZE
N_GRUPOS_INICIAL = random.randint(MIN_GRUPOS, MAX_GRUPOS)

# =============================================================================
# CONFIGURAÇÃO DO DIRETÓRIO DE SAÍDA
# =============================================================================
# Define o nome do diretório onde todos os resultados serão salvos
# Modifique esta variável para escolher o nome do diretório
# Exemplo: OUTPUT_DIR = 'teste_16' ou OUTPUT_DIR = 'teste_36_turbinas'
OUTPUT_DIR = 'results_16'  # <-- MODIFIQUE AQUI o nome do diretório

# Parâmetros de detecção de sobreposição de cabos
MIN_CABLE_DISTANCE = 100.0  # Distância mínima permitida entre segmentos de cabos (metros)
MIN_ANGLE_SUBSTATION = 30  # Ângulo mínimo (graus) entre cabos chegando na subestação

PENALTY_CROSSING = 1e6  # Penalidade por cruzamento de cabos
PENALTY_MULTIPLE_CONNECTIONS = 1e6  # Penalidade por múltiplas conexões na mesma turbina
PENALTY_SMALL_ANGLE_SUBSTATION = 1e6  # Penalidade por ângulo muito fechado na subestação

# Penalidades MUITO reduzidas para inicialização (permite layouts ruins que podem ser evoluídos)
# Reduzidas em 1000x comparado às penalidades normais para permitir inicialização mesmo com problemas
PENALTY_CROSSING_INIT = 1e3  # Penalidade muito reduzida por cruzamento na inicialização (1000x menor)
PENALTY_MULTIPLE_CONNECTIONS_INIT = 1e3  # Penalidade muito reduzida por múltiplas conexões na inicialização
PENALTY_SMALL_ANGLE_SUBSTATION_INIT = 1e3  # Penalidade muito reduzida por ângulo fechado na inicialização

# Parâmetros Proposed Fase 2 (similar ao multi16_prioriza_aep.py)
HOF_SIZE_P1 = 50  # Tamanho do Hall of Fame Fase 1
N_TOP_LAYOUTS = 30  # Número de melhores layouts da Fase 1 usados na Fase 2
PERTURBATION_SIGMA_MIN = 150  # Sigma mínimo para perturbação de layouts (metros)
PERTURBATION_SIGMA_MAX = 300  # Sigma máximo para perturbação de layouts (metros)
PATIENCE_P2 = 100  # Gerações sem melhoria antes de parar Fase 2
MIN_DELTA_AEP_P2 = 10.0  # Melhoria mínima em AEP (MWh) para resetar estagnação
MIN_DELTA_COST_P2 = 100.0  # Melhoria mínima em custo (USD) para resetar estagnação
PROB_MUTATE_GROUPS_P2 = 0.3  # Probabilidade de mutar número de grupos
PROB_MUTATE_SUBSTATION_P2 = 0.9  # Probabilidade de mutar posição da subestação
SIGMA_SUB_MULTIPLIER = 5  # Multiplicador do sigma para mutação da subestação
SIGMA_SUB_MIN = 200.0  # Sigma mínimo para mutação da subestação (metros)

# =============================================================================
# CARREGAMENTO DE DADOS DO CONFIG
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = "config"

# Caminhos para arquivos YAML (explícitos)
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
windrose_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
turbine_attrs_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")

# Carrega coordenadas iniciais das turbinas (do YAML)
initial_coordinates, _, _ = getTurbLocYAML(main_yaml_path)

# Carrega dados de vento (wind rose) e características das turbinas
TURB_ATRBT_DATA = getTurbAtrbtYAML(turbine_attrs_yaml_path)  # [turb_ci, turb_co, rated_ws, rated_pwr, turb_diam]
WIND_ROSE_DATA = getWindRoseYAML(windrose_yaml_path)  # [wind_dir, wind_freq, wind_speed]

# =============================================================================
# CONFIGURAÇÃO DEAP
# =============================================================================

# Limpa tipos anteriores se existirem
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "IndividualBaseline"):
    del creator.IndividualBaseline
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "IndividualPhase1"):
    del creator.IndividualPhase1
if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "IndividualSequential"):
    del creator.IndividualSequential

# Baseline: Multi-objetivo (maximizar AEP, minimizar custo)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualBaseline", list, fitness=creator.FitnessMulti)

# Sequential Fase 1: Single-objective (maximizar AEP bruto)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

# Sequential Fase 2: Single-objective (minimizar custo)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualSequential", list, fitness=creator.FitnessMin)

# Proposed Fase 2: Multi-objective (maximizar AEP, minimizar custo) - usa mesmo tipo do Baseline
# Reutiliza IndividualBaseline pois tem a mesma estrutura: [coords turbinas] + [n_grupos] + [coords subestação]

toolbox_baseline = base.Toolbox()
toolbox_seq_p1 = base.Toolbox()  # Fase 1 Sequential
toolbox_seq_p2 = base.Toolbox()  # Fase 2 Sequential
toolbox_prop_p2 = base.Toolbox()  # Fase 2 Proposed

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def is_within_circle(x, y, radius):
    """Verifica se coordenadas estão dentro do círculo."""
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
    """
    sub_pos = np.array(centroid)
    dist_to_turbines = np.linalg.norm(turb_coords - sub_pos, axis=1)
    min_dist = np.min(dist_to_turbines)
    
    if min_dist < min_distance:
        # Move subestação na direção oposta à turbina mais próxima
        closest_idx = np.argmin(dist_to_turbines)
        direction = sub_pos - turb_coords[closest_idx]
        direction = direction / np.linalg.norm(direction)
        sub_pos = turb_coords[closest_idx] + direction * min_distance
    
    return sub_pos.tolist()

def detectar_sobreposicao_cabos(paths, coords, min_distance=100.0, substation_idx=None, init_mode=False):
    """
    Detecta cruzamentos, proximidade excessiva e múltiplas conexões na mesma turbina.
    
    Args:
        paths: Lista de caminhos de cabeamento
        coords: Coordenadas dos pontos (turbinas + subestação)
        min_distance: Distância mínima permitida entre segmentos de cabos (metros)
        substation_idx: Índice da subestação (pode ter múltiplas conexões, mas turbinas não)
    
    Returns:
        Penalidade total (cruzamentos + proximidade + múltiplas conexões)
    """
    def segmentos_intersectam(p1, p2, q1, q2):
        """
        Verifica se dois segmentos de linha se cruzam usando teste de orientação (CCW).
        Implementação robusta que evita erros numéricos.
        """
        def ccw(o, a, b):
            """
            Counter-clockwise test: retorna orientação de três pontos.
            Retorna: >0 se counter-clockwise, <0 se clockwise, 0 se colineares
            """
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def no_segmento(p, q, r, tol=1e-9):
            """
            Verifica se o ponto q está no segmento pr.
            Usa tolerância para lidar com erros numéricos.
            """
            # Verifica se q está na caixa delimitadora de pr
            if not (min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol):
                return False
            if not (min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol):
                return False
            # Verifica se q está colinear com pr (usando produto vetorial)
            cross = abs(ccw(p, q, r))
            return cross < tol
        
        # Teste de orientação (CCW) para cada par de pontos
        o1 = ccw(p1, p2, q1)
        o2 = ccw(p1, p2, q2)
        o3 = ccw(q1, q2, p1)
        o4 = ccw(q1, q2, p2)
        
        # Caso geral: segmentos se cruzam se orientações são diferentes
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        
        # Casos especiais: pontos colineares
        tol = 1e-9
        if abs(o1) < tol and no_segmento(p1, q1, p2, tol):
            return True
        if abs(o2) < tol and no_segmento(p1, q2, p2, tol):
            return True
        if abs(o3) < tol and no_segmento(q1, p1, q2, tol):
            return True
        if abs(o4) < tol and no_segmento(q1, p2, q2, tol):
            return True
        
        return False
    
    def distancia_ponto_segmento(p, seg_start, seg_end):
        """Calcula distância mínima de um ponto a um segmento de linha."""
        # Vetor do segmento
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-9:
            # Segmento degenerado (ponto)
            return np.linalg.norm(p - seg_start)
        
        # Vetor do início do segmento ao ponto
        p_vec = p - seg_start
        
        # Projeção do ponto no segmento (parâmetro t)
        t = max(0.0, min(1.0, np.dot(p_vec, seg_vec) / seg_len_sq))
        
        # Ponto mais próximo no segmento
        closest = seg_start + t * seg_vec
        
        # Distância do ponto ao segmento
        return np.linalg.norm(p - closest)
    
    def distancia_segmentos(p1, p2, q1, q2):
        """Calcula distância mínima entre dois segmentos de linha."""
        # Calcula distância de cada ponto de um segmento ao outro segmento
        d1 = distancia_ponto_segmento(p1, q1, q2)
        d2 = distancia_ponto_segmento(p2, q1, q2)
        d3 = distancia_ponto_segmento(q1, p1, p2)
        d4 = distancia_ponto_segmento(q2, p1, p2)
        
        return min(d1, d2, d3, d4)
    
    n_cruzamentos = 0
    penalty_proximidade = 0.0
    n_multiplas_conexoes = 0
    n_angulos_fechados_sub = 0  # Contador de ângulos muito fechados na subestação
    
    # Detecta múltiplas conexões na mesma turbina (exceto subestação)
    # Em um sistema de cabeamento em árvore, cada turbina deve aparecer em apenas UM path
    # Se aparecer em múltiplos paths, há múltiplas conexões (problema)
    turbina_paths = {}  # {turbina_idx: [lista de paths que contêm essa turbina]}
    for path_idx, path in enumerate(paths):
        # Cada turbina no path (exceto a última que é a subestação)
        for turb_idx in path[:-1]:  # Todas exceto a última (subestação)
            if turb_idx != substation_idx:  # Ignora subestação
                if turb_idx not in turbina_paths:
                    turbina_paths[turb_idx] = []
                turbina_paths[turb_idx].append(path_idx)
    
    # Penaliza turbinas que aparecem em múltiplos paths
    # Cada turbina deve aparecer em apenas 1 path (exceto subestação)
    for turb_idx, path_list in turbina_paths.items():
        if len(path_list) > 1:  # Turbina aparece em múltiplos paths = problema
            n_multiplas_conexoes += len(path_list) - 1  # Penaliza paths extras
    
    # Detecta cruzamentos e proximidade entre segmentos
    # 1. Cruzamentos DENTRO do mesmo path (intra-string)
    for path_idx, path in enumerate(paths):
        for k in range(len(path) - 1):
            p1 = np.array(coords[path[k]])
            p2 = np.array(coords[path[k + 1]])
            
            # Verifica contra segmentos não consecutivos do mesmo path
            for l in range(len(path) - 1):
                # Ignora segmentos consecutivos (são válidos, formam o caminho)
                if abs(k - l) <= 1:
                    continue
                
                q1 = np.array(coords[path[l]])
                q2 = np.array(coords[path[l + 1]])
                
                # Detecta cruzamentos intra-string
                if segmentos_intersectam(p1, p2, q1, q2):
                    n_cruzamentos += 1
                else:
                    # Detecta proximidade excessiva intra-string
                    dist = distancia_segmentos(p1, p2, q1, q2)
                    if dist < min_distance:
                        violation = min_distance - dist
                        penalty_proximidade += violation * (min_distance / max(dist, 1.0)) * 1000
    
    # 2. Cruzamentos ENTRE paths diferentes (inter-string)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_i = paths[i]
            path_j = paths[j]
            
            # Verifica ângulo entre os últimos segmentos (chegando na subestação)
            # Se ambos os paths terminam na subestação, calcula o ângulo entre os últimos segmentos
            if (len(path_i) >= 2 and len(path_j) >= 2 and 
                path_i[-1] == substation_idx and path_j[-1] == substation_idx):
                # Último segmento do path_i: penúltimo ponto -> subestação
                p_penultimo_i = np.array(coords[path_i[-2]])
                p_sub_i = np.array(coords[path_i[-1]])
                vec_i = p_sub_i - p_penultimo_i  # Vetor do penúltimo para a subestação
                
                # Último segmento do path_j: penúltimo ponto -> subestação
                p_penultimo_j = np.array(coords[path_j[-2]])
                p_sub_j = np.array(coords[path_j[-1]])
                vec_j = p_sub_j - p_penultimo_j  # Vetor do penúltimo para a subestação
                
                # Calcula ângulo entre os vetores usando produto escalar
                # cos(θ) = (v1 · v2) / (|v1| * |v2|)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 1e-9 and norm_j > 1e-9:  # Evita divisão por zero
                    cos_angle = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    # Limita cos_angle entre -1 e 1 (por erros numéricos)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_degrees = np.degrees(np.arccos(cos_angle))
                    
                    # Se o ângulo for menor que o mínimo, os cabos estão muito próximos
                    if angle_degrees < MIN_ANGLE_SUBSTATION:
                        n_angulos_fechados_sub += 1
            
            for k in range(len(path_i) - 1):
                p1 = np.array(coords[path_i[k]])
                p2 = np.array(coords[path_i[k + 1]])
                
                for l in range(len(path_j) - 1):
                    q1 = np.array(coords[path_j[l]])
                    q2 = np.array(coords[path_j[l + 1]])
                    
                    # Ignora se compartilham pontos (conexões válidas em nós)
                    if (path_i[k] == path_j[l] or path_i[k] == path_j[l + 1] or
                        path_i[k + 1] == path_j[l] or path_i[k + 1] == path_j[l + 1]):
                        continue
                    
                    # Detecta cruzamentos inter-string
                    if segmentos_intersectam(p1, p2, q1, q2):
                        n_cruzamentos += 1
                    else:
                        # Detecta proximidade excessiva inter-string
                        dist = distancia_segmentos(p1, p2, q1, q2)
                        if dist < min_distance:
                            violation = min_distance - dist
                            penalty_proximidade += violation * (min_distance / max(dist, 1.0)) * 1000
    
    # Penalidade para cruzamentos: usa valores diferentes para inicialização vs. evolução
    if init_mode:
        penalty_cruzamentos = n_cruzamentos * PENALTY_CROSSING_INIT
        penalty_multiplas = n_multiplas_conexoes * PENALTY_MULTIPLE_CONNECTIONS_INIT
        penalty_angulo_fechado = n_angulos_fechados_sub * PENALTY_SMALL_ANGLE_SUBSTATION_INIT
    else:
        # Penalidade EXTREMA para cruzamentos: elimina soluções com cruzamentos
        # Cada cruzamento é inaceitável e deve ser fortemente punido
        penalty_cruzamentos = n_cruzamentos * PENALTY_CROSSING
        
        # Penalidade EXTREMA para múltiplas conexões: elimina soluções com turbinas conectadas múltiplas vezes
        # Cada conexão múltipla é inaceitável e deve ser fortemente punida
        penalty_multiplas = n_multiplas_conexoes * PENALTY_MULTIPLE_CONNECTIONS
        
        # Penalidade para ângulos muito fechados na subestação
        # Cabos chegando com ângulo < 15° estão muito próximos e devem ser penalizados
        penalty_angulo_fechado = n_angulos_fechados_sub * PENALTY_SMALL_ANGLE_SUBSTATION
    
    # Penalidade total: extremamente alta para garantir eliminação
    penalty_total = penalty_cruzamentos + penalty_proximidade + penalty_multiplas + penalty_angulo_fechado
    
    return penalty_total

# =============================================================================
# POPULAÇÃO INICIAL GLOBAL (COMPARTILHADA POR TODOS OS MÉTODOS)
# =============================================================================

def create_global_initial_population(seed, pop_size):
    """
    Cria população inicial GLOBAL de coordenadas de turbinas.
    Esta população será compartilhada por Baseline, Sequential e Proposed.
    
    Estratégia ROBUSTA:
    - 100% dos indivíduos: coordenadas EXATAS do YAML (sem perturbações)
    - Isso garante que todos os métodos comecem com layouts válidos e conhecidos
    
    Args:
        seed: Seed para reprodutibilidade (mantido para compatibilidade, mas não usado)
        pop_size: Tamanho da população
    
    Returns:
        global_coords_pop: Lista de arrays numpy, cada um com shape (IND_SIZE, 2)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    global_coords_pop = []
    
    # TODOS os indivíduos usam coordenadas EXATAS do YAML
    # Isso garante máxima robustez na inicialização
    for i in range(pop_size):
        # Coordenadas EXATAS do YAML (cópia para evitar referências compartilhadas)
        coords = initial_coordinates.copy()
        global_coords_pop.append(coords.copy())
    
    return global_coords_pop

# =============================================================================
# INICIALIZAÇÃO DA POPULAÇÃO (BASELINE) - USA POPULAÇÃO GLOBAL
# =============================================================================

# Variável global para armazenar população inicial de coordenadas
_global_coords_population = None
_global_coords_index = 0

def create_baseline_individual(conservative=False, n_grupos_target=None):
    """
    Cria indivíduo Baseline usando coordenadas da população inicial GLOBAL.
    
    Estrutura do genoma:
    - [IND_SIZE*2 coords turbinas]: coordenadas (x, y) de cada turbina
    - [1 n_grupos]: número de grupos normalizado [0, 1]
    - [2 coords subestação]: coordenadas (x, y) da subestação
    Total: IND_SIZE*2 + 3 variáveis
    
    Args:
        conservative: Se True, usa estratégia conservadora (coordenadas exatas, mais grupos, subestação no centroide)
        n_grupos_target: Se especificado, força número de grupos específico (normalizado [0, 1])
    """
    global _global_coords_population, _global_coords_index
    
    # Usa coordenadas da população global (cíclica)
    if _global_coords_population is None or len(_global_coords_population) == 0:
        # Fallback: usa coordenadas exatas do YAML se população global não estiver definida
        coords = initial_coordinates.flatten().tolist()
    else:
        if conservative:
            # Modo conservador: sempre usa coordenadas exatas do YAML (primeiro elemento)
            coords = initial_coordinates.flatten().tolist()
        else:
            coords_array = _global_coords_population[_global_coords_index % len(_global_coords_population)]
            _global_coords_index += 1
            coords = coords_array.flatten().tolist()
    
    coords_array = np.array(coords).reshape((IND_SIZE, 2))
    centroid = np.mean(coords_array, axis=0)
    
    if n_grupos_target is not None:
        # Força número de grupos específico
        n_grupos_norm = n_grupos_target
    elif conservative:
        # Modo conservador: número de grupos maior (mais seguro, menos chance de cruzamentos)
        # Usa número de grupos entre 60% e 80% do máximo (mais grupos = menos cruzamentos)
        n_grupos_norm = random.uniform(0.6, 0.8)
        n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))
    else:
        # Modo normal: VARIADO para diversidade
        # 2. Gene de Grupos [0, 1] - Distribui uniformemente entre MIN_GRUPOS e MAX_GRUPOS
        n_grupos_norm = random.uniform(0.0, 1.0)
        n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))
    
    if conservative:
        # Subestação no centroide (mais seguro)
        sub_pos = displace_substation_from_turbines(centroid, coords_array, min_distance=MIN_SUB_TURB_DIST)
    else:
        # 3. Subestação - POSIÇÃO VARIADA para diversidade
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(MIN_SUB_TURB_DIST, CIRCLE_RADIUS * 0.3)
        sub_pos_candidate = centroid + np.array([radius * np.cos(angle), radius * np.sin(angle)])
        sub_pos = displace_substation_from_turbines(sub_pos_candidate, coords_array, min_distance=MIN_SUB_TURB_DIST)
    
    # Monta genoma completo
    full_genome = coords + [n_grupos_norm] + sub_pos
    return creator.IndividualBaseline(full_genome)

toolbox_baseline.register("individual", create_baseline_individual)
toolbox_baseline.register("population", tools.initRepeat, list, toolbox_baseline.individual)

# =============================================================================
# FUNÇÃO DE AVALIAÇÃO (BASELINE)
# =============================================================================

def evaluate_baseline(individual, init_mode=False):
    """
    Avalia indivíduo Baseline: calcula AEP líquido e Custo total.
    
    Returns:
        (aep_liquido, custo_total): Tupla com AEP líquido (MWh) e Custo total (USD)
    """
    try:
        n_coords = IND_SIZE * 2
        
        # Extrai componentes do genoma
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        
        # Converte número de grupos
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        n_grupos = min(n_grupos, IND_SIZE)  # Limita ao número de turbinas
        
        turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # Penalidades geométricas (reduzidas na inicialização para permitir layouts ruins)
        penalty_mult_geom = 1e3 if init_mode else 1e5  # Multiplicador de penalidades geométricas (100x menor na inicialização)
        
        dist_turb = np.linalg.norm(turb_coords, axis=1)
        violations_turb = np.maximum(0, dist_turb - CIRCLE_RADIUS)
        pen_turb_out = np.sum(violations_turb) * penalty_mult_geom
        
        # Distância Turbina-Turbina
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_u, j_u = np.triu_indices(IND_SIZE, k=1)
        violations = np.maximum(0, N_DIAMETERS - dists[i_u, j_u])
        pen_spacing = np.sum(violations) * penalty_mult_geom
        
        # Distância Sub-Turbina
        d_sub_turb = np.linalg.norm(turb_coords - sub_pos, axis=1)
        violation_sub_close = np.maximum(0, MIN_SUB_TURB_DIST - np.min(d_sub_turb))
        pen_sub_close = violation_sub_close * penalty_mult_geom
        
        # Subestação fora do círculo
        dist_sub = np.linalg.norm(sub_pos)
        pen_sub_out = np.maximum(0, dist_sub - CIRCLE_RADIUS) * penalty_mult_geom
        
        # Calcula AEP bruto
        wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
        turb_diam = TURB_ATRBT_DATA[4]
        aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                   TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], 
                                   TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
        
        # Calcula cabeamento
        coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
        substation_idx = IND_SIZE  # Índice da subestação após todas as turbinas
        try:
            plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=substation_idx, n_grupos=n_grupos)
            custo_usd = res['custo_total_usd']
            perdas_mwh = res['perda_anual_mwh']
            
            # SEMPRE detecta cruzamentos, proximidade excessiva e múltiplas conexões
            # Esta penalidade é aplicada globalmente em todos os métodos
            # Na inicialização, usa penalidades reduzidas para garantir que pelo menos algumas soluções sejam válidas
            penalty_overlap = detectar_sobreposicao_cabos(
                plant.paths, coords_all, min_distance=MIN_CABLE_DISTANCE, substation_idx=substation_idx, init_mode=init_mode)
        except Exception as e:
            # Em caso de erro no cabeamento, aplica penalidade extrema
            return -1e6, 1e12
        
        # AEP líquido e custo final (com penalidades)
        # Na inicialização, permite AEP negativo pequeno (será corrigido na evolução)
        aep_liq = aep_bruto - perdas_mwh - pen_turb_out - pen_spacing - pen_sub_out - pen_sub_close - penalty_overlap
        custo_final = custo_usd + pen_turb_out + pen_spacing + pen_sub_out + pen_sub_close + penalty_overlap
        
        # Na inicialização, aceita soluções mesmo com AEP negativo pequeno (até -1000 MWh)
        # Isso permite que layouts ruins sejam aceitos e evoluídos
        if init_mode:
            if aep_liq <= -1000:  # Apenas rejeita se AEP for muito negativo
                return -1e6, 1e12
        else:
            if aep_liq <= 0:  # Durante evolução, rejeita AEP não positivo
                return -1e6, 1e12
        
        return aep_liq, custo_final
        
    except Exception as e:
        return -1e6, 1e12

toolbox_baseline.register("evaluate", evaluate_baseline)

# Wrapper para avaliação na inicialização (com penalidades reduzidas)
def evaluate_baseline_init(individual):
    """Wrapper para evaluate_baseline com init_mode=True (penalidades reduzidas)."""
    return evaluate_baseline(individual, init_mode=True)

# =============================================================================
# OPERADORES GENÉTICOS (BASELINE)
# =============================================================================

def mutate_baseline(individual, mu, sigma, indpb):
    """
    Mutação para Baseline: aplica mutação gaussiana diferenciada.
    
    Estrutura: [32 coords turbinas] + [1 n_grupos] + [2 coords subestação]
    """
    individual_arr = np.array(individual)
    n_coords = IND_SIZE * 2
    
    if random.random() < indpb:
        # Muta coordenadas das turbinas
        for i in range(n_coords):
            individual_arr[i] += random.gauss(mu, sigma)
        
        # Muta número de grupos
        if random.random() < 0.3:  # 30% de chance
            individual_arr[n_coords] += random.gauss(0, 0.1)
            individual_arr[n_coords] = max(0.0, min(1.0, individual_arr[n_coords]))
        
        # Muta posição da subestação (com sigma maior)
        if random.random() < 0.9:  # 90% de chance
            sigma_sub = max(sigma * 5, 200.0)
            individual_arr[n_coords + 1] += random.gauss(0, sigma_sub)
            individual_arr[n_coords + 2] += random.gauss(0, sigma_sub)
        
        # Aplica restrições
        mutated_list = individual_arr.tolist()
        enforce_circle(mutated_list[:n_coords])
        substation_pos = [mutated_list[n_coords + 1], mutated_list[n_coords + 2]]
        substation_pos = enforce_substation(substation_pos)
        mutated_list[n_coords + 1] = substation_pos[0]
        mutated_list[n_coords + 2] = substation_pos[1]
        
        for i in range(len(individual)):
            individual[i] = mutated_list[i]
    
    return individual,

def mate_baseline(ind1, ind2):
    """
    Crossover Blend para Baseline: aplica a todas as variáveis.
    """
    n_coords = IND_SIZE * 2
    
    # Crossover para coordenadas das turbinas
    tools.cxBlend(ind1[:n_coords], ind2[:n_coords], alpha=0.5)
    
    # Crossover para número de grupos
    gamma = (1. + 2. * 0.5) * random.random() - 0.5
    temp = (1. - gamma) * ind1[n_coords] + gamma * ind2[n_coords]
    ind2[n_coords] = gamma * ind1[n_coords] + (1. - gamma) * ind2[n_coords]
    ind1[n_coords] = temp
    ind1[n_coords] = max(0.0, min(1.0, ind1[n_coords]))
    ind2[n_coords] = max(0.0, min(1.0, ind2[n_coords]))
    
    # Crossover para posição da subestação
    tools.cxBlend(ind1[n_coords+1:n_coords+3], ind2[n_coords+1:n_coords+3], alpha=0.5)
    
    return ind1, ind2

toolbox_baseline.register("mate", mate_baseline)
toolbox_baseline.register("mutate", mutate_baseline, mu=0, sigma=SIGMA, indpb=INDPB)
toolbox_baseline.register("select", tools.selNSGA2)

# =============================================================================
# MÉTODO BASELINE (NSGA-II)
# =============================================================================

def run_baseline_method(seed, global_coords_pop=None, pool=None, run_id=None, output_dir=None):
    """
    Executa método Baseline: NSGA-II puro.
    
    Evolui população inicial de turbinas maximizando AEP e minimizando custo.
    Usa população inicial GLOBAL compartilhada para comparação justa.
    
    Args:
        seed: Seed para reprodutibilidade
        global_coords_pop: População inicial global de coordenadas (compartilhada)
        pool: Pool de processos para paralelização (opcional)
        run_id: ID da execução (para salvar métricas de convergência)
        output_dir: Diretório de saída (para salvar métricas de convergência)
    
    Returns:
        pareto_front: Lista de soluções não-dominadas
        hv_history: Lista de tuplas (geração, hipervolume) calculado a cada 20 gerações
    """
    global _global_coords_population, _global_coords_index
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Registra pool.map no toolbox se pool for fornecido
    if pool is not None:
        toolbox_baseline.register("map", pool.map)
    else:
        toolbox_baseline.register("map", map)
    
    # Define população global (compartilhada)
    if global_coords_pop is not None:
        _global_coords_population = global_coords_pop
        _global_coords_index = 0  # Reset índice
    
    # Cria população inicial usando coordenadas da população global
    pop = toolbox_baseline.population(n=POP_SIZE)
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox_baseline.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Filtra apenas soluções válidas
    pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
    
    # ESTRATÉGIA ROBUSTA MULTI-NÍVEL: Tenta múltiplas estratégias até encontrar soluções válidas
    if len(pop) == 0:
        print(f"   AVISO: População inicial sem soluções válidas. Tentando estratégias robustas...")
        
        # Estratégia 1: Conservadora padrão (60-80% grupos)
        print(f"      Tentativa 1/4: Estratégia conservadora (60-80% grupos)...")
        pop_conservative = []
        for _ in range(POP_SIZE):
            ind = create_baseline_individual(conservative=True)
            pop_conservative.append(ind)
        
        invalid_ind = [ind for ind in pop_conservative if not ind.fitness.valid]
        # Usa avaliação com penalidades reduzidas na inicialização
        fits = list(map(evaluate_baseline_init, invalid_ind))
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        pop = [ind for ind in pop_conservative if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 2: Ultra-conservadora com muitos grupos (80-95% do máximo)
        if len(pop) == 0:
            print(f"      Tentativa 2/4: Estratégia ultra-conservadora (80-95% grupos)...")
            pop_ultra = []
            for _ in range(POP_SIZE):
                # Força número de grupos alto (80-95% do máximo)
                n_grupos_norm = random.uniform(0.8, 0.95)
                ind = create_baseline_individual(conservative=True, n_grupos_target=n_grupos_norm)
                pop_ultra.append(ind)
            
            invalid_ind = [ind for ind in pop_ultra if not ind.fitness.valid]
            fits = list(map(evaluate_baseline_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            pop = [ind for ind in pop_ultra if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 3: Número de grupos próximo ao número de turbinas (máxima segurança)
        if len(pop) == 0:
            print(f"      Tentativa 3/4: Máximo número de grupos (próximo a {IND_SIZE} grupos)...")
            pop_max_groups = []
            for _ in range(POP_SIZE):
                # Usa número de grupos próximo ao número de turbinas (normalizado)
                # Ex: 36 turbinas -> 30-36 grupos = 0.85-1.0 normalizado
                n_grupos_target = random.uniform(0.85, 1.0)
                ind = create_baseline_individual(conservative=True, n_grupos_target=n_grupos_target)
                pop_max_groups.append(ind)
            
            invalid_ind = [ind for ind in pop_max_groups if not ind.fitness.valid]
            fits = list(map(evaluate_baseline_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            pop = [ind for ind in pop_max_groups if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 4: Última tentativa - número de grupos fixo no máximo seguro
        if len(pop) == 0:
            print(f"      Tentativa 4/4: Número de grupos fixo no máximo ({IND_SIZE} grupos)...")
            pop_fixed = []
            # Usa número de grupos igual ao número de turbinas (máxima segurança)
            n_grupos_fixed = 1.0  # Normalizado = 1.0 = MAX_GRUPOS
            for _ in range(POP_SIZE):
                ind = create_baseline_individual(conservative=True, n_grupos_target=n_grupos_fixed)
                pop_fixed.append(ind)
            
            invalid_ind = [ind for ind in pop_fixed if not ind.fitness.valid]
            fits = list(map(evaluate_baseline_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            pop = [ind for ind in pop_fixed if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        if len(pop) == 0:
            print(f"   ERRO: Todas as estratégias falharam. Nenhuma solução válida encontrada.")
            print(f"   Isso pode indicar problemas graves com cabeamento ou configuração do problema.")
            print(f"   Verifique: MIN_GRUPOS={MIN_GRUPOS}, MAX_GRUPOS={MAX_GRUPOS}, IND_SIZE={IND_SIZE}")
            return [], []
        else:
            print(f"   ✓ Estratégia robusta funcionou: {len(pop)} soluções válidas encontradas")
    
    # Histórico de hipervolume
    hv_history = []
    
    # Loop NSGA-II
    for gen in range(1, NGEN_BASELINE + 1):
        # Seleção
        offspring = toolbox_baseline.select(pop, len(pop))
        offspring = [toolbox_baseline.clone(ind) for ind in offspring]
        
        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox_baseline.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        # Mutação
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_baseline.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # Avalia novos indivíduos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox_baseline.map(toolbox_baseline.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        # Seleção NSGA-II (pop + offspring)
        combined = pop + offspring
        pop = toolbox_baseline.select(combined, POP_SIZE)
        
        # Calcula hipervolume a cada 20 gerações e coleta métricas de convergência
        if gen % 20 == 0:
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            if len(valid_pop) > 0:
                # Cria frente de Pareto temporária para calcular hipervolume
                temp_pf = tools.ParetoFront()
                temp_pf.update(valid_pop)
                hv = calculate_hypervolume(list(temp_pf))
                hv_history.append((gen, hv))
                
                # Coleta métricas de convergência para convergence_history.csv
                if run_id is not None and output_dir is not None:
                    conv_metrics = collect_convergence_metrics(
                        run_id, 'Baseline', gen, pop,
                        pareto_front=list(temp_pf), ref_point=None,
                        hypervolume_func=calculate_hypervolume
                    )
                    save_convergence_history(conv_metrics, output_dir=output_dir, append=True)
        
        if gen % 100 == 0:
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            if len(valid_pop) > 0:
                aep_max = max(ind.fitness.values[0] for ind in valid_pop)
                cost_min = min(ind.fitness.values[1] for ind in valid_pop)
                print(f"Gen {gen}: AEP Max={aep_max:.2f} MWh, Cost Min={cost_min:.2f} USD, Valid={len(valid_pop)}/{len(pop)}")
    
    # Retorna frente de Pareto
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop)
    valid_solutions = [ind for ind in pareto_front if ind.fitness.valid and ind.fitness.values[0] > 0]
    
    return valid_solutions, hv_history

# =============================================================================
# MÉTODO SEQUENTIAL - FASE 1 (IGUAL AO wind_farm_GA_16.py)
# =============================================================================

# Variável global para índice de coordenadas (Fase 1)
_global_coords_index_p1 = 0

def create_phase1_individual():
    """
    Cria indivíduo Fase 1 usando coordenadas da população inicial GLOBAL.
    Garante que Baseline, Sequential e Proposed começam com as mesmas coordenadas.
    """
    global _global_coords_population, _global_coords_index_p1
    
    # Usa coordenadas da população global (cíclica)
    if _global_coords_population is None or len(_global_coords_population) == 0:
        # Fallback: usa coordenadas exatas do YAML se população global não estiver definida
        coords = initial_coordinates.flatten().tolist()
    else:
        coords_array = _global_coords_population[_global_coords_index_p1 % len(_global_coords_population)]
        _global_coords_index_p1 += 1
        coords = coords_array.flatten().tolist()
    
    return creator.IndividualPhase1(coords)

def enforce_circle_phase1(individual):
    """Projeta coordenadas para dentro do círculo (Fase 1)."""
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            individual[2*i] = CIRCLE_RADIUS * np.cos(angle)
            individual[2*i + 1] = CIRCLE_RADIUS * np.sin(angle)
    return individual

def evaluate_phase1(individual):
    """
    Avalia indivíduo Fase 1: calcula AEP bruto (sem perdas).
    Igual ao evaluate_otimizado do wind_farm_GA_16.py
    """
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Penalidades
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6
    
    num_turb = len(turb_coords)
    if num_turb > 1:
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6
    else:
        penalty_close_turbines = 0
    
    # AEP bruto
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1],
                  TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3])
    
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    return fitness,

def mutate_phase1(individual, mu, sigma, indpb):
    """Mutação para Fase 1 (igual ao wind_farm_GA_16.py)."""
    individual_arr = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual_arr)):
            individual_arr[i] += random.gauss(mu, sigma)
        enforce_circle_phase1(individual_arr)
    return creator.IndividualPhase1(individual_arr.tolist()),

toolbox_seq_p1.register("individual", create_phase1_individual)
toolbox_seq_p1.register("population", tools.initRepeat, list, toolbox_seq_p1.individual)
toolbox_seq_p1.register("evaluate", evaluate_phase1)
toolbox_seq_p1.register("mate", tools.cxBlend, alpha=0.5)
toolbox_seq_p1.register("mutate", mutate_phase1, mu=0, sigma=100, indpb=0.4)
toolbox_seq_p1.register("select", tools.selTournament, tournsize=5)

def run_sequential_phase1(seed, return_top_n=None, ngen=None, global_coords_pop=None, pool=None):
    """
    Executa Fase 1 do Sequential/Proposed: maximiza AEP bruto.
    Implementação idêntica ao wind_farm_GA_16.py
    
    Args:
        seed: Seed para reprodutibilidade
        return_top_n: Se especificado, retorna os top N layouts (para Proposed)
        ngen: Número de gerações (se None, usa NGEN_SEQUENTIAL_P1)
        global_coords_pop: População inicial global de coordenadas (compartilhada)
        pool: Pool de processos para paralelização (opcional)
    
    Returns:
        Se return_top_n=None: melhor layout único (para Sequential)
        Se return_top_n especificado: lista dos top N layouts (para Proposed)
    """
    global _global_coords_population, _global_coords_index_p1
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Registra pool.map no toolbox se pool for fornecido
    if pool is not None:
        toolbox_seq_p1.register("map", pool.map)
    else:
        toolbox_seq_p1.register("map", map)
    
    # Define população global (compartilhada)
    if global_coords_pop is not None:
        _global_coords_population = global_coords_pop
        _global_coords_index_p1 = 0  # Reset índice
    
    pop = toolbox_seq_p1.population(n=POP_SIZE)
    hof_size = return_top_n if return_top_n else 1
    hof = tools.HallOfFame(maxsize=hof_size)
    
    # Parâmetros (igual ao wind_farm_GA_16.py)
    PATIENCE = 50
    MIN_DELTA = 10.0
    CXPB = 0.95
    MUTPB = 0.7
    SIGMA_NORMAL = 100
    SIGMA_AGGRESSIVE = 250
    AGGRESSIVE_DURATION = 15
    
    # Número de gerações
    if ngen is None:
        ngen = NGEN_SEQUENTIAL_P1
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox_seq_p1.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    hof.update(pop)
    last_max_fitness = hof[0].fitness.values[0]
    
    stagnation_counter = 0
    aggressive_phase_triggered = False
    aggressive_phase_countdown = 0
    
    for gen in range(1, ngen + 1):
        current_max_fitness = hof[0].fitness.values[0]
        
        if (current_max_fitness - last_max_fitness) < MIN_DELTA:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        last_max_fitness = current_max_fitness
        
        # Lógica de mutação adaptativa (sem early stopping para garantir todas as gerações)
        if stagnation_counter >= PATIENCE:
            if not aggressive_phase_triggered:
                toolbox_seq_p1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)
                aggressive_phase_triggered = True
                aggressive_phase_countdown = AGGRESSIVE_DURATION
                stagnation_counter = 0
            # Continua executando todas as gerações (removido early stopping)
            # A mutação agressiva ajuda a escapar de ótimos locais, mas não para o algoritmo
        
        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                toolbox_seq_p1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)
        
        # Seleção
        offspring = toolbox_seq_p1.select(pop, len(pop))
        offspring = [toolbox_seq_p1.clone(ind) for ind in offspring]
        
        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox_seq_p1.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        # Mutação
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_seq_p1.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # Avalia novos indivíduos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox_seq_p1.map(toolbox_seq_p1.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        pop[:] = offspring
        hof.update(pop)
        
        # Print de progresso a cada 100 gerações
        if gen % 100 == 0 or gen == ngen:
            current_best_aep = hof[0].fitness.values[0]
            print(f"Gen {gen}: AEP Max={current_best_aep:.2f} MWh")
    
    if return_top_n:
        # Retorna top N layouts ordenados por fitness
        top_layouts = sorted(hof, key=lambda x: x.fitness.values[0], reverse=True)
        return top_layouts[:return_top_n]
    else:
        return hof[0]  # Retorna melhor layout único

# =============================================================================
# MÉTODO SEQUENTIAL - FASE 2 (OTIMIZAÇÃO DE CABEAMENTO)
# =============================================================================

def create_sequential_individual(best_turbine_layout):
    """
    Cria indivíduo Fase 2 com 3 genes:
    - n_grupos_norm [0, 1]
    - sub_x, sub_y (coordenadas subestação)
    """
    # Número de grupos inicial (normalizado)
    n_grupos_norm = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))
    
    # Subestação: centroide das turbinas deslocado
    turb_coords = np.array(best_turbine_layout).reshape((IND_SIZE, 2))
    centroid = np.mean(turb_coords, axis=0)
    sub_pos = displace_substation_from_turbines(centroid, turb_coords, min_distance=MIN_SUB_TURB_DIST)
    
    # Genoma: [n_grupos_norm, sub_x, sub_y]
    genome = [n_grupos_norm, sub_pos[0], sub_pos[1]]
    return creator.IndividualSequential(genome)

def evaluate_sequential(individual, best_turbine_layout):
    """
    Avalia indivíduo Fase 2: minimiza custo de cabeamento.
    Turbinas são fixas (best_turbine_layout).
    """
    n_grupos_norm = individual[0]
    sub_pos = np.array([individual[1], individual[2]])
    
    # Converte número de grupos
    n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
    n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
    n_grupos = min(n_grupos, IND_SIZE)
    
    turb_coords = np.array(best_turbine_layout).reshape((IND_SIZE, 2))
    
    # Penalidades
    penalty_sub_out = np.maximum(0, np.linalg.norm(sub_pos) - CIRCLE_RADIUS) * 1e5
    min_dist_sub_turb = np.min(np.linalg.norm(turb_coords - sub_pos, axis=1))
    penalty_sub_close = np.maximum(0, MIN_SUB_TURB_DIST - min_dist_sub_turb) * 1e5
    
    # Calcula cabeamento
    coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
    substation_idx = IND_SIZE  # Índice da subestação após todas as turbinas
    try:
        plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=substation_idx, n_grupos=n_grupos)
        custo_usd = res['custo_total_usd']
        
        # SEMPRE detecta cruzamentos, proximidade excessiva e múltiplas conexões
        # Esta penalidade é aplicada globalmente em todos os métodos
        penalty_overlap = detectar_sobreposicao_cabos(
            plant.paths, coords_all, min_distance=MIN_CABLE_DISTANCE, substation_idx=substation_idx)
    except Exception as e:
        # Em caso de erro no cabeamento, aplica penalidade extrema
        return 1e12,  # Penalidade extrema (maximizar negativo = minimizar custo)
    
    # Custo final (com penalidades)
    custo_final = custo_usd + penalty_sub_out + penalty_sub_close + penalty_overlap
    
    # Retorna custo (FitnessMin já minimiza o valor)
    return custo_final,

def mutate_sequential(individual, best_turbine_layout, mu, sigma_n_grupos, sigma_sub, indpb):
    """
    Mutação para Fase 2: diferenciada para n_grupos e subestação.
    """
    if random.random() < indpb:
        # Muta número de grupos (sigma pequeno)
        if random.random() < 0.3:
            individual[0] += random.gauss(0, sigma_n_grupos)
            individual[0] = max(0.0, min(1.0, individual[0]))
        
        # Muta posição da subestação (sigma maior)
        if random.random() < 0.9:
            individual[1] += random.gauss(0, sigma_sub)
            individual[2] += random.gauss(0, sigma_sub)
            
            # Aplica restrições
            sub_pos = [individual[1], individual[2]]
            sub_pos = enforce_substation(sub_pos)
            individual[1] = sub_pos[0]
            individual[2] = sub_pos[1]
    
    return individual,

def mate_sequential(ind1, ind2):
    """Crossover Blend para os 3 genes."""
    # Crossover para n_grupos
    gamma = (1. + 2. * 0.5) * random.random() - 0.5
    temp = (1. - gamma) * ind1[0] + gamma * ind2[0]
    ind2[0] = gamma * ind1[0] + (1. - gamma) * ind2[0]
    ind1[0] = temp
    ind1[0] = max(0.0, min(1.0, ind1[0]))
    ind2[0] = max(0.0, min(1.0, ind2[0]))
    
    # Crossover para subestação
    tools.cxBlend(ind1[1:3], ind2[1:3], alpha=0.5)
    
    return ind1, ind2

# Funções wrapper pickleáveis para multiprocessing
def _eval_sequential_wrapper(individual, best_turbine_layout):
    """Wrapper pickleável para evaluate_sequential."""
    return evaluate_sequential(individual, best_turbine_layout)

def _mutate_sequential_wrapper(individual, best_turbine_layout, mu=0, sigma_n_grupos=0.1, sigma_sub=100.0, indpb=0.4):
    """Wrapper pickleável para mutate_sequential."""
    return mutate_sequential(individual, best_turbine_layout, 
                           mu=mu, sigma_n_grupos=sigma_n_grupos, sigma_sub=sigma_sub, indpb=indpb)

def run_sequential_phase2(seed, best_turbine_layout, pool=None):
    """
    Executa Fase 2 do Sequential: minimiza custo de cabeamento.
    
    Args:
        seed: Seed para reprodutibilidade
        best_turbine_layout: Layout de turbinas da Fase 1
        pool: Pool de processos para paralelização (opcional)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Garante que seja uma lista simples (não um objeto IndividualPhase1) para serialização
    if best_turbine_layout is None:
        raise ValueError("best_turbine_layout não pode ser None")
    
    # Converte para lista simples de floats (garante serialização correta para multiprocessing)
    best_turbine_layout_list = [float(x) for x in best_turbine_layout]
    
    # Validação: verifica se o layout tem o tamanho correto
    if len(best_turbine_layout_list) != IND_SIZE * 2:
        raise ValueError(f"Layout de turbinas inválido: esperado {IND_SIZE * 2} elementos, recebido {len(best_turbine_layout_list)}")
    
    # Configura toolbox com layout fixo
    toolbox_seq_p2.register("individual", create_sequential_individual, best_turbine_layout=best_turbine_layout_list)
    toolbox_seq_p2.register("population", tools.initRepeat, list, toolbox_seq_p2.individual)
    
    # Função de avaliação com layout fixo - usa partial para passar best_turbine_layout como argumento
    toolbox_seq_p2.register("evaluate", partial(_eval_sequential_wrapper, best_turbine_layout=best_turbine_layout_list))
    
    # Mutação com layout fixo - usa partial para passar best_turbine_layout como argumento
    sigma_sub_sequential = 100.0  # Sigma da subestação: aumentado para permitir mais exploração
    toolbox_seq_p2.register("mutate", partial(_mutate_sequential_wrapper, 
                                              best_turbine_layout=best_turbine_layout_list,
                                              mu=0, sigma_n_grupos=0.1, sigma_sub=sigma_sub_sequential, indpb=0.4))
    
    toolbox_seq_p2.register("mate", mate_sequential)
    toolbox_seq_p2.register("select", tools.selTournament, tournsize=5)
    
    # Registra pool.map no toolbox se pool for fornecido
    if pool is not None:
        toolbox_seq_p2.register("map", pool.map)
    else:
        toolbox_seq_p2.register("map", map)
    
    pop = toolbox_seq_p2.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fits = toolbox_seq_p2.map(toolbox_seq_p2.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    hof.update(pop)
    
    # Loop GA (sem mutação adaptativa na Fase 2)
    for gen in range(1, NGEN_SEQUENTIAL_P2 + 1):
        # Seleção
        offspring = toolbox_seq_p2.select(pop, len(pop))
        offspring = [toolbox_seq_p2.clone(ind) for ind in offspring]
        
        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox_seq_p2.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        # Mutação
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_seq_p2.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # Avalia novos indivíduos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox_seq_p2.map(toolbox_seq_p2.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        # Elitismo: mantém o melhor indivíduo da geração anterior
        combined = pop + offspring
        # Seleciona os melhores (mantém diversidade mas preserva o melhor)
        pop[:] = toolbox_seq_p2.select(combined, len(pop))
        hof.update(pop)
        
        # Print de progresso a cada 100 gerações
        if gen % 100 == 0 or gen == NGEN_SEQUENTIAL_P2:
            current_best_cost = hof[0].fitness.values[0]
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] < 1e12]
            print(f"Gen {gen}: Cost Min={current_best_cost:.2f} USD, Valid={len(valid_pop)}/{len(pop)}")
    
    return hof[0], best_turbine_layout  # Retorna melhor solução Fase 2 + layout fixo

def run_sequential_method(seed, global_coords_pop=None, pool=None):
    """
    Executa método Sequential completo:
    1. Fase 1: Maximiza AEP bruto (layout turbinas)
    2. Fase 2: Minimiza custo (subestação + cabeamento, turbinas fixas)
    
    Args:
        seed: Seed para reprodutibilidade
        global_coords_pop: População inicial global de coordenadas (compartilhada)
        pool: Pool de processos para paralelização (opcional)
    
    Returns:
        best_individual: Melhor solução Fase 2 (genoma de 3 genes)
        best_turbine_layout: Layout de turbinas da Fase 1 (IND_SIZE*2 genes)
    """
    print(">>> Executando Sequential - Fase 1 (Maximizar AEP)...")
    best_turbine_layout = run_sequential_phase1(seed, global_coords_pop=global_coords_pop, pool=pool)
    
    print(f">>> Executando Sequential - Fase 2 (Minimizar Custo)...")
    best_sequential, _ = run_sequential_phase2(seed, best_turbine_layout, pool=pool)
    
    return best_sequential, best_turbine_layout

# =============================================================================
# MÉTODO PROPOSED - FASE 2 (NSGA-II MULTIOBJETIVO)
# =============================================================================

def create_individual_proposed_phase2_from_coords(coords, substation_pos=None):
    """
    Cria um indivíduo da Fase 2 Proposed a partir de coordenadas da Fase 1.
    
    Estrutura do genoma: [32 coords turbinas] + [1 n_grupos] + [2 coords subestação] = 35 variáveis
    Reutiliza IndividualBaseline pois tem a mesma estrutura.
    """
    if isinstance(coords, list):
        coords = np.array(coords)
    
    if coords.ndim == 2:
        coords_flat = coords.flatten().tolist()
    else:
        coords_flat = coords.tolist() if isinstance(coords, np.ndarray) else list(coords)
    
    if len(coords_flat) != IND_SIZE * 2:
        raise ValueError(f"Coordenadas devem ter {IND_SIZE * 2} elementos, recebido {len(coords_flat)}")
    
    n_grupos_normalizado = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    
    if substation_pos is None:
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(CIRCLE_RADIUS * 0.2, CIRCLE_RADIUS * 0.7)
        substation_pos = [radius * np.cos(angle), radius * np.sin(angle)]
    
    substation_pos = enforce_substation(np.array(substation_pos)).tolist()
    
    return creator.IndividualBaseline(coords_flat + [n_grupos_normalizado] + list(substation_pos))

def evaluate_proposed_phase2(individual, init_mode=False):
    """
    Avaliação Fase 2 Proposed: igual ao evaluate_baseline (AEP líquido + Custo).
    Inclui penalidades por cruzamento de cabos e proximidade excessiva.
    """
    return evaluate_baseline(individual, init_mode=init_mode)

def mutate_proposed_phase2(individual, mu, sigma, indpb):
    """
    Mutação para Fase 2 Proposed: aplica mutação gaussiana diferenciada.
    Similar ao multi16_prioriza_aep.py.
    """
    individual_arr = np.array(individual)
    n_coords = IND_SIZE * 2
    
    if random.random() < indpb:
        # Muta coordenadas das turbinas
        for i in range(n_coords):
            individual_arr[i] += random.gauss(mu, sigma)
        
        # Muta número de grupos
        if random.random() < PROB_MUTATE_GROUPS_P2:
            individual_arr[n_coords] += random.gauss(0, 0.1)
            individual_arr[n_coords] = max(0.0, min(1.0, individual_arr[n_coords]))
        
        # Muta posição da subestação
        if random.random() < PROB_MUTATE_SUBSTATION_P2:
            mutation_sigma_sub = max(sigma * SIGMA_SUB_MULTIPLIER, SIGMA_SUB_MIN)
            individual_arr[n_coords + 1] += random.gauss(0, mutation_sigma_sub)
            individual_arr[n_coords + 2] += random.gauss(0, mutation_sigma_sub)
        
        mutated_list = individual_arr.tolist()
        enforce_circle(mutated_list[:n_coords])
        
        substation_pos_mutated = [mutated_list[n_coords + 1], mutated_list[n_coords + 2]]
        substation_pos_mutated = enforce_substation(substation_pos_mutated)
        mutated_list[n_coords + 1] = substation_pos_mutated[0]
        mutated_list[n_coords + 2] = substation_pos_mutated[1]
        
        for i in range(len(individual)):
            individual[i] = mutated_list[i]
    
    return individual,

def mate_proposed_phase2(ind1, ind2):
    """
    Crossover Blend para Fase 2 Proposed: aplica a todas as variáveis.
    Similar ao multi16_prioriza_aep.py.
    """
    n_coords = IND_SIZE * 2
    
    # Crossover para coordenadas das turbinas
    tools.cxBlend(ind1[:n_coords], ind2[:n_coords], alpha=0.5)
    
    # Crossover para número de grupos
    gamma = (1. + 2. * 0.5) * random.random() - 0.5
    temp = (1. - gamma) * ind1[n_coords] + gamma * ind2[n_coords]
    ind2[n_coords] = gamma * ind1[n_coords] + (1. - gamma) * ind2[n_coords]
    ind1[n_coords] = temp
    ind1[n_coords] = max(0.0, min(1.0, ind1[n_coords]))
    ind2[n_coords] = max(0.0, min(1.0, ind2[n_coords]))
    
    # Crossover para posição da subestação
    tools.cxBlend(ind1[n_coords+1:n_coords+3], ind2[n_coords+1:n_coords+3], alpha=0.5)
    
    return ind1, ind2

toolbox_prop_p2.register("evaluate", evaluate_proposed_phase2)

# Wrapper para avaliação na inicialização (com penalidades reduzidas)
def evaluate_proposed_phase2_init(individual):
    """Wrapper para evaluate_proposed_phase2 com init_mode=True (penalidades reduzidas)."""
    return evaluate_proposed_phase2(individual, init_mode=True)
toolbox_prop_p2.register("mate", mate_proposed_phase2)
toolbox_prop_p2.register("mutate", mutate_proposed_phase2, mu=0, sigma=SIGMA, indpb=INDPB)
toolbox_prop_p2.register("select", tools.selNSGA2)

def run_proposed_phase2(seed, best_layouts_phase1, pool=None, run_id=None, output_dir=None):
    """
    Executa Fase 2 do Proposed: NSGA-II multiobjetivo.
    Similar ao optimize_phase2 do multi16_prioriza_aep.py.
    
    Args:
        seed: Seed para reprodutibilidade
        best_layouts_phase1: Lista dos melhores layouts da Fase 1
        pool: Pool de processos para paralelização (opcional)
        run_id: ID da execução (para salvar métricas de convergência)
        output_dir: Diretório de saída (para salvar métricas de convergência)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Registra pool.map no toolbox se pool for fornecido
    if pool is not None:
        toolbox_prop_p2.register("map", pool.map)
    else:
        toolbox_prop_p2.register("map", map)
    
    # Inicializa população: top layouts da Fase 1
    pop = []
    for i, layout in enumerate(best_layouts_phase1):
        coords_list = list(layout)
        if len(coords_list) != IND_SIZE * 2:
            continue
        
        coords_array = np.array(coords_list, dtype=float)
        coords = coords_array.reshape((IND_SIZE, 2))
        centroide = np.mean(coords, axis=0)
        
        # Diversifica posição inicial da subestação
        mod = len(pop) % 5
        if mod == 0:
            substation_pos = centroide.tolist()
        elif mod == 1:
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.2),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.2)
            ]
        elif mod == 2:
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.4),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.4)
            ]
        elif mod == 3:
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.6),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.6)
            ]
        else:  # mod == 4
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(CIRCLE_RADIUS * 0.5, CIRCLE_RADIUS * 0.9)
            substation_pos = [
                centroide[0] + radius * np.cos(angle),
                centroide[1] + radius * np.sin(angle)
            ]
        
        substation_pos = enforce_substation(np.array(substation_pos)).tolist()
        ind_phase2 = create_individual_proposed_phase2_from_coords(coords, substation_pos=substation_pos)
        pop.append(ind_phase2)
    
    # Completa população com perturbações dos melhores layouts
    n_remaining = POP_SIZE - len(pop)
    if n_remaining > 0:
        for i in range(n_remaining):
            layout_idx = i % len(best_layouts_phase1)
            layout = best_layouts_phase1[layout_idx]
            
            coords_list = list(layout)
            coords_array = np.array(coords_list, dtype=float)
            coords = coords_array.reshape((IND_SIZE, 2))
            
            coords_perturbed = coords.copy()
            perturbation_sigma = random.uniform(PERTURBATION_SIGMA_MIN, PERTURBATION_SIGMA_MAX)
            for j in range(len(coords_perturbed)):
                coords_perturbed[j, 0] += random.gauss(0, perturbation_sigma)
                coords_perturbed[j, 1] += random.gauss(0, perturbation_sigma)
            
            # Garante que está dentro do círculo
            for j in range(len(coords_perturbed)):
                x, y = coords_perturbed[j, 0], coords_perturbed[j, 1]
                if not is_within_circle(x, y, CIRCLE_RADIUS):
                    angle = np.arctan2(y, x)
                    coords_perturbed[j, 0] = CIRCLE_RADIUS * np.cos(angle)
                    coords_perturbed[j, 1] = CIRCLE_RADIUS * np.sin(angle)
            
            centroide_perturbed = np.mean(coords_perturbed, axis=0)
            substation_pos_perturbed = [
                centroide_perturbed[0] + random.gauss(0, CIRCLE_RADIUS * 0.4),
                centroide_perturbed[1] + random.gauss(0, CIRCLE_RADIUS * 0.4)
            ]
            substation_pos_perturbed = enforce_substation(np.array(substation_pos_perturbed)).tolist()
            
            ind_phase2 = create_individual_proposed_phase2_from_coords(coords_perturbed, substation_pos=substation_pos_perturbed)
            pop.append(ind_phase2)
    
    # Avalia população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox_prop_p2.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Filtra apenas soluções válidas
    pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
    
    # ESTRATÉGIA ROBUSTA MULTI-NÍVEL: Tenta múltiplas estratégias até encontrar soluções válidas
    if len(pop) == 0:
        print(f"   AVISO: População inicial Proposed Fase 2 sem soluções válidas. Tentando estratégias robustas...")
        
        def create_conservative_pop_proposed(n_grupos_min_norm, n_grupos_max_norm):
            """Helper para criar população conservadora com faixa de grupos especificada."""
            pop_cons = []
            for i, layout in enumerate(best_layouts_phase1):
                coords_list = list(layout)
                if len(coords_list) != IND_SIZE * 2:
                    continue
                
                coords_array = np.array(coords_list, dtype=float)
                coords = coords_array.reshape((IND_SIZE, 2))
                centroide = np.mean(coords, axis=0)
                
                n_grupos_normalizado = random.uniform(n_grupos_min_norm, n_grupos_max_norm)
                substation_pos = displace_substation_from_turbines(centroide, coords, min_distance=MIN_SUB_TURB_DIST)
                substation_pos = enforce_substation(np.array(substation_pos)).tolist()
                
                ind_phase2 = create_individual_proposed_phase2_from_coords(coords, substation_pos=substation_pos)
                n_coords = IND_SIZE * 2
                ind_phase2[n_coords] = n_grupos_normalizado
                pop_cons.append(ind_phase2)
            
            # Completa população
            n_remaining = POP_SIZE - len(pop_cons)
            if n_remaining > 0:
                for i in range(n_remaining):
                    layout_idx = i % len(best_layouts_phase1)
                    layout = best_layouts_phase1[layout_idx]
                    coords_list = list(layout)
                    coords_array = np.array(coords_list, dtype=float)
                    coords = coords_array.reshape((IND_SIZE, 2))
                    
                    # Perturbação menor
                    coords_perturbed = coords.copy()
                    perturbation_sigma = PERTURBATION_SIGMA_MIN * 0.5
                    for j in range(len(coords_perturbed)):
                        coords_perturbed[j, 0] += random.gauss(0, perturbation_sigma)
                        coords_perturbed[j, 1] += random.gauss(0, perturbation_sigma)
                    
                    # Garante que está dentro do círculo
                    for j in range(len(coords_perturbed)):
                        x, y = coords_perturbed[j, 0], coords_perturbed[j, 1]
                        if not is_within_circle(x, y, CIRCLE_RADIUS):
                            angle = np.arctan2(y, x)
                            coords_perturbed[j, 0] = CIRCLE_RADIUS * np.cos(angle)
                            coords_perturbed[j, 1] = CIRCLE_RADIUS * np.sin(angle)
                    
                    centroide_perturbed = np.mean(coords_perturbed, axis=0)
                    substation_pos_perturbed = displace_substation_from_turbines(
                        centroide_perturbed, coords_perturbed, min_distance=MIN_SUB_TURB_DIST)
                    substation_pos_perturbed = enforce_substation(np.array(substation_pos_perturbed)).tolist()
                    
                    n_grupos_normalizado = random.uniform(n_grupos_min_norm, n_grupos_max_norm)
                    ind_phase2 = create_individual_proposed_phase2_from_coords(
                        coords_perturbed, substation_pos=substation_pos_perturbed)
                    n_coords = IND_SIZE * 2
                    ind_phase2[n_coords] = n_grupos_normalizado
                    pop_cons.append(ind_phase2)
            
            return pop_cons
        
        # Estratégia 1: Conservadora padrão (60-80% grupos)
        print(f"      Tentativa 1/4: Estratégia conservadora (60-80% grupos)...")
        pop_conservative = create_conservative_pop_proposed(0.6, 0.8)
        invalid_ind = [ind for ind in pop_conservative if not ind.fitness.valid]
        # Usa avaliação com penalidades reduzidas na inicialização
        fits = list(map(evaluate_proposed_phase2_init, invalid_ind))
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        pop = [ind for ind in pop_conservative if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 2: Ultra-conservadora (80-95% grupos)
        if len(pop) == 0:
            print(f"      Tentativa 2/4: Estratégia ultra-conservadora (80-95% grupos)...")
            pop_ultra = create_conservative_pop_proposed(0.8, 0.95)
            invalid_ind = [ind for ind in pop_ultra if not ind.fitness.valid]
            fits = list(map(evaluate_proposed_phase2_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            pop = [ind for ind in pop_ultra if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 3: Máximo número de grupos (85-100%)
        if len(pop) == 0:
            print(f"      Tentativa 3/4: Máximo número de grupos (85-100%)...")
            pop_max = create_conservative_pop_proposed(0.85, 1.0)
            invalid_ind = [ind for ind in pop_max if not ind.fitness.valid]
            fits = list(map(evaluate_proposed_phase2_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            pop = [ind for ind in pop_max if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        # Estratégia 4: Número de grupos fixo no máximo
        if len(pop) == 0:
            print(f"      Tentativa 4/4: Número de grupos fixo no máximo ({IND_SIZE} grupos)...")
            pop_fixed = create_conservative_pop_proposed(1.0, 1.0)  # Força máximo
            invalid_ind = [ind for ind in pop_fixed if not ind.fitness.valid]
            fits = list(map(evaluate_proposed_phase2_init, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            pop = [ind for ind in pop_fixed if ind.fitness.valid and ind.fitness.values[0] > 0]
        
        if len(pop) == 0:
            print(f"   ERRO: Todas as estratégias falharam. Nenhuma solução válida encontrada.")
            return [], []
        else:
            print(f"   ✓ Estratégia robusta funcionou: {len(pop)} soluções válidas encontradas")
    
    # Histórico de hipervolume
    hv_history = []
    
    # Loop NSGA-II
    stagnation_counter = 0
    last_best_aep = 0.0
    last_best_cost = float('inf')
    
    for gen in range(1, NGEN_PROPOSED_P2 + 1):
        # Seleção
        offspring = toolbox_prop_p2.select(pop, len(pop))
        offspring = [toolbox_prop_p2.clone(ind) for ind in offspring]
        
        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox_prop_p2.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        # Mutação
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox_prop_p2.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # Avalia novos indivíduos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox_prop_p2.map(toolbox_prop_p2.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        # Seleção NSGA-II (pop + offspring)
        combined = pop + offspring
        pop = toolbox_prop_p2.select(combined, POP_SIZE)
        
        # Detecção de estagnação
        valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
        if len(valid_pop) > 0:
            current_best_aep = max(ind.fitness.values[0] for ind in valid_pop)
            current_best_cost = min(ind.fitness.values[1] for ind in valid_pop)
            
            aep_improved = (current_best_aep - last_best_aep) >= MIN_DELTA_AEP_P2
            cost_improved = (last_best_cost - current_best_cost) >= MIN_DELTA_COST_P2
            
            if aep_improved or cost_improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if current_best_aep > last_best_aep:
                last_best_aep = current_best_aep
            if current_best_cost < last_best_cost:
                last_best_cost = current_best_cost
            
            # Removido early stopping - continua até completar todas as gerações
            # if stagnation_counter >= PATIENCE_P2:
            #     break
            
            # Calcula hipervolume a cada 20 gerações e coleta métricas de convergência
            if gen % 20 == 0:
                # Cria frente de Pareto temporária para calcular hipervolume
                temp_pf = tools.ParetoFront()
                temp_pf.update(valid_pop)
                hv = calculate_hypervolume(list(temp_pf))
                hv_history.append((gen, hv))
                
                # Coleta métricas de convergência para convergence_history.csv
                if run_id is not None and output_dir is not None:
                    conv_metrics = collect_convergence_metrics(
                        run_id, 'Proposed', gen, pop,
                        pareto_front=list(temp_pf), ref_point=None,
                        hypervolume_func=calculate_hypervolume
                    )
                    save_convergence_history(conv_metrics, output_dir=output_dir, append=True)
        
        if gen % 100 == 0:
            if len(valid_pop) > 0:
                print(f"Gen {gen}: AEP Max={current_best_aep:.2f} MWh, Cost Min={current_best_cost:.2f} USD, Valid={len(valid_pop)}/{len(pop)}")
    
    # Retorna frente de Pareto
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop)
    valid_solutions = [ind for ind in pareto_front if ind.fitness.valid and ind.fitness.values[0] > 0]
    
    return valid_solutions, hv_history

def run_proposed_method(seed, global_coords_pop=None, pool=None, run_id=None, output_dir=None):
    """
    Executa método Proposed completo:
    1. Fase 1: Maximiza AEP bruto (layout turbinas) - reutiliza Sequential Fase 1
    2. Fase 2: NSGA-II multiobjetivo (AEP líquido + Custo) usando melhores layouts da Fase 1
    
    Similar ao multi16_prioriza_aep.py: usa os top N layouts da Fase 1 como sementes.
    
    Args:
        seed: Seed para reprodutibilidade
        global_coords_pop: População inicial global de coordenadas (compartilhada)
        pool: Pool de processos para paralelização (opcional)
        run_id: ID da execução (para salvar métricas de convergência)
        output_dir: Diretório de saída (para salvar métricas de convergência)
    
    Returns:
        pareto_front: Lista de soluções não-dominadas da Fase 2
        hv_history: Lista de tuplas (geração, hipervolume) calculado a cada 20 gerações
    """
    print(">>> Executando Proposed - Fase 1 (Maximizar AEP)...")
    # Retorna os top N layouts da Fase 1 (similar ao multi16_prioriza_aep.py)
    # Usa NGEN_PROPOSED_P1 para número de gerações
    best_layouts_p1 = run_sequential_phase1(seed, return_top_n=N_TOP_LAYOUTS, ngen=NGEN_PROPOSED_P1, global_coords_pop=global_coords_pop, pool=pool)
    
    print(f">>> Executando Proposed - Fase 2 (NSGA-II Multiobjetivo)...")
    pareto_front, hv_history = run_proposed_phase2(seed, best_layouts_p1, pool=pool, run_id=run_id, output_dir=output_dir)
    
    return pareto_front, hv_history

# =============================================================================
# FUNÇÕES DE SELEÇÃO DE MELHOR SOLUÇÃO
# =============================================================================

def find_knee_point(pareto_front):
    """
    Seleciona a melhor solução usando o método do knee point (ponto de joelho).
    O knee point é a solução que minimiza a distância normalizada ao ponto ideal (máximo AEP, mínimo custo).
    Similar ao select_best_tradeoff_solution do multi16_prioriza_aep.py.
    
    Args:
        pareto_front: Lista de soluções não-dominadas (cada uma com fitness.values = (aep, custo))
    
    Returns:
        best_solution: Solução com melhor trade-off (knee point)
    """
    if len(pareto_front) == 0:
        return None
    
    if len(pareto_front) == 1:
        return pareto_front[0]
    
    # Extrai AEP e custo de todas as soluções
    aeps = np.array([ind.fitness.values[0] for ind in pareto_front])
    costs = np.array([ind.fitness.values[1] for ind in pareto_front])
    
    # Normaliza AEP e custo para [0, 1]
    # AEP: quanto maior melhor, então normaliza como (aep - min) / (max - min)
    # Custo: quanto menor melhor, então normaliza como (max - cost) / (max - min)
    aep_min, aep_max = aeps.min(), aeps.max()
    cost_min, cost_max = costs.min(), costs.max()
    
    if aep_max == aep_min:
        aep_norm = np.ones_like(aeps)
    else:
        aep_norm = (aeps - aep_min) / (aep_max - aep_min)
    
    if cost_max == cost_min:
        cost_norm = np.ones_like(costs)
    else:
        cost_norm = (cost_max - costs) / (cost_max - cost_min)  # Invertido: menor custo = maior valor
    
    # Ponto ideal: (1, 1) = máximo AEP normalizado, mínimo custo normalizado
    # Calcula distância de cada ponto ao ideal
    ideal_point = np.array([1.0, 1.0])
    distances = []
    for i in range(len(pareto_front)):
        point = np.array([aep_norm[i], cost_norm[i]])
        dist = np.linalg.norm(ideal_point - point)
        distances.append(dist)
    
    # Retorna a solução com menor distância ao ideal (knee point)
    knee_idx = np.argmin(distances)
    return pareto_front[knee_idx]

def get_lowest_cost_solution(sequential_solution):
    """
    Para Sequential, retorna a solução com menor custo.
    Como Sequential já retorna o melhor indivíduo (menor custo) da Fase 2,
    apenas retorna o indivíduo passado.
    
    Args:
        sequential_solution: Melhor solução Sequential (já é o menor custo)
    
    Returns:
        sequential_solution: Mesma solução (já é o menor custo)
    """
    return sequential_solution

def calculate_spread(pareto_front):
    """
    Calcula o spread (diversidade) da frente de Pareto.
    Spread mede a distribuição das soluções ao longo da frente.
    
    Args:
        pareto_front: Lista de soluções não-dominadas
    
    Returns:
        spread: Valor do spread (0 = todas soluções iguais, maior = mais diverso)
    """
    if len(pareto_front) <= 1:
        return 0.0
    
    aeps = np.array([ind.fitness.values[0] for ind in pareto_front])
    costs = np.array([ind.fitness.values[1] for ind in pareto_front])
    
    # Normaliza
    aep_range = aeps.max() - aeps.min()
    cost_range = costs.max() - costs.min()
    
    if aep_range == 0 and cost_range == 0:
        return 0.0
    
    # Calcula distâncias entre soluções consecutivas (ordenadas por AEP)
    sorted_indices = np.argsort(aeps)
    distances = []
    for i in range(len(sorted_indices) - 1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
        daep = (aeps[idx2] - aeps[idx1]) / aep_range if aep_range > 0 else 0
        dcost = (costs[idx2] - costs[idx1]) / cost_range if cost_range > 0 else 0
        dist = np.sqrt(daep**2 + dcost**2)
        distances.append(dist)
    
    if len(distances) == 0:
        return 0.0
    
    # Spread = desvio padrão das distâncias
    mean_dist = np.mean(distances)
    spread = np.std(distances) if len(distances) > 1 else 0.0
    
    return spread

def calculate_c_metric(pareto_front_a, pareto_front_b):
    """
    Calcula C-metric: porcentagem de soluções de B dominadas por pelo menos uma solução de A.
    
    C(A, B) = |{b in B: existe a in A tal que a domina b}| / |B|
    
    Args:
        pareto_front_a: Lista de soluções do método A (cada uma com fitness.values = (aep, cost))
        pareto_front_b: Lista de soluções do método B
    
    Returns:
        c_metric: Valor entre 0 e 1 (0 = nenhuma solução dominada, 1 = todas dominadas)
    """
    if len(pareto_front_b) == 0:
        return 0.0
    if len(pareto_front_a) == 0:
        return 0.0
    
    # Extrai AEP e custo de A
    aeps_a = np.array([ind.fitness.values[0] for ind in pareto_front_a])
    costs_a = np.array([ind.fitness.values[1] for ind in pareto_front_a])
    
    # Conta quantas soluções de B são dominadas por pelo menos uma solução de A
    dominated_count = 0
    for ind_b in pareto_front_b:
        aep_b = ind_b.fitness.values[0]
        cost_b = ind_b.fitness.values[1]
        
        # Verifica se existe alguma solução em A que domina esta solução de B
        # A domina B se: AEP_A >= AEP_B AND Cost_A <= Cost_B, com pelo menos uma desigualdade estrita
        is_dominated = False
        for i in range(len(pareto_front_a)):
            aep_a = aeps_a[i]
            cost_a = costs_a[i]
            
            # Verifica dominância: A domina B se (AEP_A >= AEP_B AND Cost_A <= Cost_B) 
            # E pelo menos uma desigualdade é estrita
            if (aep_a >= aep_b and cost_a <= cost_b) and (aep_a > aep_b or cost_a < cost_b):
                is_dominated = True
                break
        
        if is_dominated:
            dominated_count += 1
    
    return dominated_count / len(pareto_front_b)

def calculate_hypervolume(pareto_front, ref_point=None):
    """
    Calcula o hipervolume da frente de Pareto.
    
    Args:
        pareto_front: Lista de soluções não-dominadas
        ref_point: Ponto de referência [Cost, -AEP] para minimização
    
    Returns:
        hv: Valor do hipervolume
    """
    if len(pareto_front) == 0:
        return 0.0
    
    if ref_point is None:
        # Ponto de referência padrão: pior que qualquer solução viável
        ref_point = [2e7, 0]  # [Max Cost, Min -AEP]
    
    # Transforma para minimização: [Cost, -AEP]
    # (Original: [AEP, Cost] -> FitnessMulti weights=(1.0, -1.0))
    pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] 
                 for ind in pareto_front 
                 if ind.fitness.valid and ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
    
    if len(pf_points) == 0:
        return 0.0
    
    pf_array = np.array(pf_points)
    hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
    
    return hv

def extract_solution_metrics(individual, method_name, is_sequential=False, turbine_layout=None):
    """
    Extrai todas as métricas de uma solução para salvar no CSV.
    
    Args:
        individual: Indivíduo Baseline/Proposed (35 genes) ou Sequential (3 genes)
        method_name: Nome do método ('Baseline', 'Proposed', 'Sequential')
        is_sequential: Se True, individual é da Fase 2 Sequential (3 genes)
        turbine_layout: Layout de turbinas (32 genes) para Sequential
    
    Returns:
        metrics: Dicionário com todas as métricas
    """
    if is_sequential:
        n_grupos_norm = individual[0]
        sub_pos = np.array([individual[1], individual[2]])
        turb_coords = np.array(turbine_layout).reshape((IND_SIZE, 2))
    else:
        n_coords = IND_SIZE * 2
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    
    n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
    n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
    n_grupos = min(n_grupos, IND_SIZE)
    
    coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
    
    # Calcula cabeamento
    plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
    
    # Calcula AEP
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                               TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1],
                               TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
    aep_liq = aep_bruto - res['perda_anual_mwh']
    custo = res['custo_total_usd']
    
    metrics = {
        'Method': method_name,
        'AEP_Liquido_MWh': aep_liq,
        'AEP_Bruto_MWh': aep_bruto,
        'Custo_Total_USD': custo,
        'Perdas_Joule_MWh': res['perda_anual_mwh'],
        'N_Grupos': len(plant.paths),
        'Comprimento_Total_km': res['comprimento_total_m'] / 1000.0,
        'Secao_Cabo_mm2': res['secao_cabo_mm2'],
        'Substation_X_m': sub_pos[0],
        'Substation_Y_m': sub_pos[1],
    }
    
    return metrics

def save_results_to_csv(results_dict, output_dir='.', filename='case_study_results.csv'):
    """
    Salva resultados em CSV para análise e plots.
    
    Args:
        results_dict: Dicionário com métricas de cada método
        output_dir: Diretório onde salvar o arquivo
        filename: Nome do arquivo CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame([results_dict])
    df.to_csv(filepath, index=False, float_format='%.2f')
    print(f"\n✓ Resultados salvos em: {filepath}")

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO REMOVIDAS
# =============================================================================
# As funções de plotagem foram movidas para case_study_plot.py
# Este arquivo apenas executa os testes e salva dados em CSV

# =============================================================================
# FUNÇÕES DE COLETA DE MÉTRICAS (INTEGRAÇÃO COM metrics_collector)
# =============================================================================
# As funções de coleta estão no módulo metrics_collector.py
# Aqui apenas mantemos funções auxiliares necessárias para a execução

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Cria o diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Diretório de saída: {os.path.abspath(OUTPUT_DIR)}")
    
    print("="*80)
    print("ESTUDO DE CASO COMPARATIVO")
    print("="*80)
    print(f"Configuração:")
    print(f"  Raio: {CIRCLE_RADIUS} m")
    print(f"  Turbinas: {IND_SIZE}")
    print(f"  Número de Execuções: {N_RUNS}")
    print(f"  População: {POP_SIZE}")
    print(f"  Gerações Baseline: {NGEN_BASELINE}")
    print(f"  Gerações Sequential Fase 1: {NGEN_SEQUENTIAL_P1}")
    print(f"  Gerações Sequential Fase 2: {NGEN_SEQUENTIAL_P2}")
    print(f"  Gerações Proposed Fase 1: {NGEN_PROPOSED_P1}")
    print(f"  Gerações Proposed Fase 2: {NGEN_PROPOSED_P2}")
    
    # Obtém número de processadores disponíveis
    num_cores = os.cpu_count()
    print(f"  Processadores disponíveis: {num_cores}")
    print(f"  Paralelização: HABILITADA (multiprocessing)")
    print("\n" + "="*80 + "\n")
    
    # Cria pool de processos
    pool = multiprocessing.Pool(processes=num_cores)
    
    try:
        # Lista para coletar todas as métricas de todas as execuções
        all_metrics_all_runs = []
        all_seeds = []
        
        # Listas para armazenar históricos de hipervolume de todas as execuções
        all_hv_history_baseline = []  # Lista de listas: cada elemento é o hv_history de uma execução
        all_hv_history_proposed = []  # Lista de listas: cada elemento é o hv_history de uma execução
        
        # Contadores para taxa de sucesso do Baseline (NSGA-II)
        baseline_success_count = 0  # Número de execuções que retornaram soluções válidas
        baseline_failure_count = 0  # Número de execuções que falharam (0 soluções)
        
        # Loop sobre múltiplas execuções
        for run_num in range(1, N_RUNS + 1):
            print("\n" + "="*80)
            print(f"EXECUÇÃO {run_num}/{N_RUNS}")
            print("="*80)
            
            # Gera seed aleatória para esta execução
            # Usa random.randint diretamente (random já foi importado no topo)
            # Usa run_num como parte da seed para garantir que cada execução tenha seed diferente
            SEED = random.randint(0, 2**31 - 1)
            all_seeds.append(SEED)
            
            print(f"Seed para esta execução: {SEED}")
            
            # Fixa seed para esta execução
            random.seed(SEED)
            np.random.seed(SEED)
            
            # CRIA POPULAÇÃO INICIAL GLOBAL (COMPARTILHADA POR TODOS OS MÉTODOS)
            # Isso garante comparação justa: todos começam com as mesmas coordenadas iniciais
            print(f"\n>>> Criando população inicial global (compartilhada)...")
            global_coords_pop = create_global_initial_population(SEED, POP_SIZE)
            print(f"   População inicial criada: {len(global_coords_pop)} indivíduos")
            print(f"   - 100% com coordenadas EXATAS do YAML (estratégia robusta)")
            
            # Dicionário para coletar métricas desta execução
            all_metrics = []
            
            # Executa Baseline
            print("\n>>> Executando Método Baseline (NSGA-II)...")
            start_time_baseline = time.time()
            pf_baseline, hv_history_baseline = run_baseline_method(
                SEED, global_coords_pop=global_coords_pop, pool=pool,
                run_id=run_num, output_dir=OUTPUT_DIR
            )
            time_baseline = time.time() - start_time_baseline
            print(f"Soluções encontradas: Baseline={len(pf_baseline)}")
            print(f"Tempo de execução: {time_baseline:.2f} segundos")
            
            # Armazena histórico de hipervolume
            all_hv_history_baseline.append(hv_history_baseline)
            # Nota: Métricas de convergência já foram coletadas durante a execução (a cada 20 gerações)
            
            # Executa Sequential
            print("\n>>> Executando Método Sequential...")
            start_time_sequential = time.time()
            best_sequential, best_turbine_layout = run_sequential_method(SEED, global_coords_pop=global_coords_pop, pool=pool)
            time_sequential = time.time() - start_time_sequential
            print(f"Sequential concluído.")
            print(f"Tempo de execução: {time_sequential:.2f} segundos")
            
            # Executa Proposed
            print("\n>>> Executando Método Proposed...")
            start_time_proposed = time.time()
            pf_proposed, hv_history_proposed = run_proposed_method(
                SEED, global_coords_pop=global_coords_pop, pool=pool,
                run_id=run_num, output_dir=OUTPUT_DIR
            )
            time_proposed = time.time() - start_time_proposed
            print(f"Soluções encontradas: Proposed={len(pf_proposed)}")
            print(f"Tempo de execução: {time_proposed:.2f} segundos")
            
            # Armazena histórico de hipervolume
            all_hv_history_proposed.append(hv_history_proposed)
        
            # Processa resultados Baseline
            if len(pf_baseline) > 0:
                # Conta como sucesso
                baseline_success_count += 1
                
                # Seleciona melhor solução usando knee point
                best_baseline = find_knee_point(pf_baseline)
                
                # Extrai métricas
                metrics_baseline = extract_solution_metrics(best_baseline, 'Baseline', is_sequential=False)
                metrics_baseline['Time_Total_seconds'] = time_baseline
                metrics_baseline['N_Solutions_Pareto'] = len(pf_baseline)
                metrics_baseline['Spread'] = calculate_spread(pf_baseline)
                metrics_baseline['Hypervolume'] = calculate_hypervolume(pf_baseline)
                metrics_baseline['AEP_Max_MWh'] = max(ind.fitness.values[0] for ind in pf_baseline)
                metrics_baseline['AEP_Min_MWh'] = min(ind.fitness.values[0] for ind in pf_baseline)
                metrics_baseline['Cost_Max_USD'] = max(ind.fitness.values[1] for ind in pf_baseline)
                metrics_baseline['Cost_Min_USD'] = min(ind.fitness.values[1] for ind in pf_baseline)
                metrics_baseline['N_Generations'] = NGEN_BASELINE
                all_metrics.append(metrics_baseline)
                
                # Coleta métricas para summary_results.csv usando metrics_collector
                summary_metrics_baseline = collect_summary_metrics(
                    best_baseline, 'Baseline', run_num, SEED, IND_SIZE,
                    is_sequential=False,
                    time_total_s=time_baseline, time_phase1_s=0.0, time_phase2_s=0.0,
                    pareto_front=pf_baseline, final_hypervolume=metrics_baseline['Hypervolume'],
                    pareto_spread=metrics_baseline['Spread'], num_solutions_pareto=len(pf_baseline),
                    feasibility_rate_final=100.0,  # TODO: calcular taxa de viabilidade real
                    calc_aep_func=calcAEP, cabling_func=cabling_v3.analisar_layout_completo,
                    wind_data=WIND_ROSE_DATA, turb_data=TURB_ATRBT_DATA
                )
                save_summary_results(summary_metrics_baseline, output_dir=OUTPUT_DIR, append=(run_num > 1))
                
                # Salva coordenadas da solução representativa (knee point)
                coords_metrics_baseline = collect_representative_solution_coords(
                    best_baseline, 'Baseline', run_num, SEED, IND_SIZE,
                    is_sequential=False,
                    net_aep_gwh=summary_metrics_baseline['Net_AEP_GWh'],
                    total_cost_usd=summary_metrics_baseline['Total_Cost_USD']
                )
                save_representative_solutions(coords_metrics_baseline, output_dir=OUTPUT_DIR, append=(run_num > 1))
                
                # Coleta métricas para all_pareto_fronts.csv
                pareto_metrics_baseline = collect_pareto_front_metrics(
                    pf_baseline, run_num, 'Baseline',
                    calc_aep_func=calcAEP, cabling_func=cabling_v3.analisar_layout_completo,
                    wind_data=WIND_ROSE_DATA, turb_data=TURB_ATRBT_DATA, turbine_count=IND_SIZE
                )
                save_pareto_fronts(pareto_metrics_baseline, output_dir=OUTPUT_DIR, append=(run_num > 1))
                
                print(f"\nMelhor solução Baseline:")
                print(f"  AEP Líquido: {best_baseline.fitness.values[0]/1000:.2f} GWh")
                print(f"  Custo: ${best_baseline.fitness.values[1]/1e6:.2f}M USD")
            else:
                # Conta como falha
                baseline_failure_count += 1
                print("ERRO: Baseline não retornou soluções válidas!")
            
            # Processa resultados Sequential
            if best_sequential is not None:
                # Extrai métricas
                metrics_sequential = extract_solution_metrics(best_sequential, 'Sequential', 
                                                             is_sequential=True, turbine_layout=best_turbine_layout)
                metrics_sequential['Time_Total_seconds'] = time_sequential
                metrics_sequential['Time_Phase1_seconds'] = 0  # Será calculado se necessário
                metrics_sequential['Time_Phase2_seconds'] = 0  # Será calculado se necessário
                metrics_sequential['N_Solutions_Pareto'] = 1  # Sequential retorna apenas 1 solução
                metrics_sequential['Spread'] = 0.0  # Sequential não tem frente de Pareto
                # Para Sequential, não calculamos hipervolume (não tem frente de Pareto)
                # Mas podemos criar um indivíduo Phase2 para compatibilidade
                # Calcula AEP líquido e custo para criar um indivíduo compatível
                n_grupos_norm = best_sequential[0]
                sub_pos = np.array([best_sequential[1], best_sequential[2]])
                turb_coords = np.array(best_turbine_layout).reshape((IND_SIZE, 2))
                n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
                n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
                n_grupos = min(n_grupos, IND_SIZE)
                coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
                plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
                wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
                turb_diam = TURB_ATRBT_DATA[4]
                aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                           TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1],
                                           TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
                aep_liq_seq = aep_bruto - res['perda_anual_mwh']
                custo_seq = res['custo_total_usd']
                # Cria indivíduo compatível para calcular HV (usa IndividualBaseline que tem FitnessMulti)
                full_genome = turb_coords.flatten().tolist() + [n_grupos_norm] + sub_pos.tolist()
                ind_seq_p2 = creator.IndividualBaseline(full_genome)
                ind_seq_p2.fitness.values = (aep_liq_seq, custo_seq)
                sequential_pf = [ind_seq_p2]
                metrics_sequential['Hypervolume'] = calculate_hypervolume(sequential_pf) if len(sequential_pf) > 0 else 0.0
                metrics_sequential['AEP_Max_MWh'] = metrics_sequential['AEP_Liquido_MWh']
                metrics_sequential['AEP_Min_MWh'] = metrics_sequential['AEP_Liquido_MWh']
                metrics_sequential['Cost_Max_USD'] = metrics_sequential['Custo_Total_USD']
                metrics_sequential['Cost_Min_USD'] = metrics_sequential['Custo_Total_USD']
                metrics_sequential['N_Generations'] = NGEN_SEQUENTIAL_P1 + NGEN_SEQUENTIAL_P2
                metrics_sequential['N_Generations_Phase1'] = NGEN_SEQUENTIAL_P1
                metrics_sequential['N_Generations_Phase2'] = NGEN_SEQUENTIAL_P2
                # AEP bruto da Fase 1 (layout fixo)
                turb_coords = np.array(best_turbine_layout).reshape((IND_SIZE, 2))
                wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
                turb_diam = TURB_ATRBT_DATA[4]
                aep_bruto_p1 = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                              TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1],
                                              TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
                metrics_sequential['AEP_Bruto_Phase1_MWh'] = aep_bruto_p1
                all_metrics.append(metrics_sequential)
                
                # Coleta métricas para summary_results.csv
                summary_metrics_sequential = collect_summary_metrics(
                    best_sequential, 'Sequential', run_num, SEED, IND_SIZE,
                    is_sequential=True, turbine_layout=best_turbine_layout,
                    time_total_s=time_sequential, time_phase1_s=0.0, time_phase2_s=0.0,
                    pareto_front=None, final_hypervolume=metrics_sequential['Hypervolume'],
                    pareto_spread=0.0, num_solutions_pareto=1,
                    feasibility_rate_final=100.0,
                    calc_aep_func=calcAEP, cabling_func=cabling_v3.analisar_layout_completo,
                    wind_data=WIND_ROSE_DATA, turb_data=TURB_ATRBT_DATA
                )
                save_summary_results(summary_metrics_sequential, output_dir=OUTPUT_DIR, append=True)
                
                # Salva coordenadas da solução representativa (Sequential)
                coords_metrics_sequential = collect_representative_solution_coords(
                    best_sequential, 'Sequential', run_num, SEED, IND_SIZE,
                    is_sequential=True, turbine_layout=best_turbine_layout,
                    net_aep_gwh=summary_metrics_sequential['Net_AEP_GWh'],
                    total_cost_usd=summary_metrics_sequential['Total_Cost_USD']
                )
                save_representative_solutions(coords_metrics_sequential, output_dir=OUTPUT_DIR, append=True)
                
                print(f"\nMelhor solução Sequential:")
                print(f"  AEP Líquido: {metrics_sequential['AEP_Liquido_MWh']/1000:.2f} GWh")
                print(f"  Custo: ${metrics_sequential['Custo_Total_USD']/1e6:.2f}M USD")
                print(f"  Número de Grupos: {metrics_sequential['N_Grupos']}")
            else:
                print("ERRO: Sequential não retornou solução válida!")
            
            # Processa resultados Proposed
            if len(pf_proposed) > 0:
                # Seleciona melhor solução usando knee point
                best_proposed = find_knee_point(pf_proposed)
                
                # Extrai métricas
                metrics_proposed = extract_solution_metrics(best_proposed, 'Proposed', is_sequential=False)
                metrics_proposed['Time_Total_seconds'] = time_proposed
                metrics_proposed['Time_Phase1_seconds'] = 0  # Será calculado se necessário
                metrics_proposed['Time_Phase2_seconds'] = 0  # Será calculado se necessário
                metrics_proposed['N_Solutions_Pareto'] = len(pf_proposed)
                metrics_proposed['Spread'] = calculate_spread(pf_proposed)
                metrics_proposed['Hypervolume'] = calculate_hypervolume(pf_proposed)
                metrics_proposed['AEP_Max_MWh'] = max(ind.fitness.values[0] for ind in pf_proposed)
                metrics_proposed['AEP_Min_MWh'] = min(ind.fitness.values[0] for ind in pf_proposed)
                metrics_proposed['Cost_Max_USD'] = max(ind.fitness.values[1] for ind in pf_proposed)
                metrics_proposed['Cost_Min_USD'] = min(ind.fitness.values[1] for ind in pf_proposed)
                metrics_proposed['N_Generations'] = NGEN_PROPOSED_P1 + NGEN_PROPOSED_P2
                metrics_proposed['N_Generations_Phase1'] = NGEN_PROPOSED_P1
                metrics_proposed['N_Generations_Phase2'] = NGEN_PROPOSED_P2
                metrics_proposed['N_Top_Layouts_Phase1'] = N_TOP_LAYOUTS
                all_metrics.append(metrics_proposed)
                
                # Coleta métricas para summary_results.csv
                summary_metrics_proposed = collect_summary_metrics(
                    best_proposed, 'Proposed', run_num, SEED, IND_SIZE,
                    is_sequential=False,
                    time_total_s=time_proposed, time_phase1_s=0.0, time_phase2_s=0.0,
                    pareto_front=pf_proposed, final_hypervolume=metrics_proposed['Hypervolume'],
                    pareto_spread=metrics_proposed['Spread'], num_solutions_pareto=len(pf_proposed),
                    feasibility_rate_final=100.0,
                    calc_aep_func=calcAEP, cabling_func=cabling_v3.analisar_layout_completo,
                    wind_data=WIND_ROSE_DATA, turb_data=TURB_ATRBT_DATA
                )
                save_summary_results(summary_metrics_proposed, output_dir=OUTPUT_DIR, append=True)
                
                # Salva coordenadas da solução representativa (knee point)
                coords_metrics_proposed = collect_representative_solution_coords(
                    best_proposed, 'Proposed', run_num, SEED, IND_SIZE,
                    is_sequential=False,
                    net_aep_gwh=summary_metrics_proposed['Net_AEP_GWh'],
                    total_cost_usd=summary_metrics_proposed['Total_Cost_USD']
                )
                save_representative_solutions(coords_metrics_proposed, output_dir=OUTPUT_DIR, append=True)
                
                # Coleta métricas para all_pareto_fronts.csv
                pareto_metrics_proposed = collect_pareto_front_metrics(
                    pf_proposed, run_num, 'Proposed',
                    calc_aep_func=calcAEP, cabling_func=cabling_v3.analisar_layout_completo,
                    wind_data=WIND_ROSE_DATA, turb_data=TURB_ATRBT_DATA, turbine_count=IND_SIZE
                )
                save_pareto_fronts(pareto_metrics_proposed, output_dir=OUTPUT_DIR, append=True)
                
                print(f"\nMelhor solução Proposed:")
                print(f"  AEP Líquido: {best_proposed.fitness.values[0]/1000:.2f} GWh")
                print(f"  Custo: ${best_proposed.fitness.values[1]/1e6:.2f}M USD")
            else:
                print("ERRO: Proposed não retornou soluções válidas!")
            
            # Adiciona seed e número da execução às métricas desta execução
            for metrics in all_metrics:
                metrics['Seed'] = SEED
                metrics['Run'] = run_num
                all_metrics_all_runs.append(metrics)
            
            # Métricas já foram salvas nos CSVs durante o processamento acima
            print(f"\n✓ Métricas da execução {run_num} salvas nos CSVs")
            
            # Plots são salvos apenas na última execução (ou podem ser sobrescritos a cada execução)
            # Se quiser salvar plots de todas as execuções, adicione sufixo: f'baseline_solution_run{run_num}.png'
            
            # Preserva variáveis da última execução para plot de frentes de Pareto
            if run_num == N_RUNS:
                pf_baseline_last = pf_baseline
                pf_proposed_last = pf_proposed
                best_sequential_last = best_sequential
                best_turbine_layout_last = best_turbine_layout
        
        # Salva todas as seeds usadas (após todas as execuções)
        seed_file = os.path.join(OUTPUT_DIR, 'seed_used.txt')
        with open(seed_file, 'w') as f:
            for i, seed in enumerate(all_seeds, 1):
                f.write(f"Run {i}: {seed}\n")
        
        print("\n" + "="*80)
        print(f"TODAS AS {N_RUNS} EXECUÇÕES CONCLUÍDAS")
        print("="*80)
        print(f"✓ Todas as seeds salvas em: {os.path.join(OUTPUT_DIR, 'seed_used.txt')}")
        
        # =============================================================================
        # ESTATÍSTICAS DE TAXA DE SUCESSO DO BASELINE (NSGA-II)
        # =============================================================================
        total_baseline_runs = baseline_success_count + baseline_failure_count
        if total_baseline_runs > 0:
            baseline_success_rate = (baseline_success_count / total_baseline_runs) * 100
            print("\n" + "="*80)
            print("TAXA DE SUCESSO DO BASELINE (NSGA-II)")
            print("="*80)
            print(f"Total de execuções: {total_baseline_runs}")
            print(f"Sucessos (soluções válidas encontradas): {baseline_success_count}")
            print(f"Falhas (nenhuma solução válida): {baseline_failure_count}")
            print(f"Taxa de sucesso: {baseline_success_rate:.1f}%")
            print(f"Taxa de falha: {(100 - baseline_success_rate):.1f}%")
            print("="*80)
            
            # Salva estatísticas em arquivo de texto
            stats_file = os.path.join(OUTPUT_DIR, 'baseline_success_rate.txt')
            with open(stats_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TAXA DE SUCESSO DO BASELINE (NSGA-II)\n")
                f.write("="*80 + "\n")
                f.write(f"Total de execuções: {total_baseline_runs}\n")
                f.write(f"Sucessos (soluções válidas encontradas): {baseline_success_count}\n")
                f.write(f"Falhas (nenhuma solução válida): {baseline_failure_count}\n")
                f.write(f"Taxa de sucesso: {baseline_success_rate:.1f}%\n")
                f.write(f"Taxa de falha: {(100 - baseline_success_rate):.1f}%\n")
                f.write("="*80 + "\n")
            print(f"✓ Estatísticas de taxa de sucesso salvas em: {stats_file}")
        
        # Resumo final de todas as execuções
        if len(all_metrics_all_runs) > 0:
            df_all = pd.DataFrame(all_metrics_all_runs)
            print(f"\nResumo de todas as {N_RUNS} execuções:")
            print(df_all[['Run', 'Method', 'AEP_Liquido_MWh', 'Custo_Total_USD', 'Time_Total_seconds', 
                          'N_Solutions_Pareto', 'Spread', 'Hypervolume']].to_string(index=False))
        
        print("\n" + "="*80)
        print("ESTUDO DE CASO CONCLUÍDO")
        print("="*80)
        print("✓ Todos os dados foram salvos nos arquivos CSV:")
        print(f"  - {os.path.join(OUTPUT_DIR, 'summary_results.csv')}")
        print(f"  - {os.path.join(OUTPUT_DIR, 'all_pareto_fronts.csv')}")
        print(f"  - {os.path.join(OUTPUT_DIR, 'convergence_history.csv')}")
        print("\nPara gerar os gráficos, execute: python multi_objetivo/case_study_plot.py")
        print("="*80)
    
    finally:
        # Fecha o pool de processos
        pool.close()
        pool.join()
        print("✓ Pool de processos encerrado corretamente.")

