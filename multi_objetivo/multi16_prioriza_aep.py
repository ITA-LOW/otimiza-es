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

# Limpa tipos anteriores se existirem (para evitar conflitos ao reexecutar)
# O DEAP mantém tipos criados anteriormente, então precisamos limpar antes de recriar
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "IndividualPhase1"):
    del creator.IndividualPhase1
if hasattr(creator, "IndividualPhase2"):
    del creator.IndividualPhase2

# Fase 1: Single-objective (apenas AEP bruto)
# FitnessMax: maximiza um único objetivo (AEP bruto)
# IndividualPhase1: indivíduo da Fase 1 = lista de coordenadas [x1, y1, x2, y2, ...]
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

# Fase 2: Multi-objective (AEP líquido + Custo)
# FitnessMulti: maximiza AEP (peso +1.0) e minimiza Custo (peso -1.0)
# IndividualPhase2: indivíduo da Fase 2 = [coords turbinas] + [n_grupos] + [coords subestação]
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)

# Cria toolboxes separadas para cada fase
# Cada toolbox registra operadores genéticos específicos (crossover, mutação, seleção)
toolbox_phase1 = base.Toolbox()
toolbox_phase2 = base.Toolbox()

# =============================================================================
# PARÂMETROS DO PARQUE EÓLICO
# =============================================================================
IND_SIZE = 16  # Número de turbinas eólicas no parque
CIRCLE_RADIUS = 5000  # Raio do círculo de restrição (metros) - todas as turbinas devem estar dentro
N_DIAMETERS = 260  # Distância mínima entre turbinas em diâmetros de rotor (restrição de wake)
MIN_SUB_TURB_DIST = 50.0  # Distância mínima entre subestação e turbinas (metros)

# Limites para número de grupos de cabeamento (será otimizado pelo AG na Fase 2)
# O número de grupos é codificado como gene normalizado [0,1] e mapeado para [MIN_GRUPOS, MAX_GRUPOS]
MIN_GRUPOS = 2   # Mínimo: 2 grupos (todas as turbinas em 2 strings)
MAX_GRUPOS = 64  # Máximo: 64 grupos (uma turbina por grupo - limite superior flexível)
N_GRUPOS_INICIAL = MIN_GRUPOS  # Valor inicial do gene de agrupamento (normalizado)

# =============================================================================
# PARÂMETROS DO ALGORITMO GENÉTICO - FASE 1
# =============================================================================
# Fase 1: Otimização rápida de layout (apenas AEP bruto, sem cálculo de cabeamento)
# Objetivo: Explorar intensamente o espaço de layouts para encontrar configurações
# com alto AEP bruto, que servirão como ponto de partida para a Fase 2

POP_SIZE_P1 = 300  # Tamanho da população (número de indivíduos por geração)
NGEN_P1 = 1500  # Número máximo de gerações
CXPB_P1 = 0.95  # Probabilidade de crossover (95% dos pares fazem crossover)
MUTPB_P1 = 0.7  # Probabilidade de mutação (70% dos indivíduos são mutados)
INDPB_P1 = 0.4  # Probabilidade de mutar cada gene individualmente (40% dos genes)

# Parâmetros de mutação Fase 1 (mutação gaussiana)
MU_P1 = 0  # Média da distribuição gaussiana (centro em zero = mutação simétrica)
SIGMA_P1 = 100  # Desvio padrão da distribuição gaussiana (metros) - controla intensidade da mutação

# Parâmetros de crossover Fase 1 (Blend Crossover)
# Blend Crossover: combina dois pais usando combinação linear controlada por alpha
# alpha=0.5: filhos ficam entre os pais (exploração moderada)
CROSSOVER_ALPHA_P1 = 0.5  # Parâmetro alpha do crossover blend

# Parâmetros de seleção Fase 1 (Seleção por Torneio)
TOURNSIZE_P1 = 5  # Tamanho do torneio (maior = mais pressão seletiva, favorece melhores)

# Parâmetros de estagnação e parada precoce Fase 1
# Sistema adaptativo: detecta quando a otimização para de melhorar e aumenta
# a intensidade da mutação para escapar de ótimos locais
PATIENCE_P1 = 150  # Número de gerações sem melhoria antes de ativar mutação agressiva
MIN_DELTA_P1 = 10.0  # Melhoria mínima (MWh) para resetar contador de estagnação
SIGMA_NORMAL_P1 = 100  # Sigma normal para mutação (exploração local)
SIGMA_AGGRESSIVE_P1 = 250  # Sigma agressivo quando detecta estagnação (exploração global)
AGGRESSIVE_DURATION_P1 = 15  # Duração (gerações) da fase de mutação agressiva

# Parâmetros do Hall of Fame Fase 1
# Hall of Fame: mantém os melhores indivíduos encontrados durante toda a otimização
HOF_SIZE_P1 = 50  # Número de melhores indivíduos mantidos no Hall of Fame
N_TOP_LAYOUTS = 30  # Número de melhores layouts da Fase 1 usados como sementes na Fase 2

# =============================================================================
# PARÂMETROS DO ALGORITMO GENÉTICO - FASE 2
# =============================================================================
# Fase 2: Otimização multiobjetivo (AEP líquido + Custo de cabeamento)
# Objetivo: Refinar os melhores layouts da Fase 1 considerando cabeamento completo
# e encontrar a frente de Pareto otimizando simultaneamente AEP e Custo

POP_SIZE_P2 = 300  # Tamanho da população (mantém mesmo tamanho da Fase 1)
NGEN_P2 = 1500  # Número máximo de gerações
CXPB_P2 = CXPB_P1  # Probabilidade de crossover (usa mesmo valor da Fase 1)
MUTPB_P2 = MUTPB_P1  # Probabilidade de mutação (usa mesmo valor da Fase 1)
INDPB_P2 = INDPB_P1  # Probabilidade de mutar cada gene individualmente

# Parâmetros de mutação Fase 2 (mutação gaussiana diferenciada por componente)
MU_P2 = 0  # Média da distribuição gaussiana (centro em zero)
SIGMA_P2 = 100  # Desvio padrão para mutação das coordenadas das turbinas (metros)
SIGMA_SUB_MULTIPLIER = 5  # Multiplicador do sigma para mutação da subestação (maior exploração)
SIGMA_SUB_MIN = 200.0  # Sigma mínimo para mutação da subestação (metros) - garante exploração mínima

# Probabilidades de mutação específicas Fase 2
# A Fase 2 tem componentes adicionais (número de grupos e posição da subestação)
# que precisam de estratégias de mutação diferentes
PROB_MUTATE_GROUPS_P2 = 0.3  # Probabilidade de mutar número de grupos (30% - mutação menos frequente)
PROB_MUTATE_SUBSTATION_P2 = 0.9  # Probabilidade de mutar posição da subestação (90% - mutação frequente)
PROB_AGGRESSIVE_SUB_MUTATION = 0.25  # Probabilidade de mutação agressiva da subestação (25% - exploração ampla)
PROB_EXTREME_SUB_MUTATION = 0.1  # Probabilidade de mutação extrema da subestação (10% - exploração global)

# Fatores de mutação agressiva/extrema da subestação
# Mutação agressiva/extrema: permite saltos grandes no espaço de busca da subestação
# para escapar de ótimos locais e explorar diferentes regiões do parque
AGGRESSIVE_SUB_RADIUS_FACTOR = 0.7  # Fator do raio para mutação agressiva (70% do CIRCLE_RADIUS)
EXTREME_SUB_RADIUS_FACTOR = 1.0  # Fator do raio para mutação extrema (100% do CIRCLE_RADIUS)

# Parâmetros de crossover Fase 2 (Blend Crossover)
# TODAS as variáveis (turbinas, grupos, subestação) usam Blend Crossover para consistência
CROSSOVER_ALPHA_P2 = 0.5  # Parâmetro alpha do crossover blend (mesmo valor para todas as variáveis)

# Parâmetros de estagnação e parada precoce Fase 2
# Sistema adaptativo para multiobjetivo: detecta estagnação em ambos os objetivos
PATIENCE_P2 = 100  # Número de gerações sem melhoria antes de parar
MIN_DELTA_AEP_P2 = 10.0  # Melhoria mínima em AEP (MWh) para resetar contador de estagnação
MIN_DELTA_COST_P2 = 100.0  # Melhoria mínima em custo (USD) para resetar contador de estagnação

# Parâmetros de inicialização da população Fase 2
# A população inicial da Fase 2 é criada a partir dos melhores layouts da Fase 1
# com perturbações para manter diversidade
PERTURBATION_SIGMA_MIN = 150  # Sigma mínimo para perturbação de layouts da Fase 1 (metros)
PERTURBATION_SIGMA_MAX = 300  # Sigma máximo para perturbação de layouts da Fase 1 (metros)

# Parâmetros de detecção de sobreposição de cabos
# Sistema de penalidades para garantir soluções fisicamente viáveis:
# - Cruzamentos de cabos são inaceitáveis (penalidade extrema)
# - Múltiplas conexões na mesma turbina são inaceitáveis (penalidade extrema)
# - Cabos muito próximos na subestação são penalizados (problema de segurança)
MIN_CABLE_DISTANCE = 100.0  # Distância mínima permitida entre segmentos de cabos (metros)
MIN_ANGLE_SUBSTATION = 15  # Ângulo mínimo (graus) entre cabos chegando na subestação (menor = muito próximo)
PENALTY_CROSSING = 1e3  # Penalidade por cruzamento de cabos (alta, mas permite recuperação)
PENALTY_MULTIPLE_CONNECTIONS = 1e9  # Penalidade por múltiplas conexões na mesma turbina (extrema - elimina solução)
PENALTY_SMALL_ANGLE_SUBSTATION = 1e7  # Penalidade por ângulo muito fechado na subestação (muito alta - quase elimina)

# =============================================================================
# PRÉ-CARREGAMENTO DE DADOS
# =============================================================================
# Carrega dados de configuração do parque eólico uma única vez no início
# para evitar recarregamento repetido durante a otimização (otimização de performance)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = "config"
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
# Carrega coordenadas iniciais das turbinas (usadas como semente para população inicial)
initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)

# Carrega dados de vento (wind rose) e características das turbinas
full_path_wr = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
full_path_turb = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)  # [turb_ci, turb_co, rated_ws, rated_pwr, turb_diam]
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)  # [wind_dir, wind_freq, wind_speed]

# =============================================================================
# FUNÇÕES DE INICIALIZAÇÃO, RESTRIÇÃO E MUTAÇÃO - FASE 1
# COPIADO EXATAMENTE DE wind_farm_GA_16.py
# =============================================================================

def create_individual_from_coordinates(coords):
    """
    Cria indivíduo da Fase 1 a partir de coordenadas.
    Converte coordenadas 2D (IND_SIZE, 2) para lista plana [x1, y1, x2, y2, ...]
    """
    individual = creator.IndividualPhase1(np.array(coords).flatten().tolist())
    return individual

# Registra funções de criação de indivíduos e população na toolbox da Fase 1
toolbox_phase1.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox_phase1.register("population", tools.initRepeat, list, toolbox_phase1.individual)

def is_within_circle(x, y, radius):
    """
    Verifica se coordenadas estão dentro do círculo de restrição.
    Usa equação do círculo: x² + y² ≤ r²
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return x**2 + y**2 <= radius**2

def enforce_circle(individual):
    """
    Aplica restrição de círculo: se uma turbina está fora do círculo,
    projeta ela de volta para a borda do círculo mantendo o ângulo.
    Isso garante que todas as turbinas fiquem dentro da área permitida.
    """
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            # Calcula ângulo e projeta para a borda do círculo
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

def enforce_substation_circle(substation_pos):
    """Garante que a subestação fique dentro do círculo de restrição."""
    x, y = substation_pos[0], substation_pos[1]
    dist = np.sqrt(x**2 + y**2)
    if dist > CIRCLE_RADIUS:
        angle = np.arctan2(y, x)
        substation_pos[0] = CIRCLE_RADIUS * np.cos(angle)
        substation_pos[1] = CIRCLE_RADIUS * np.sin(angle)
    return substation_pos

def mutate_phase1(individual, mu, sigma, indpb):
    """Mutação - EXATA de wind_farm_GA_16.py (cria novo indivíduo)"""
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.IndividualPhase1(individual.tolist()),

def create_individual_phase2_from_coords(coords, substation_pos=None):
    """
    Cria um indivíduo da Fase 2 a partir de coordenadas da Fase 1.
    
    Estrutura do genoma da Fase 2:
    - [32 coords turbinas]: coordenadas (x, y) de cada turbina (IND_SIZE * 2 = 32)
    - [1 n_grupos]: número de grupos normalizado [0, 1] (será mapeado para [MIN_GRUPOS, MAX_GRUPOS])
    - [2 coords subestação]: coordenadas (x, y) da subestação offshore
    Total: 35 variáveis
    
    Args:
        coords: Coordenadas das turbinas (array 2D ou lista plana)
        substation_pos: Posição inicial da subestação (se None, gera aleatória)
    """
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
    
    # Adiciona posição inicial da subestação
    # Se não fornecida, usa posição aleatória balanceada (não enviesada)
    if substation_pos is None:
        # Posição inicial: aleatória em qualquer direção para exploração balanceada
        # Não enviesar para quadrante inferior permite melhor exploração do espaço
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(CIRCLE_RADIUS * 0.2, CIRCLE_RADIUS * 0.7)
        substation_pos = [
            radius * np.cos(angle),
            radius * np.sin(angle)
        ]
    
    # Garante que a subestação inicial também fique dentro do círculo
    substation_pos = enforce_substation_circle(np.array(substation_pos)).tolist()
    
    return creator.IndividualPhase2(coords_flat + [n_grupos_normalizado] + list(substation_pos))

# Não registramos individual diretamente pois precisamos passar coords dinamicamente
# Usaremos create_individual_phase2_from_coords diretamente quando necessário

def mutate_phase2(individual, mu, sigma, indpb):
    """
    Mutação para Fase 2: aplica mutação gaussiana diferenciada para cada componente.
    
    Estrutura do indivíduo: [32 coords turbinas] + [1 n_grupos] + [2 coords subestação] = 35 variáveis
    
    Estratégia de mutação:
    1. Coordenadas das turbinas: mutação gaussiana padrão (sigma)
    2. Número de grupos: mutação gaussiana com sigma menor (0.1) para mudanças suaves
    3. Posição da subestação: mutação gaussiana com sigma maior (SIGMA_SUB_MULTIPLIER * sigma)
       + mutações agressivas/extremas ocasionais para exploração global
    
    Args:
        individual: Indivíduo da Fase 2 a ser mutado
        mu: Média da distribuição gaussiana (geralmente 0)
        sigma: Desvio padrão base para mutação das turbinas
        indpb: Probabilidade de mutar cada gene individualmente
    """
    individual_arr = np.array(individual)
    n_coords = IND_SIZE * 2
    
    if random.random() < indpb:
        # Muta coordenadas das turbinas
        for i in range(n_coords):
            individual_arr[i] += random.gauss(mu, sigma)
        
        # Muta número de grupos (índice n_coords)
        if random.random() < PROB_MUTATE_GROUPS_P2:
            individual_arr[n_coords] += random.gauss(0, 0.1)
            individual_arr[n_coords] = max(0.0, min(1.0, individual_arr[n_coords]))
        
        # Muta posição da subestação (índices n_coords+1 e n_coords+2)
        # MUTAÇÃO INDEPENDENTE: sempre tenta mutar a subestação, independente da mutação das turbinas
        # Isso garante exploração contínua do espaço de busca da subestação
        if random.random() < PROB_MUTATE_SUBSTATION_P2:
            # Mutação mais ampla: usa sigma maior e permite exploração em área maior
            mutation_sigma_sub = max(sigma * SIGMA_SUB_MULTIPLIER, SIGMA_SUB_MIN)
            individual_arr[n_coords + 1] += random.gauss(0, mutation_sigma_sub)
            individual_arr[n_coords + 2] += random.gauss(0, mutation_sigma_sub)
            
            # Ocasionalmente, faz mutação muito agressiva
            if random.random() < PROB_AGGRESSIVE_SUB_MUTATION:
                # Mutação agressiva: explora até AGGRESSIVE_SUB_RADIUS_FACTOR do raio do círculo
                individual_arr[n_coords + 1] += random.gauss(0, CIRCLE_RADIUS * AGGRESSIVE_SUB_RADIUS_FACTOR)
                individual_arr[n_coords + 2] += random.gauss(0, CIRCLE_RADIUS * AGGRESSIVE_SUB_RADIUS_FACTOR)
            
            # Raramente, faz mutação extremamente agressiva para exploração global
            if random.random() < PROB_EXTREME_SUB_MUTATION:
                # Mutação extrema: explora até EXTREME_SUB_RADIUS_FACTOR do raio do círculo
                individual_arr[n_coords + 1] += random.gauss(0, CIRCLE_RADIUS * EXTREME_SUB_RADIUS_FACTOR)
                individual_arr[n_coords + 2] += random.gauss(0, CIRCLE_RADIUS * EXTREME_SUB_RADIUS_FACTOR)
        
        mutated_list = individual_arr.tolist()
        enforce_circle(mutated_list[:n_coords])
        
        # Garante que a subestação também fique dentro do círculo
        substation_pos_mutated = [mutated_list[n_coords + 1], mutated_list[n_coords + 2]]
        substation_pos_mutated = enforce_substation_circle(substation_pos_mutated)
        mutated_list[n_coords + 1] = substation_pos_mutated[0]
        mutated_list[n_coords + 2] = substation_pos_mutated[1]
        
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
    Avaliação da Fase 1: calcula apenas AEP bruto (sem considerar cabeamento).
    
    Esta função é muito rápida porque não calcula cabeamento, permitindo muitas
    avaliações e exploração intensa do espaço de layouts.
    
    Penalidades aplicadas:
    - Turbinas fora do círculo: penalidade extrema (1e6 por turbina)
    - Turbinas muito próximas (< N_DIAMETERS): penalidade extrema (1e6 por violação)
    
    Returns:
        fitness: AEP bruto total (MWh) - penalidades
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

def detectar_sobreposicao_cabos(paths, coords, min_distance=50.0, substation_idx=None):
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
    
    # Penalidade EXTREMA para cruzamentos: elimina soluções com cruzamentos
    # Cada cruzamento é inaceitável e deve ser fortemente punido
    penalty_cruzamentos = n_cruzamentos * PENALTY_CROSSING
    
    # Penalidade EXTREMA para múltiplas conexões: elimina soluções com turbinas conectadas múltiplas vezes
    # Cada conexão múltipla é inaceitável e deve ser fortemente punida
    penalty_multiplas = n_multiplas_conexoes * PENALTY_MULTIPLE_CONNECTIONS
    
    # Penalidade para ângulos muito fechados na subestação
    # Cabos chegando com ângulo < 30° estão muito próximos e devem ser penalizados
    penalty_angulo_fechado = n_angulos_fechados_sub * PENALTY_SMALL_ANGLE_SUBSTATION
    
    # Penalidade total: extremamente alta para garantir eliminação
    penalty_total = penalty_cruzamentos + penalty_proximidade + penalty_multiplas + penalty_angulo_fechado
    
    return penalty_total

def evaluate_phase2(individual):
    """
    Avaliação da Fase 2: calcula AEP líquido e Custo total (com cabeamento completo).
    
    Esta função é mais lenta que evaluate_phase1 porque calcula cabeamento completo,
    mas permite otimização multiobjetivo considerando trade-offs reais.
    
    Estrutura do indivíduo: [32 coords turbinas] + [1 n_grupos] + [2 coords subestação] = 35 variáveis
    
    Processo de avaliação:
    1. Extrai coordenadas, número de grupos e posição da subestação do genoma
    2. Calcula AEP bruto (mesmo método da Fase 1)
    3. Calcula cabeamento completo usando cabling_v3 (inclui perdas Joule)
    4. Detecta violações (cruzamentos, múltiplas conexões, etc.)
    5. Calcula AEP líquido = AEP bruto - perdas Joule - penalidades
    6. Calcula Custo total = custo de cabeamento + penalidades
    
    Penalidades aplicadas:
    - Turbinas fora do círculo
    - Turbinas muito próximas
    - Subestação fora do círculo
    - Subestação muito próxima das turbinas
    - Cruzamentos de cabos
    - Múltiplas conexões na mesma turbina
    - Ângulos muito fechados na subestação
    
    Returns:
        (aep_liquido, custo_penalizado): Tupla com AEP líquido (MWh) e Custo total (USD)
    """
    try:
        n_coords = IND_SIZE * 2
        # Extrai coordenadas, número de grupos e posição da subestação do indivíduo
        coords_flat = individual[:n_coords]
        n_grupos_normalizado = individual[n_coords]  # Índice n_coords (32)
        substation_pos = np.array([individual[n_coords + 1], individual[n_coords + 2]])  # Índices 33 e 34
        
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
        
        # Penalidade: subestação fora do círculo
        dist_sub_from_center = np.linalg.norm(substation_pos)
        penalty_sub_out_of_circle = np.maximum(0, dist_sub_from_center - CIRCLE_RADIUS) * 1e6
        
        # Penalidade: distância mínima entre subestação e turbinas (50m)
        dist_sub_to_turbines = np.linalg.norm(turb_coords - substation_pos, axis=1)
        min_dist_sub_turb = np.min(dist_sub_to_turbines)
        penalty_sub_too_close = np.maximum(0, MIN_SUB_TURB_DIST - min_dist_sub_turb) * 1e6
        
        # AEP Bruto
        _, _, _, _, turb_diam = TURB_ATRBT_DATA
        aep_bruto = np.sum(calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], 
                                   WIND_ROSE_DATA[0], turb_diam, *TURB_ATRBT_DATA[0:2], 
                                   *TURB_ATRBT_DATA[2:4]))

        # Cabeamento usando posição da subestação do genoma
        # IMPORTANTE: Adiciona a subestação às coordenadas para que o cálculo de cabeamento
        # use a posição real otimizada pelo GA, não apenas a turbina mais próxima
        coords_with_substation = np.vstack([turb_coords, substation_pos.reshape(1, 2)])
        substation_idx = IND_SIZE  # Índice da subestação após todas as turbinas
        
        try:
            planta, resultados = cabling_v3.analisar_layout_completo(
                coords_with_substation, sub=substation_idx, n_grupos=n_grupos)
            
            custo_total = resultados['custo_total_usd']
            perdas_joule_mwh = resultados['perda_anual_mwh']
            
            # Detecta cruzamentos, proximidade excessiva e múltiplas conexões
            # Passa índice da subestação para permitir múltiplas conexões nela (mas não em turbinas)
            penalty_overlap = detectar_sobreposicao_cabos(
                planta.paths, coords_with_substation, min_distance=MIN_CABLE_DISTANCE, substation_idx=substation_idx)
            
            # Penalidade EXTREMA: elimina soluções com cruzamentos ou múltiplas conexões
            # Cruzamentos e múltiplas conexões são inaceitáveis e devem ser eliminados
            penalty_total_cabos = penalty_overlap
            
            aep_liquido = aep_bruto - perdas_joule_mwh - penalty_out_of_circle - penalty_close_turbines - penalty_total_cabos - penalty_sub_too_close - penalty_sub_out_of_circle
            custo_penalizado = custo_total + penalty_out_of_circle + penalty_close_turbines + penalty_total_cabos + penalty_sub_too_close + penalty_sub_out_of_circle
            
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

# Toolbox Fase 1 (Single-objective)
toolbox_phase1.register("mate", tools.cxBlend, alpha=CROSSOVER_ALPHA_P1)
toolbox_phase1.register("mutate", mutate_phase1, mu=MU_P1, sigma=SIGMA_P1, indpb=INDPB_P1)
toolbox_phase1.register("select", tools.selTournament, tournsize=TOURNSIZE_P1)
toolbox_phase1.register("evaluate", evaluate_phase1)

# Toolbox Fase 2 (Multi-objective)
def mate_phase2(ind1, ind2):
    """
    Crossover Blend para Fase 2: aplica Blend Crossover consistentemente para todas as variáveis.
    
    Estrutura do indivíduo: [32 coords turbinas] + [1 n_grupos] + [2 coords subestação] = 35 variáveis
    
    Estratégia de crossover:
    1. Coordenadas das turbinas: Blend Crossover usando cxBlend do DEAP
    2. Número de grupos: Blend Crossover manual (mesma fórmula, aplicada a um único gene)
    3. Posição da subestação: Blend Crossover usando cxBlend do DEAP
    
    Blend Crossover: combina dois pais usando combinação linear controlada por alpha
    - alpha=0.5: filhos ficam entre os pais (exploração moderada)
    - gamma = (1 + 2*alpha) * random() - alpha: permite filhos além dos pais (exploração ampla)
    
    Args:
        ind1, ind2: Dois indivíduos da Fase 2 a serem cruzados
    
    Returns:
        ind1, ind2: Indivíduos modificados in-place (crossover em DEAP modifica in-place)
    """
    n_coords = IND_SIZE * 2
    
    # Crossover blend para coordenadas das turbinas
    tools.cxBlend(ind1[:n_coords], ind2[:n_coords], alpha=CROSSOVER_ALPHA_P2)
    
    # Crossover blend para número de grupos (índice n_coords)
    # Aplica blend manualmente para o gene único (mesma fórmula do cxBlend)
    gamma = (1. + 2. * CROSSOVER_ALPHA_P2) * random.random() - CROSSOVER_ALPHA_P2
    temp = (1. - gamma) * ind1[n_coords] + gamma * ind2[n_coords]
    ind2[n_coords] = gamma * ind1[n_coords] + (1. - gamma) * ind2[n_coords]
    ind1[n_coords] = temp
    
    # Garante limites para número de grupos
    ind1[n_coords] = max(0.0, min(1.0, ind1[n_coords]))
    ind2[n_coords] = max(0.0, min(1.0, ind2[n_coords]))
    
    # Crossover blend para posição da subestação (índices n_coords+1 e n_coords+2)
    tools.cxBlend(ind1[n_coords+1:n_coords+3], ind2[n_coords+1:n_coords+3], alpha=CROSSOVER_ALPHA_P2)
    
    return ind1, ind2

toolbox_phase2.register("mate", mate_phase2)
toolbox_phase2.register("mutate", mutate_phase2, mu=MU_P2, sigma=SIGMA_P2, indpb=INDPB_P2)
toolbox_phase2.register("select", tools.selNSGA2)
toolbox_phase2.register("evaluate", evaluate_phase2)

# =============================================================================
# FUNÇÃO PRINCIPAL - ESTRATÉGIA HÍBRIDA EM DUAS FASES
# =============================================================================

def optimize_phase1(POP_SIZE, NGEN, CXPB, MUTPB):
    """
    Fase 1: Otimização de Layout (AEP Bruto apenas).
    
    Esta fase é muito rápida porque não calcula cabeamento, permitindo:
    - Muitas avaliações (10-50x mais rápido que Fase 2)
    - Exploração intensa do espaço de layouts
    - Encontrar configurações com alto AEP bruto
    
    Algoritmo: Algoritmo Genético single-objective com:
    - Seleção por Torneio
    - Blend Crossover
    - Mutação Gaussiana
    - Sistema adaptativo de mutação agressiva (quando detecta estagnação)
    - Parada precoce (quando estagnado por muitas gerações)
    
    Returns:
        best_layouts: Lista dos N_TOP_LAYOUTS melhores layouts encontrados
        last_gen_phase1: Número da última geração executada
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
    hof = tools.HallOfFame(maxsize=HOF_SIZE_P1)  # Guarda os melhores layouts
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

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

        if (current_max_fitness - last_max_fitness) < MIN_DELTA_P1:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        last_max_fitness = current_max_fitness

        if stagnation_counter >= PATIENCE_P1:
            if not aggressive_phase_triggered:
                print(f"--- Stagnation detected at gen {gen}. Increasing sigma to {SIGMA_AGGRESSIVE_P1} for {AGGRESSIVE_DURATION_P1} generations. ---")
                toolbox_phase1.register("mutate", mutate_phase1, mu=MU_P1, sigma=SIGMA_AGGRESSIVE_P1, indpb=INDPB_P1)
                aggressive_phase_triggered = True
                aggressive_phase_countdown = AGGRESSIVE_DURATION_P1
                stagnation_counter = 0
            else:
                print(f"--- Stagnation persists after aggressive mutation. Stopping early at generation {gen}. ---")
                break

        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                print(f"--- End of aggressive mutation phase at generation {gen}. Reverting sigma to {SIGMA_NORMAL_P1}. ---")
                toolbox_phase1.register("mutate", mutate_phase1, mu=MU_P1, sigma=SIGMA_NORMAL_P1, indpb=INDPB_P1)

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
    n_top = min(N_TOP_LAYOUTS, len(top_layouts))
    best_layouts = top_layouts[:n_top]
    
    print(f"\nFase 1 concluída. Melhor AEP bruto: {hof[0].fitness.values[0]/1000:.2f} GWh")
    print(f"Selecionados {len(best_layouts)} melhores layouts para Fase 2")
    
    # Retorna layouts e número da última geração
    last_gen_phase1 = gen  # Última geração executada (pode ser menor que NGEN se parou cedo)
    return best_layouts, last_gen_phase1

def optimize_phase2(best_layouts_phase1, POP_SIZE, NGEN, CXPB, MUTPB, start_gen_number=0):
    """
    Fase 2: Otimização Multiobjetivo (AEP Líquido + Custo).
    
    Esta fase parte dos melhores layouts da Fase 1 e refina considerando:
    - Cabeamento completo (cálculo de custo e perdas Joule)
    - Posição otimizada da subestação offshore
    - Número otimizado de grupos de cabeamento
    - Trade-offs entre AEP líquido e Custo
    
    Algoritmo: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    - Seleção: NSGA-II (mantém diversidade na frente de Pareto)
    - Blend Crossover para todas as variáveis
    - Mutação Gaussiana diferenciada por componente
    - Sistema de detecção de estagnação multiobjetivo
    
    Inicialização da população:
    - Usa os melhores layouts da Fase 1 como sementes
    - Diversifica posições da subestação ao redor do centroide
    - Adiciona perturbações para manter diversidade
    
    Args:
        best_layouts_phase1: Lista dos melhores layouts da Fase 1
        POP_SIZE: Tamanho da população
        NGEN: Número máximo de gerações
        CXPB: Probabilidade de crossover
        MUTPB: Probabilidade de mutação
        start_gen_number: Número da geração inicial (continua numeração da Fase 1)
    
    Returns:
        hof: Hall of Fame com a frente de Pareto final (soluções não-dominadas)
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
    for i, layout in enumerate(best_layouts_phase1):
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
        
        # Inicializa posição da subestação como centroide do layout (como k-means)
        # O centroide é o ponto médio das turbinas, que é uma boa posição inicial
        centroide = np.mean(coords, axis=0)
        
        # Diversifica posição inicial da subestação ao redor do centroide
        # Isso permite exploração mantendo a subestação próxima ao centro das turbinas
        mod = len(pop) % 5
        if mod == 0:
            # Exatamente no centroide (sem variação)
            substation_pos = centroide.tolist()
        elif mod == 1:
            # Próxima ao centroide (pequena variação)
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.2),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.2)
            ]
        elif mod == 2:
            # Média distância do centroide (variação moderada)
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.4),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.4)
            ]
        elif mod == 3:
            # Longe do centroide (variação grande) - exploração
            substation_pos = [
                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.6),
                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.6)
            ]
        else:  # mod == 4
            # Muito longe do centroide (exploração agressiva)
            # Usa distribuição circular ao redor do centroide
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(CIRCLE_RADIUS * 0.5, CIRCLE_RADIUS * 0.9)
            substation_pos = [
                centroide[0] + radius * np.cos(angle),
                centroide[1] + radius * np.sin(angle)
            ]
        
        # Garante que a subestação fique dentro do círculo
        substation_pos = enforce_substation_circle(np.array(substation_pos)).tolist()
        
        # Cria indivíduo da Fase 2 com posição variada da subestação
        ind_phase2 = create_individual_phase2_from_coords(coords, substation_pos=substation_pos)
        pop.append(ind_phase2)
    
    # Completa população usando os melhores layouts da Fase 1 (não coordenadas iniciais!)
    # Se precisar de mais indivíduos, usa os melhores layouts com perturbações
    n_remaining = POP_SIZE - len(pop)
    if n_remaining > 0:
        # Repete os melhores layouts da Fase 1 com perturbações para diversidade
        # Isso garante que todos os indivíduos partam de layouts bons, não ruins
        for i in range(n_remaining):
            # Seleciona um layout da Fase 1 (cicla pelos melhores)
            layout_idx = i % len(best_layouts_phase1)
            layout = best_layouts_phase1[layout_idx]
            
            # Converte para coordenadas
            coords_list = list(layout)
            coords_array = np.array(coords_list, dtype=float)
            coords = coords_array.reshape((IND_SIZE, 2))
            
            # Adiciona perturbação maior para diversidade, mas mantém base nos melhores layouts
            coords_perturbed = coords.copy()
            # Perturbação variável para explorar mais o espaço
            perturbation_sigma = random.uniform(PERTURBATION_SIGMA_MIN, PERTURBATION_SIGMA_MAX)
            for j in range(len(coords_perturbed)):
                coords_perturbed[j, 0] += random.gauss(0, perturbation_sigma)
                coords_perturbed[j, 1] += random.gauss(0, perturbation_sigma)
            
            # Garante que está dentro do círculo
            for j in range(len(coords_perturbed)):
                x, y = coords_perturbed[j, 0], coords_perturbed[j, 1]
                dist = np.sqrt(x**2 + y**2)
                if dist > CIRCLE_RADIUS:
                    angle = np.arctan2(y, x)
                    coords_perturbed[j, 0] = CIRCLE_RADIUS * np.cos(angle)
                    coords_perturbed[j, 1] = CIRCLE_RADIUS * np.sin(angle)
            
            # Calcula centroide das turbinas perturbadas
            centroide_perturbed = np.mean(coords_perturbed, axis=0)
            
            # Posição da subestação ao redor do centroide (variação para exploração)
            substation_pos_perturbed = [
                centroide_perturbed[0] + random.gauss(0, CIRCLE_RADIUS * 0.4),
                centroide_perturbed[1] + random.gauss(0, CIRCLE_RADIUS * 0.4)
            ]
            
            # Garante que a subestação fique dentro do círculo
            substation_pos_perturbed = enforce_substation_circle(np.array(substation_pos_perturbed)).tolist()
            
            ind_phase2 = create_individual_phase2_from_coords(coords_perturbed, substation_pos=substation_pos_perturbed)
            pop.append(ind_phase2)
    
    # Avalia população inicial
    print(f"\nAvaliando população inicial da Fase 2 ({len(pop)} indivíduos)...")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    if len(invalid_ind) > 0:
        fitnesses = toolbox_phase2.map(toolbox_phase2.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
    # Verifica se há indivíduos válidos (fitness > 0)
    valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
    if len(valid_pop) == 0:
        print("\nERRO: Todos os indivíduos da população inicial foram rejeitados!")
        print("Isso pode acontecer se todas as soluções têm cruzamentos ou múltiplas conexões.")
        print("Tentando criar população alternativa com layouts mais simples...")
        
        # Tenta criar população alternativa com layouts mais simples (menos grupos)
        pop = []
        for i, layout in enumerate(best_layouts_phase1[:min(10, len(best_layouts_phase1))]):
            coords_list = list(layout)
            if len(coords_list) != IND_SIZE * 2:
                continue
            coords_array = np.array(coords_list, dtype=float)
            coords = coords_array.reshape((IND_SIZE, 2))
            centroide = np.mean(coords, axis=0)
            substation_pos = centroide.tolist()  # Subestação exatamente no centroide
            # Garante que a subestação fique dentro do círculo
            substation_pos = enforce_substation_circle(np.array(substation_pos)).tolist()
            ind_phase2 = create_individual_phase2_from_coords(coords, substation_pos=substation_pos)
            pop.append(ind_phase2)
        
        # Avalia novamente
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if len(invalid_ind) > 0:
            fitnesses = toolbox_phase2.map(toolbox_phase2.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
        if len(valid_pop) == 0:
            raise ValueError("Não foi possível criar população inicial válida. Verifique as penalidades de cruzamento.")
        
        print(f"População alternativa criada: {len(valid_pop)} indivíduos válidos de {len(pop)} totais")
        pop = valid_pop
    
    hof = tools.ParetoFront()
    
    stats_aep = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_cost = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats = tools.MultiStatistics(aep=stats_aep, cost=stats_cost)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Usa parâmetros de estagnação definidos no início do arquivo
    
    stagnation_counter = 0
    last_best_aep = 0.0
    last_best_cost = float('inf')
    
    def select_best_tradeoff_solution(hof_valid):
        """
        Seleciona a melhor solução usando o método do knee point (ponto de joelho).
        O knee point é a solução que minimiza a distância normalizada ao ponto ideal (máximo AEP, mínimo custo).
        Isso garante que selecionamos uma solução com bom trade-off, não apenas a de maior AEP.
        """
        if len(hof_valid) == 0:
            return None
        
        if len(hof_valid) == 1:
            return hof_valid[0]
        
        # Extrai AEP e custo de todas as soluções
        aeps = np.array([ind.fitness.values[0] for ind in hof_valid])
        costs = np.array([ind.fitness.values[1] for ind in hof_valid])
        
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
        for i in range(len(hof_valid)):
            point = np.array([aep_norm[i], cost_norm[i]])
            dist = np.linalg.norm(ideal_point - point)
            distances.append(dist)
        
        # Retorna a solução com menor distância ao ideal (knee point)
        knee_idx = np.argmin(distances)
        return hof_valid[knee_idx]
    
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
            
            # Salva solução com melhor trade-off (gen 0) usando nova diretiva
            best_ind0 = select_best_tradeoff_solution(hof_valid)
            
            gen_number = start_gen_number + 1  # +1 porque gen 0 da Fase 2 é após última gen da Fase 1
            n_coords = IND_SIZE * 2
            coords_flat0 = best_ind0[:n_coords]
            coords0 = np.array(coords_flat0).reshape((IND_SIZE, 2))
            n_grupos_normalizado0 = best_ind0[n_coords]
            n_grupos_float0 = MIN_GRUPOS + n_grupos_normalizado0 * (MAX_GRUPOS - MIN_GRUPOS)
            n_grupos0 = int(np.round(n_grupos_float0))
            n_grupos0 = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos0))
            substation_pos0 = np.array([best_ind0[n_coords + 1], best_ind0[n_coords + 2]])
            
            evolution_file0 = os.path.join(evolution_dir, f"gen_{gen_number:04d}_best.txt")
            with open(evolution_file0, 'w') as f:
                x_str = ", ".join([f"{val:.12f}" for val in coords0[:, 0]])
                y_str = ", ".join([f"{val:.12f}" for val in coords0[:, 1]])
                f.write(f"xc: [{x_str}]\n")
                f.write(f"yc: [{y_str}]\n")
                f.write(f"aep: {best_ind0.fitness.values[0]:.2f}\n")
                f.write(f"cost: {best_ind0.fitness.values[1]:.2f}\n")
                f.write(f"n_grupos: {n_grupos0}\n")
                f.write(f"substation_x: {substation_pos0[0]:.12f}\n")
                f.write(f"substation_y: {substation_pos0[1]:.12f}\n")
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
        
        # Verifica se há indivíduos válidos antes de compilar estatísticas
        valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
        n_valid = len(valid_pop)
        
        if n_valid == 0:
            print(f"\nAVISO: Gen {gen}: Nenhum indivíduo válido na população!")
            print("Todos os indivíduos foram rejeitados. Tentando recuperar população...")
            # Tenta usar indivíduos do hall of fame se disponível
            if len(hof_valid) > 0:
                pop = hof_valid[:POP_SIZE]
                # Completa com novos indivíduos se necessário
                while len(pop) < POP_SIZE:
                    # Cria novo indivíduo baseado nos melhores
                    if len(best_layouts_phase1) > 0:
                        base_layout = random.choice(best_layouts_phase1)
                        coords_list = list(base_layout)
                        if len(coords_list) == IND_SIZE * 2:
                            coords_array = np.array(coords_list, dtype=float)
                            coords = coords_array.reshape((IND_SIZE, 2))
                            centroide = np.mean(coords, axis=0)
                            substation_pos = [
                                centroide[0] + random.gauss(0, CIRCLE_RADIUS * 0.3),
                                centroide[1] + random.gauss(0, CIRCLE_RADIUS * 0.3)
                            ]
                            ind_phase2 = create_individual_phase2_from_coords(coords, substation_pos=substation_pos)
                            pop.append(ind_phase2)
                # Reavalia
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                if len(invalid_ind) > 0:
                    fitnesses = toolbox_phase2.map(toolbox_phase2.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
                n_valid = len(valid_pop)
        
        if n_valid == 0:
            raise ValueError(f"Gen {gen}: Não foi possível recuperar população válida. Parando otimização.")
        
        # Compila estatísticas apenas com indivíduos válidos
        record = stats.compile(valid_pop)
        
        aep_max_mwh = record['aep']['max']
        cost_min_usd = record['cost']['min']
        
        # Detecção de estagnação (verifica melhorias nos extremos da frente de Pareto)
        current_best_aep = aep_max_mwh
        current_best_cost = cost_min_usd
        
        # Verifica se houve melhoria significativa em AEP ou custo
        # Compara com o melhor histórico, não apenas com a última geração
        aep_improved = (current_best_aep - last_best_aep) >= MIN_DELTA_AEP_P2
        cost_improved = (last_best_cost - current_best_cost) >= MIN_DELTA_COST_P2
        
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
        if stagnation_counter >= PATIENCE_P2:
            print(f"\n--- Gen {gen}: Estagnação detectada na Fase 2. Parando precocemente. ---")
            print(f"   Melhor AEP: {last_best_aep/1000:.2f} GWh, Melhor Custo: {last_best_cost:.2f} USD")
            print(f"   Sem melhoria significativa por {PATIENCE_P2} gerações.")
            break
        
        # Calcula número médio de grupos
        if n_valid > 0:
            n_coords = IND_SIZE * 2  # Índice onde começa n_grupos
            grupos_list = [
                int(np.round(MIN_GRUPOS + ind[n_coords] * (MAX_GRUPOS - MIN_GRUPOS))) 
                for ind in pop if ind.fitness.values[0] > 0
            ]
            n_grupos_medio = int(np.round(np.mean(grupos_list))) if grupos_list else 0
        else:
            n_grupos_medio = 0
        
        # Salva solução com melhor trade-off usando nova diretiva
        if len(hof_valid) > 0:
            best_ind = select_best_tradeoff_solution(hof_valid)
            
            gen_number = start_gen_number + gen
            n_coords = IND_SIZE * 2
            coords_flat = best_ind[:n_coords]
            coords = np.array(coords_flat).reshape((IND_SIZE, 2))
            n_grupos_normalizado = best_ind[n_coords]
            n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
            n_grupos = int(np.round(n_grupos_float))
            n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
            substation_pos = np.array([best_ind[n_coords + 1], best_ind[n_coords + 2]])
            
            evolution_file = os.path.join(evolution_dir, f"gen_{gen_number:04d}_best.txt")
            with open(evolution_file, 'w') as f:
                x_str = ", ".join([f"{val:.12f}" for val in coords[:, 0]])
                y_str = ", ".join([f"{val:.12f}" for val in coords[:, 1]])
                f.write(f"xc: [{x_str}]\n")
                f.write(f"yc: [{y_str}]\n")
                f.write(f"aep: {best_ind.fitness.values[0]:.2f}\n")
                f.write(f"cost: {best_ind.fitness.values[1]:.2f}\n")
                f.write(f"n_grupos: {n_grupos}\n")
                f.write(f"substation_x: {substation_pos[0]:.12f}\n")
                f.write(f"substation_y: {substation_pos[1]:.12f}\n")
                f.write(f"phase: 2\n")
        
        if gen % 1 == 0 or gen == NGEN:
            print(f"Gen {gen}: AEP Max={aep_max_mwh:.2f} MWh ({aep_max_mwh/1000:.2f} GWh), "
                  f"Cost Min={cost_min_usd:.2f} USD, "
                  f"Grupos Médio={n_grupos_medio}, "
                  f"Valid={n_valid}/{len(pop)}, Pareto={len(hof)}, Stagnation={stagnation_counter}/{PATIENCE_P2}")
    
    pool.close()
    pool.join()
    
    return hof

def save_results(hof_final):
    """
    Salva os resultados da Fase 2: frente de Pareto completa.
    
    Para cada solução na frente de Pareto:
    - Salva coordenadas das turbinas
    - Salva número de grupos de cabeamento
    - Salva posição da subestação
    - Recalcula perdas Joule para consistência
    
    Gera:
    - Arquivos individuais: solution_N_coords.txt (uma por solução)
    - Arquivo CSV: pareto_summary.csv (resumo de todas as soluções)
    """
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
        
        # Extrai coordenadas, número de grupos e posição da subestação
        n_coords = IND_SIZE * 2
        coords_flat = individual[:n_coords]
        n_grupos_normalizado = individual[n_coords]
        n_grupos_float = MIN_GRUPOS + n_grupos_normalizado * (MAX_GRUPOS - MIN_GRUPOS)
        n_grupos = int(np.round(n_grupos_float))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        substation_pos = np.array([individual[n_coords + 1], individual[n_coords + 2]])
        
        coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # Recalcula perdas para salvar usando posição da subestação do genoma
        # IMPORTANTE: Adiciona a subestação às coordenadas para que o cálculo use a posição real
        coords_with_substation = np.vstack([coords, substation_pos.reshape(1, 2)])
        substation_idx = IND_SIZE  # Índice da subestação após todas as turbinas
        
        try:
            _, resultados_cabeamento = cabling_v3.analisar_layout_completo(
                coords_with_substation, sub=substation_idx, n_grupos=n_grupos)
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
            f.write(f"substation_x: {substation_pos[0]:.12f}\n")
            f.write(f"substation_y: {substation_pos[1]:.12f}\n")
        
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
    """
    Função principal: executa a estratégia híbrida em duas fases.
    
    Fluxo de execução:
    1. Fase 1: Otimização rápida de layout (AEP bruto apenas)
       - Explora intensamente o espaço de layouts
       - Encontra configurações com alto AEP bruto
       - Seleciona os N_TOP_LAYOUTS melhores layouts
    
    2. Fase 2: Otimização multiobjetivo (AEP líquido + Custo)
       - Parte dos melhores layouts da Fase 1
       - Refina considerando cabeamento completo
       - Encontra a frente de Pareto otimizando trade-offs
    
    3. Salva resultados: frente de Pareto completa em arquivos
    
    Vantagens da estratégia híbrida:
    - Fase 1: 10-50x mais rápida (sem cálculo de cabeamento)
    - Fase 2: Parte de soluções boas, foca em refinamento
    - Resultado: Melhor qualidade de soluções em menos tempo total
    """
    random.seed(42)  # Semente fixa para reprodutibilidade
    start_time = time.time()
    
    print("=" * 80)
    print("ESTRATÉGIA HÍBRIDA EM DUAS FASES")
    print("Fase 1: Otimização de Layout (AEP Bruto) - Exploração")
    print("Fase 2: Otimização Multiobjetivo (AEP Líquido + Custo) - Refinamento")
    print("=" * 80)
    
    # Fase 1: Otimização de Layout (rápida, sem cabeamento)
    best_layouts, last_gen_phase1 = optimize_phase1(POP_SIZE_P1, NGEN_P1, CXPB_P1, MUTPB_P1)
    
    # Fase 2: Otimização Multiobjetivo (continua numeração da Fase 1)
    # Usa os melhores layouts da Fase 1 como ponto de partida
    hof_final = optimize_phase2(best_layouts, POP_SIZE_P2, NGEN_P2, CXPB_P2, MUTPB_P2, start_gen_number=last_gen_phase1)
    
    # Salva resultados: frente de Pareto completa
    save_results(hof_final)
    
    total_time = time.time() - start_time
    print(f"\nTempo total: {total_time/60:.2f} minutos")

if __name__ == "__main__":
    main()

