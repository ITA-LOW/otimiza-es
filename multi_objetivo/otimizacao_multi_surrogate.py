# multi_objetivo/otimizacao_multi_surrogate.py

import sys
import os
import time
import json
import joblib
import numpy as np
import random
import multiprocessing

from deap import base, creator, tools, algorithms

# --- Tenta importar TensorFlow e dá uma mensagem de erro clara se não estiver instalado ---
try:
    import tensorflow as tf
except ImportError:
    print("Erro: A biblioteca 'tensorflow' é necessária para executar a otimização com surrogate.")
    print("Por favor, instale as dependências com: pip install -r requirements.txt")
    sys.exit(1)

# --- Configurações de Path e Módulos do Projeto ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from multi_objetivo.cabling import analisar_layout_completo

# --- 1. CONFIGURAÇÃO DA DEAP (igual ao original) ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# --- 2. PARÂMETROS GLOBAIS ---
IND_SIZE = 16
CIRCLE_RADIUS = 1300
MIN_TURB_DIST_IN_DIAMETERS = 2.0
N_GRUPOS_CABEAMENTO = int(np.sqrt(IND_SIZE))
SUBSTATION_BOUNDS = [-1500, -1300, -100, 100]

# --- PARÂMETROS DA OTIMIZAÇÃO TURBINADA ---
# Aumentamos drasticamente o número de gerações graças à velocidade do modelo de IA
NGEN = 2000
POP_SIZE = 300
CXPB = 0.95  # Probabilidade de Crossover
MUTPB = 0.7  # Probabilidade de Mutação

# --- 3. PRÉ-CARREGAMENTO DE DADOS E MODELOS ---
print("Carregando configurações, dados e modelos de IA...")
config_dir = os.path.join(project_root, 'config')
output_dir = os.path.join(project_root, 'output', 'multi_objective_sub')

# Carrega dados físicos (necessário para a verificação final)
try:
    initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(os.path.join(config_dir, f'iea37-ex{IND_SIZE}.yaml'))
    TURB_ATRBT_DATA = getTurbAtrbtYAML(os.path.join(config_dir, fname_turb))
    WIND_ROSE_DATA = getWindRoseYAML(os.path.join(config_dir, fname_wr))
    TURB_DIAM = TURB_ATRBT_DATA[-1]
    MIN_TURB_DIST_METERS = MIN_TURB_DIST_IN_DIAMETERS * TURB_DIAM
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos de configuração: {e}")
    sys.exit(1)

# Carrega os artefatos de Machine Learning
try:
    surrogate_model = tf.keras.models.load_model(os.path.join(output_dir, 'surrogate_model.h5'))
    input_scaler = joblib.load(os.path.join(output_dir, 'input_scaler.pkl'))
    output_scaler = joblib.load(os.path.join(output_dir, 'output_scaler.pkl'))
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos do modelo de IA: {e}")
    print("Certifique-se de que 'train_surrogate_model.py' foi executado com sucesso.")
    sys.exit(1)

# --- 4. FUNÇÕES DE AVALIAÇÃO ---

# Função de penalidade rápida (apenas geometria)
def calculate_penalties(individual):
    turb_coords = np.array(individual[:-2]).reshape((IND_SIZE, 2))
    
    # Penalidade por turbinas fora do círculo
    penalty_out_of_circle = np.sum(np.maximum(0, np.linalg.norm(turb_coords, axis=1) - CIRCLE_RADIUS)) * 1e6

    # Penalidade por turbinas muito próximas
    penalty_close_turbines = 0
    if IND_SIZE > 1:
        dist_matrix = np.linalg.norm(turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :], axis=2)
        upper_tri_indices = np.triu_indices(IND_SIZE, k=1)
        close_distances = dist_matrix[upper_tri_indices]
        violations = close_distances[close_distances < MIN_TURB_DIST_METERS]
        penalty_close_turbines = np.sum(MIN_TURB_DIST_METERS - violations) * 1e6
        
    return penalty_out_of_circle + penalty_close_turbines

# Nova função de avaliação RÁPIDA usando o modelo de IA
def evaluate_surrogate(individual):
    # Prepara o input para o modelo (reshape e scale)
    input_data = np.array(individual).reshape(1, -1)
    input_scaled = input_scaler.transform(input_data)

    # Faz a predição com o modelo de IA
    prediction_scaled = surrogate_model.predict(input_scaled)

    # Reverte a escala para obter os valores físicos
    prediction = output_scaler.inverse_transform(prediction_scaled)
    predicted_aep, predicted_cost = prediction[0]

    # Calcula as penalidades geométricas (operação rápida)
    penalty = calculate_penalties(individual)

    # Combina a predição com a penalidade para o fitness do AG
    aep_fitness = predicted_aep - penalty
    
    return aep_fitness, predicted_cost

# Função de avaliação de ALTA FIDELIDADE (original, para verificação final)
def evaluate_high_fidelity(individual):
    # (Esta é a função 'evaluate_multi_objective' do script original)
    turb_coords = np.array(individual[:-2]).reshape((IND_SIZE, 2))
    sub_coords = np.array(individual[-2:])
    
    aep_bruto = np.sum(calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], WIND_ROSE_DATA[0],
                               TURB_ATRBT_DATA[4], TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
    
    layout_com_sub = np.vstack([sub_coords.reshape(1, 2), turb_coords])
    try:
        _, resultados_cabeamento = analisar_layout_completo(
            coordenadas=layout_com_sub, substation_idx=0,
            n_grupos=N_GRUPOS_CABEAMENTO, Vn=33e3, P_turbina=TURB_ATRBT_DATA[3])
        cabling_cost = resultados_cabeamento['custo_total_usd']
        perda_anual_cabeamento = resultados_cabeamento['perda_anual_mwh']
    except Exception:
        return 0, 2e9

    aep_liquido = aep_bruto - perda_anual_cabeamento
    return aep_liquido, cabling_cost

# --- 5. CONFIGURAÇÃO DA TOOLBOX (apontando para a nova função) ---
toolbox = base.Toolbox()
# ... (criação e mutação de indivíduos são iguais ao original) ...
def create_individual():
    turb_part = np.array(initial_coordinates).flatten().tolist()
    sub_x = random.uniform(SUBSTATION_BOUNDS[0], SUBSTATION_BOUNDS[1])
    sub_y = random.uniform(SUBSTATION_BOUNDS[2], SUBSTATION_BOUNDS[3])
    return creator.Individual(turb_part + [sub_x, sub_y])
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def custom_mutate(individual, mu, sigma, indpb):
    size_turbines = len(individual) - 2
    for i in range(size_turbines):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
    if random.random() < indpb:
        individual[-2] += random.gauss(mu, sigma)
        individual[-1] += random.gauss(mu, sigma)
        individual[-2] = np.clip(individual[-2], SUBSTATION_BOUNDS[0], SUBSTATION_BOUNDS[1])
        individual[-1] = np.clip(individual[-1], SUBSTATION_BOUNDS[2], SUBSTATION_BOUNDS[3])
    return individual,
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", custom_mutate, mu=0, sigma=50, indpb=0.3) # Sigma um pouco menor
toolbox.register("select", tools.selNSGA2)

# <<< A MUDANÇA PRINCIPAL ESTÁ AQUI >>>
toolbox.register("evaluate", evaluate_surrogate)

# --- 6. FUNÇÃO PRINCIPAL (MAIN) ---
def main():
    random.seed(42)
    
    # Multiprocessing não é necessário com o surrogate, pois a avaliação é quase instantânea
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)
    pareto_front = tools.ParetoFront()
    
    print("--- Início da Evolução RÁPIDA com Surrogate Model ---")
    
    # Avaliação inicial da população
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    pareto_front.update(pop)
    
    start_time = time.time()
    for gen in range(1, NGEN + 1):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop = toolbox.select(offspring + pop, POP_SIZE)
        pareto_front.update(pop)

        if gen % 100 == 0: # Imprime o log a cada 100 gerações
             print(f"Geração {gen}/{NGEN} | Soluções na Fronteira: {len(pareto_front)}")
    
    end_time = time.time()
    print(f"--- Fim da Evolução RÁPIDA em {(end_time-start_time):.2f} segundos ---")
    
    # --- 7. VERIFICAÇÃO FINAL (ETAPA CRUCIAL) ---
    print(f"\n--- Verificando as {len(pareto_front)} soluções da fronteira com a simulação de alta fidelidade ---")
    verified_solutions = []
    
    for i, ind in enumerate(pareto_front):
        # Roda a simulação original e precisa
        true_aep, true_cost = evaluate_high_fidelity(ind)
        
        solution = {
            'rank': i,
            'verified_aep': true_aep,
            'verified_cost_usd': true_cost,
            'surrogate_aep_fitness': ind.fitness.values[0], # AEP com penalidade que o AG viu
            'surrogate_cost': ind.fitness.values[1],
            'layout_turbines': np.array(ind[:-2]).reshape((IND_SIZE, 2)).tolist(),
            'layout_substation': np.array(ind[-2:]).tolist()
        }
        verified_solutions.append(solution)
        print(f"Solução {i}: AEP Verificado={true_aep:.2f}, Custo Verificado={true_cost:.2f}")

    # Salva os resultados verificados
    results_path = os.path.join(output_dir, 'pareto_solutions_surrogate_verified.json')
    with open(results_path, 'w') as f:
        json.dump(verified_solutions, f, indent=2)

    print(f"\nResultados VERIFICADOS salvos em: {results_path}")

if __name__ == "__main__":
    main()
