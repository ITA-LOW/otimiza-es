# generate_training_data.py

import sys
import os
import time
import numpy as np
import pandas as pd
import random

# --- Configurações e Inserção de Path ---
# Garante que os módulos do projeto possam ser importados
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Tenta importar o SciPy e dá uma mensagem de erro clara se não estiver instalado
try:
    from scipy.stats import qmc
except ImportError:
    print("Erro: A biblioteca 'scipy' é necessária para a geração de dados.")
    print("Por favor, instale as dependências com: pip install -r requirements.txt")
    sys.exit(1)

# Importações dos módulos do projeto
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from multi_objetivo.cabling import analisar_layout_completo

# --- 1. Configurações Globais (do script de otimização) ---
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio para o posicionamento das turbinas
MIN_TURB_DIST_IN_DIAMETERS = 2.0
SUBSTATION_BOUNDS = [-1500, -1300, -100, 100]  # [xmin, xmax, ymin, ymax]
N_GRUPOS_CABEAMENTO = int(np.sqrt(IND_SIZE))
N_SAMPLES = 5000  # Número de amostras de dados a serem geradas
REPRODUCIBILITY_SEED = 42  # Semente para garantir que o LHS gere sempre os mesmos layouts

# --- 2. Pré-carregamento de Dados (essencial para a função de avaliação) ---
print("Carregando arquivos de configuração...")
config_dir = os.path.join(project_root, 'config')
main_yaml_path = os.path.join(config_dir, f'iea37-ex{IND_SIZE}.yaml')
try:
    _, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)
    full_path_turb = os.path.join(config_dir, fname_turb)
    full_path_wr = os.path.join(config_dir, fname_wr)

    TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
    WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)
    TURB_DIAM = TURB_ATRBT_DATA[-1]
    MIN_TURB_DIST_METERS = MIN_TURB_DIST_IN_DIAMETERS * TURB_DIAM
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos de configuração: {e}")
    sys.exit(1)

# --- 3. Função de Avaliação (com lógica de validação) ---
# Esta função calcula o AEP e o Custo para um único layout
def evaluate_multi_objective(individual):
    turb_coords = np.array(individual[:-2]).reshape((IND_SIZE, 2))
    sub_coords = np.array(individual[-2:])

    # --- Verificação de Validade do Layout ---
    # 1. Turbinas fora do círculo (não deve acontecer com a geração polar, mas é uma boa checagem de segurança)
    if np.any(np.linalg.norm(turb_coords, axis=1) > CIRCLE_RADIUS):
        return 0, 2e9  # Retorna AEP nulo e Custo altíssimo

    # 2. Turbinas muito próximas
    if IND_SIZE > 1:
        # Pega a parte triangular superior da matriz de distância para evitar checagens duplicadas e a diagonal
        upper_tri_indices = np.triu_indices(IND_SIZE, k=1)
        dist_matrix = np.linalg.norm(turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :], axis=2)
        if np.any(dist_matrix[upper_tri_indices] < MIN_TURB_DIST_METERS):
            return 0, 2e9  # Retorna AEP nulo e Custo altíssimo

    # --- Se o layout é válido, prossegue com os cálculos ---
    
    # AEP Bruto
    turb_ci, turb_co, rated_ws, rated_pwr, _ = TURB_ATRBT_DATA
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                               TURB_DIAM, turb_ci, turb_co, rated_ws, rated_pwr))
    
    # Custo e Perdas do Cabeamento
    layout_com_sub = np.vstack([sub_coords.reshape(1, 2), turb_coords])
    try:
        _, resultados_cabeamento = analisar_layout_completo(
            coordenadas=layout_com_sub, substation_idx=0,
            n_grupos=N_GRUPOS_CABEAMENTO, Vn=33e3, P_turbina=rated_pwr)
        cabling_cost = resultados_cabeamento['custo_total_usd']
        perda_anual_cabeamento = resultados_cabeamento['perda_anual_mwh']
    except Exception:
        # Se o cabeamento falhar mesmo em um layout geometricamente válido,
        # consideramos o resultado como péssimo.
        return 0, 2e9

    aep_liquido = aep_bruto - perda_anual_cabeamento
        
    return aep_liquido, cabling_cost

# --- 4. Geração de Dados com Hipercubo Latino (Método Otimizado) ---
def generate_layouts_lhs():
    print(f"Iniciando geração de {N_SAMPLES} layouts com Amostragem por Hipercubo Latino...")
    
    # Cada layout tem 16 turbinas (r, theta) e 1 subestação (x,y) = 34 dimensões
    n_dimensions = IND_SIZE * 2 + 2
    
    # 1. Gerar amostras em um hipercubo unitário [0, 1]
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=REPRODUCIBILITY_SEED)
    unit_samples = sampler.random(n=N_SAMPLES)
    
    layouts = []
    for sample in unit_samples:
        individual = []
        
        # --- Processa as Turbinas (32 dimensões -> 16 pares de r, theta) ---
        turb_samples = sample[:-2]
        for i in range(IND_SIZE):
            # u_r é a amostra para o componente do raio (requer transformação)
            # u_theta é a amostra para o ângulo
            u_r = turb_samples[2*i]
            u_theta = turb_samples[2*i + 1]
            
            # Para garantir distribuição espacial uniforme, o raio é r = R * sqrt(u)
            r = CIRCLE_RADIUS * np.sqrt(u_r)
            theta = 2 * np.pi * u_theta
            
            # Converte de polar para cartesiano
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            individual.extend([x, y])
            
        # --- Processa a Subestação (últimas 2 dimensões) ---
        sub_x_sample = sample[-2]
        sub_y_sample = sample[-1]
        
        # Mapeia a amostra [0,1] para os limites da subestação
        sub_x = SUBSTATION_BOUNDS[0] + sub_x_sample * (SUBSTATION_BOUNDS[1] - SUBSTATION_BOUNDS[0])
        sub_y = SUBSTATION_BOUNDS[2] + sub_y_sample * (SUBSTATION_BOUNDS[3] - SUBSTATION_BOUNDS[2])
        individual.extend([sub_x, sub_y])
        
        layouts.append(individual)

    print(f"Amostragem LHS em coordenadas polares gerou {len(layouts)} layouts válidos.")
    return layouts

# --- 5. Orquestração Principal ---
if __name__ == "__main__":
    layouts_to_evaluate = generate_layouts_lhs()
    
    all_inputs = []
    all_aep = []
    all_cost = []
    
    print(f"Avaliando {len(layouts_to_evaluate)} layouts (isso pode levar vários minutos)...")
    start_time = time.time()

    # Usar multiprocessing para acelerar a avaliação
    from multiprocessing import Pool
    
    # O número de processos será o número de CPUs disponíveis - 1
    num_processes = max(1, os.cpu_count() - 1)
    
    with Pool(processes=num_processes) as pool:
        # pool.map só aceita um argumento, então criamos uma função wrapper se necessário
        # Como evaluate_multi_objective aceita um argumento, podemos usá-la diretamente
        results = pool.map(evaluate_multi_objective, layouts_to_evaluate)

    end_time = time.time()
    
    # Processar resultados
    for i, (aep, cost) in enumerate(results):
        all_inputs.append(layouts_to_evaluate[i])
        all_aep.append(aep)
        all_cost.append(cost)

    print(f"Avaliação concluída em {(end_time - start_time)/60:.2f} minutos.")
    print(f"Total de amostras válidas geradas: {len(all_inputs)}")

    # Criar DataFrame com os resultados
    input_col_names = [f't{j}_{coord}' for j in range(IND_SIZE) for coord in ['x', 'y']]
    input_col_names += ['sub_x', 'sub_y']
    
    df_inputs = pd.DataFrame(all_inputs, columns=input_col_names)
    df_outputs = pd.DataFrame({'aep': all_aep, 'cost': all_cost})
    
    df_combined = pd.concat([df_inputs, df_outputs], axis=1).dropna()
    
    # Salvar em um arquivo CSV
    output_dir = os.path.join(project_root, 'output', 'multi_objective_sub')
    os.makedirs(output_dir, exist_ok=True)
    training_data_path = os.path.join(output_dir, 'training_data.csv')
    df_combined.to_csv(training_data_path, index=False)
    
    print(f"\nArquivo de dados de treinamento foi salvo com sucesso em:")
    print(training_data_path)
