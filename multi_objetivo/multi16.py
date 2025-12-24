import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# wind_farm_GA_multi.py

import time
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools

# Módulos customizados do projeto
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import cabling

# =============================================================================
# CONFIGURAÇÃO DO AMBIENTE DEAP E CONSTANTES
# =============================================================================

# Define o problema multiobjetivo: Maximizar AEP (1.0), Minimizar Custo (-1.0)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Cria a toolbox DEAP
toolbox = base.Toolbox()

# Parâmetros do Parque Eólico e da Otimização
IND_SIZE = 16
CIRCLE_RADIUS = 1300
N_DIAMETERS = 260
SUBSTATION_CONTINENT = np.array([[-1350, 0]]) # Subestação externa fixa

# =============================================================================
# FUNÇÕES DE INICIALIZAÇÃO, RESTRIÇÃO E MUTAÇÃO (LÓGICA ORIGINAL)
# =============================================================================

# Carrega o layout inicial como base para a população
config_dir = "config"
main_yaml_path = os.path.join(config_dir, 'iea37-ex16.yaml')
initial_coordinates, fname_turb, fname_wr = getTurbLocYAML(main_yaml_path)
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.array(initial_coordinates).flatten().tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Função de reparo: força as turbinas a ficarem dentro do círculo
def enforce_circle(individual):
    """Modifica a lista 'individual' in-place."""
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i+1]
        if x**2 + y**2 > CIRCLE_RADIUS**2:
            angle = np.arctan2(y, x)
            individual[2*i] = CIRCLE_RADIUS * np.cos(angle)
            individual[2*i+1] = CIRCLE_RADIUS * np.sin(angle)

# Função de mutação original
def mutate(individual, mu, sigma, indpb):
    """Aplica mutação gaussiana e depois repara o indivíduo."""
    individual_arr = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual_arr)):
            individual_arr[i] += random.gauss(mu, sigma)
        
        mutated_list = individual_arr.tolist()
        enforce_circle(mutated_list)
        
        # Copia os valores reparados de volta para o objeto 'individual'
        for i in range(len(individual)):
            individual[i] = mutated_list[i]
            
    return individual,

# Pré-carrega dados do vento e da turbina para evitar I/O repetido
full_path_turb = os.path.join(config_dir, fname_turb)
full_path_wr = os.path.join(config_dir, fname_wr)
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)

# =============================================================================
# FUNÇÃO DE AVALIAÇÃO MULTIOBJETIVO
# =============================================================================

def evaluate_multi_objective(individual):
    """
    Calcula os dois objetivos para um dado layout: AEP Líquido e Custo.
    """
    try:
        turb_coords = np.array(individual).reshape((IND_SIZE, 2))
        
        # Penalidades por violação de restrições de layout
        dist_from_center_sq = np.sum(turb_coords**2, axis=1)
        penalty_out_of_circle = np.sum(dist_from_center_sq > CIRCLE_RADIUS**2) * 1e9

        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(IND_SIZE, k=1)
        penalty_close_turbines = np.sum(dist_matrix[i_upper, j_upper] < N_DIAMETERS) * 1e9
        
        # Cálculo do AEP Bruto (considera efeito de esteira)
        _, _, _, _, turb_diam = TURB_ATRBT_DATA
        aep_bruto = np.sum(calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], WIND_ROSE_DATA[0], turb_diam, *TURB_ATRBT_DATA[0:2], *TURB_ATRBT_DATA[2:4]))

        # Encontra a turbina de coleta (mais próxima do continente)
        distancias_ao_continente = np.linalg.norm(turb_coords - SUBSTATION_CONTINENT, axis=1)
        ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
        
        # Chama o módulo de cabeamento para obter custo e perdas
        _, resultados_cabeamento = cabling.analisar_layout_completo(turb_coords, ponto_de_coleta_idx)
        
        custo_total = resultados_cabeamento['custo_total_usd']
        perdas_joule_mwh = resultados_cabeamento['perda_anual_mwh']
        
        # Calcula os valores finais dos objetivos
        aep_liquido = aep_bruto - perdas_joule_mwh - penalty_out_of_circle - penalty_close_turbines
        custo_penalizado = custo_total + penalty_out_of_circle + penalty_close_turbines

        return aep_liquido, custo_penalizado
    except Exception as e:
        # Se ocorrer um erro, retorna uma fitness muito ruim para eliminar o indivíduo
        print(f"Erro na avaliação: {e}. Penalizando indivíduo.")
        return 0, 1e12

# =============================================================================
# CONFIGURAÇÃO FINAL DA TOOLBOX
# =============================================================================

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate_multi_objective)

# =============================================================================
# FUNÇÃO PRINCIPAL E EXECUÇÃO
# =============================================================================

def main():
    random.seed(42)
    start_time = time.time()
    
    # Parâmetros do Algoritmo Genético
    POP_SIZE = 300
    CXPB = 0.95
    MUTPB = 0.7
    NGEN = 1500

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()
    
    stats_aep = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_cost = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats = tools.MultiStatistics(aep=stats_aep, cost=stats_cost)
    stats.register("avg", np.mean); stats.register("std", np.std)
    stats.register("min", np.min); stats.register("max", np.max)

    # 1. Avalia a população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)

    # 2. Loop principal de gerações
    for gen in range(1, NGEN + 1):
        # Seleciona os pais usando a seleção NSGA-II
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Aplica Crossover e Mutação de forma independente
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Avalia os novos indivíduos (que não têm fitness válida)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # A nova população é selecionada a partir da união de pais e filhos
        pop = toolbox.select(pop + offspring, POP_SIZE)
        
        # Atualiza o Hall of Fame e imprime as estatísticas
        hof.update(pop)
        record = stats.compile(pop)
        print(f"Gen {gen}: AEP Max={record['aep']['max']:.2f}, Cost Min={record['cost']['min']:.2f}")
    
    pool.close()
    pool.join()

    # --- Pós-Processamento e Análise dos Resultados ---
    print(f"\n--- Otimização concluída. Encontradas {len(hof)} soluções na Frente de Pareto. ---")
    
    output_dir = "pareto_front_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, individual in enumerate(hof):
        aep_liq, cost = individual.fitness.values
        coords = np.array(individual).reshape((IND_SIZE, 2))
        filename = os.path.join(output_dir, f"solution_{i+1}_coords.txt")
        
        with open(filename, 'w') as f:
            x_str = ", ".join([f"{val:.4f}" for val in coords[:, 0]])
            y_str = ", ".join([f"{val:.4f}" for val in coords[:, 1]])
            f.write(f"xc: [{x_str}]\n"); f.write(f"yc: [{y_str}]\n")
        
        results.append({'Solution': i+1, 'AEP_Liquido_MWh': aep_liq, 'Custo_USD': cost, 'File': filename})

    df_pareto = pd.DataFrame(results)
    df_pareto_sorted = df_pareto.sort_values(by='AEP_Liquido_MWh', ascending=False)
    csv_path = os.path.join(output_dir, "pareto_summary.csv")
    df_pareto_sorted.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"Resumo da Frente de Pareto salvo em: {csv_path}")

    plt.figure(figsize=(10, 8))
    plt.scatter(df_pareto_sorted['Custo_USD'] / 1e6, df_pareto_sorted['AEP_Liquido_MWh'], c='blue', alpha=0.7)
    plt.title('Frente de Pareto: AEP Líquido vs. Custo do Cabeamento', fontsize=16)
    plt.xlabel('Custo Total do Cabeamento (Milhões de USD)', fontsize=12)
    plt.ylabel('AEP Líquido (MWh/ano)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path = os.path.join(output_dir, "pareto_front_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico da Frente de Pareto salvo em: {plot_path}")

    total_time = time.time() - start_time
    print(f"Tempo total de computação: {total_time/60:.2f} minutos")

if __name__ == "__main__":
    main()