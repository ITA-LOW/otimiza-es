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
import multi_objetivo.cabling_v1 as cabling_v1
import multi_objetivo.cabling_v3 as cabling_v3

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = "config"
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
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

# Função para detectar sobreposição de cabos (segmentos que se cruzam)
def detectar_sobreposicao_cabos(paths, coords):
    """
    Detecta sobreposições entre segmentos de cabos de diferentes paths.
    Retorna uma penalidade proporcional ao número e severidade das sobreposições.
    """
    def segmentos_intersectam(p1, p2, q1, q2):
        """Verifica se dois segmentos de linha se intersectam (algoritmo de orientação)."""
        def orientacao(o, a, b):
            val = (a[1] - o[1]) * (b[0] - a[0]) - (a[0] - o[0]) * (b[1] - a[1])
            if abs(val) < 1e-9:
                return 0  # Colinear
            return 1 if val > 0 else 2  # Horário ou anti-horário
        
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
    # Compara segmentos de paths diferentes
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_i = paths[i]
            path_j = paths[j]
            
            # Compara cada segmento do path_i com cada segmento do path_j
            for k in range(len(path_i) - 1):
                p1 = coords[path_i[k]]
                p2 = coords[path_i[k + 1]]
                
                for l in range(len(path_j) - 1):
                    q1 = coords[path_j[l]]
                    q2 = coords[path_j[l + 1]]
                    
                    # Ignora se compartilham um ponto (não é sobreposição)
                    if (path_i[k] == path_j[l] or path_i[k] == path_j[l + 1] or
                        path_i[k + 1] == path_j[l] or path_i[k + 1] == path_j[l + 1]):
                        continue
                    
                    if segmentos_intersectam(p1, p2, q1, q2):
                        n_overlaps += 1
    
    # Penalidade proporcional ao número de sobreposições
    return n_overlaps * 1e6

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
full_path_wr = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
full_path_turb = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA = getWindRoseYAML(full_path_wr)

# =============================================================================
# FUNÇÃO DE AVALIAÇÃO MULTIOBJETIVO
# =============================================================================

def evaluate_multi_objective(individual):
    """
    Calcula os dois objetivos para um dado layout: AEP Líquido e Custo.
    Aplica penalidades moderadas para violações de restrições, permitindo que
    o algoritmo genético discrimine soluções inválidas durante a seleção.
    """
    try:
        turb_coords = np.array(individual).reshape((IND_SIZE, 2))
        
        # ============================================================
        # CÁLCULO DE PENALIDADES POR VIOLAÇÕES DE RESTRIÇÕES
        # ============================================================
        # Penalidade por turbinas fora do círculo (espaço de busca)
        dist_from_center = np.linalg.norm(turb_coords, axis=1)
        violations_out_of_circle = dist_from_center > CIRCLE_RADIUS
        # Penalidade proporcional à distância além do limite
        penalty_out_of_circle = np.sum(np.maximum(0, dist_from_center - CIRCLE_RADIUS)) * 1e6
        
        # Penalidade por turbinas muito próximas (< 2 diâmetros de rotor)
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(IND_SIZE, k=1)
        close_distances = dist_matrix[i_upper, j_upper]
        violations_close = close_distances < N_DIAMETERS
        # Penalidade proporcional à violação (quanto mais próximo, maior a penalidade)
        penalty_close_turbines = np.sum(np.maximum(0, N_DIAMETERS - close_distances[violations_close])) * 1e6
        
        # Penalidade por sobreposição de cabos (será calculada após obter os paths do cabeamento)
        penalty_cable_overlap = 0.0
        
        # ============================================================
        # CÁLCULO DO AEP BRUTO (considera efeito de esteira)
        # ============================================================
        _, _, _, _, turb_diam = TURB_ATRBT_DATA
        aep_bruto = np.sum(calcAEP(turb_coords, WIND_ROSE_DATA[1], WIND_ROSE_DATA[2], WIND_ROSE_DATA[0], turb_diam, *TURB_ATRBT_DATA[0:2], *TURB_ATRBT_DATA[2:4]))

        # ============================================================
        # CÁLCULO DE CUSTO E PERDAS POR CABEAMENTO
        # ============================================================
        # Encontra a turbina de coleta (mais próxima do continente)
        distancias_ao_continente = np.linalg.norm(turb_coords - SUBSTATION_CONTINENT, axis=1)
        ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
        n_grupos = 4
        
        # Chama o módulo de cabeamento (cabling_v3 usa heurística determinística:
        # agrupamento angular + balanceamento rápido, mais estável que KMeans estocástico)
        planta, resultados_cabeamento = cabling_v3.analisar_layout_completo(turb_coords, sub=ponto_de_coleta_idx, n_grupos=n_grupos)
        
        custo_total = resultados_cabeamento['custo_total_usd']
        perdas_joule_mwh = resultados_cabeamento['perda_anual_mwh']
        
        # Detecta sobreposições de cabos (segmentos que se cruzam)
        penalty_cable_overlap = detectar_sobreposicao_cabos(planta.paths, turb_coords)
        
        # ============================================================
        # APLICAÇÃO DE PENALIDADES NOS OBJETIVOS
        # ============================================================
        # AEP líquido: penaliza violações reduzindo o AEP
        # (soluções com violações terão AEP menor e serão eliminadas na seleção)
        aep_liquido = aep_bruto - perdas_joule_mwh - penalty_out_of_circle - penalty_close_turbines - penalty_cable_overlap
        
        # Custo: penaliza violações aumentando o custo
        # (soluções com violações terão custo maior e serão eliminadas na seleção)
        custo_penalizado = custo_total + penalty_out_of_circle + penalty_close_turbines + penalty_cable_overlap
        
        # ============================================================
        # VALIDAÇÃO FINAL: AEP líquido deve ser positivo
        # ============================================================
        # Se mesmo após penalidades o AEP for negativo, a solução é muito ruim
        # Mas ainda retornamos os valores penalizados para permitir comparação
        # O filtro final no Hall of Fame removerá soluções com AEP <= 0
        
        return aep_liquido, custo_penalizado
        
    except Exception as e:
        # Se ocorrer um erro, retorna uma fitness muito ruim para eliminar o indivíduo
        print(f"Erro na avaliação: {e}. Penalizando indivíduo.")
        return -1e6, 1e12

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
    
    # Filtra soluções inválidas (AEP negativo) da população inicial
    pop = [ind for ind in pop if ind.fitness.values[0] > 0]
    if len(pop) < POP_SIZE:
        print(f"AVISO: Apenas {len(pop)}/{POP_SIZE} soluções válidas na população inicial. Regenerando...")
        # Regenera indivíduos até ter população válida
        while len(pop) < POP_SIZE:
            new_ind = toolbox.individual()
            fit = toolbox.evaluate(new_ind)
            new_ind.fitness.values = fit
            if fit[0] > 0:  # AEP positivo
                pop.append(new_ind)
    
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
        
        # Filtra soluções inválidas do Hall of Fame (AEP negativo)
        hof_valid = [ind for ind in hof if ind.fitness.values[0] > 0]
        hof.clear()
        hof.update(hof_valid)
        
        record = stats.compile(pop)
        n_valid = sum(1 for ind in pop if ind.fitness.values[0] > 0)
        print(f"Gen {gen}: AEP Max={record['aep']['max']:.2f}, Cost Min={record['cost']['min']:.2f}, Valid={n_valid}/{len(pop)}, Pareto={len(hof)}")
    
    pool.close()
    pool.join()

    # --- Pós-Processamento e Análise dos Resultados ---
    # Filtra soluções inválidas uma última vez (AEP negativo)
    hof_final = [ind for ind in hof if ind.fitness.values[0] > 0]
    
    print(f"\n--- Otimização concluída. Encontradas {len(hof_final)} soluções válidas na Frente de Pareto (de {len(hof)} total). ---")
    
    if len(hof_final) == 0:
        print("AVISO: Nenhuma solução válida encontrada! Verifique os parâmetros do algoritmo.")
        return
    
    output_dir = "pareto_front_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, individual in enumerate(hof_final):
        aep_liq, cost = individual.fitness.values
        
        # Validação adicional de segurança
        if aep_liq <= 0:
            print(f"AVISO: Solução {i+1} tem AEP negativo ({aep_liq:.2f}), pulando...")
            continue
            
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