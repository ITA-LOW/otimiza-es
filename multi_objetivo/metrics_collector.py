"""
MÓDULO DE COLETA DE MÉTRICAS
============================
Coleta e salva métricas em 3 arquivos CSV conforme plano estruturado:
1. summary_results.csv - Uma linha por execução/método (melhor solução)
2. all_pareto_fronts.csv - Múltiplas linhas por execução (todas soluções Pareto)
3. convergence_history.csv - Uma linha a cada X gerações (dinâmica do algoritmo)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# =============================================================================
# FUNÇÕES AUXILIARES PARA CÁLCULO DE MÉTRICAS
# =============================================================================

def calculate_wake_loss_percentage(gross_aep_mwh, net_aep_mwh, electrical_loss_mwh):
    """
    Calcula percentual de perda devido ao efeito esteira.
    
    Wake loss = Gross AEP - Net AEP - Electrical Loss
    Wake loss % = (Wake loss / Gross AEP) * 100
    
    Nota: O Gross AEP já inclui wake loss (calculado pelo modelo GaussianWake).
    Para calcular wake loss isoladamente, precisaríamos do AEP teórico sem wake.
    Por enquanto, assumimos que a diferença entre Gross e Net é apenas elétrica.
    """
    if gross_aep_mwh <= 0:
        return 0.0
    
    # Wake loss é a diferença entre o AEP teórico (sem wake) e o Gross AEP
    # Como não temos o AEP teórico, usamos uma aproximação:
    # Wake loss ≈ (Gross AEP - Net AEP) - Electrical Loss
    # Mas na verdade, o Gross AEP já tem wake embutido, então:
    # Wake loss % ≈ 0 (já está no Gross AEP)
    # Para uma estimativa melhor, precisaríamos calcular AEP sem wake
    
    # Por enquanto, retornamos 0 e deixamos para calcular depois se necessário
    # ou podemos usar uma estimativa baseada em literatura
    return 0.0  # TODO: Implementar cálculo real de wake loss se necessário

def calculate_lcoe_proxy(total_cost_usd, net_aep_gwh):
    """
    Calcula proxy de LCOE (Levelized Cost of Energy).
    
    LCOE Proxy = Total Cost / Net AEP
    
    Args:
        total_cost_usd: Custo total em USD
        net_aep_gwh: Energia anual líquida em GWh
    
    Returns:
        lcoe_proxy: Razão custo por energia (USD/MWh)
    """
    if net_aep_gwh <= 0:
        return float('inf')
    return total_cost_usd / (net_aep_gwh * 1000.0)  # Converte GWh para MWh

def calculate_substation_eccentricity(turb_coords, sub_pos):
    """
    Calcula excentricidade da subestação (distância ao centroide das turbinas).
    
    Args:
        turb_coords: Array de coordenadas das turbinas (N, 2)
        sub_pos: Posição da subestação (2,)
    
    Returns:
        eccentricity_m: Distância em metros
    """
    centroid = np.mean(turb_coords, axis=0)
    eccentricity = np.linalg.norm(sub_pos - centroid)
    return eccentricity

def calculate_cable_strings_stats(plant):
    """
    Calcula estatísticas dos ramais de cabos.
    
    Args:
        plant: Objeto Plant com paths
    
    Returns:
        dict: Estatísticas dos ramais
    """
    if not hasattr(plant, 'paths') or len(plant.paths) == 0:
        return {
            'num_strings': 0,
            'avg_turbines_per_string': 0.0,
            'std_turbines_per_string': 0.0
        }
    
    # Conta turbinas por string (excluindo subestação que é o último elemento)
    turbines_per_string = [len(path) - 1 for path in plant.paths if len(path) > 1]
    
    if len(turbines_per_string) == 0:
        return {
            'num_strings': 0,
            'avg_turbines_per_string': 0.0,
            'std_turbines_per_string': 0.0
        }
    
    return {
        'num_strings': len(turbines_per_string),
        'avg_turbines_per_string': np.mean(turbines_per_string),
        'std_turbines_per_string': np.std(turbines_per_string) if len(turbines_per_string) > 1 else 0.0
    }

def get_cable_sections_used(plant):
    """
    Extrai tipos de cabos utilizados.
    
    Args:
        plant: Objeto Plant com cables
    
    Returns:
        str: Descrição dos tipos de cabos, ex: "3x95mm, 5x150mm"
    """
    if not hasattr(plant, 'cables_flat') or len(plant.cables_flat) == 0:
        return ""
    
    # Extrai seções únicas
    sections = {}
    for cable in plant.cables_flat:
        if hasattr(cable, 'section_mm2'):
            sec = cable.section_mm2
            sections[sec] = sections.get(sec, 0) + 1
    
    # Formata string
    if len(sections) == 0:
        return ""
    
    sections_str = ", ".join([f"{count}x{int(sec)}mm²" for sec, count in sorted(sections.items())])
    return sections_str

def calculate_population_diversity(population, objective_idx=0):
    """
    Calcula diversidade da população em um objetivo específico.
    
    Args:
        population: Lista de indivíduos com fitness
        objective_idx: Índice do objetivo (0=AEP, 1=Cost)
    
    Returns:
        std_dev: Desvio padrão dos valores do objetivo
    """
    if len(population) == 0:
        return 0.0
    
    values = []
    for ind in population:
        if hasattr(ind, 'fitness') and ind.fitness.valid:
            if len(ind.fitness.values) > objective_idx:
                values.append(ind.fitness.values[objective_idx])
    
    if len(values) == 0:
        return 0.0
    
    return np.std(values)

def calculate_feasibility_rate(population):
    """
    Calcula taxa de viabilidade da população (percentual sem cruzamentos).
    
    Args:
        population: Lista de indivíduos
    
    Returns:
        feasibility_rate: Percentual de indivíduos válidos (0-100)
    """
    if len(population) == 0:
        return 0.0
    
    valid_count = sum(1 for ind in population 
                     if hasattr(ind, 'fitness') and ind.fitness.valid 
                     and ind.fitness.values[0] > 0)  # AEP > 0
    
    return (valid_count / len(population)) * 100.0

# =============================================================================
# FUNÇÕES DE COLETA DE MÉTRICAS PARA OS 3 CSVs
# =============================================================================

def collect_summary_metrics(individual, method_name, run_id, seed, turbine_count,
                           is_sequential=False, turbine_layout=None,
                           time_total_s=0.0, time_phase1_s=0.0, time_phase2_s=0.0,
                           pareto_front=None, final_hypervolume=0.0,
                           pareto_spread=0.0, num_solutions_pareto=0,
                           feasibility_rate_final=0.0,
                           calc_aep_func=None, cabling_func=None,
                           wind_data=None, turb_data=None):
    """
    Coleta métricas para summary_results.csv (melhor solução/knee point).
    
    Args:
        individual: Indivíduo Baseline/Proposed (35 genes) ou Sequential (3 genes)
        method_name: 'Baseline', 'Proposed', ou 'Sequential'
        run_id: Número da execução (1 a N_RUNS)
        seed: Semente aleatória utilizada
        turbine_count: Escala do problema (16, 36, ou 64)
        is_sequential: Se True, individual é da Fase 2 Sequential
        turbine_layout: Layout de turbinas para Sequential
        time_total_s: Tempo total de execução (segundos)
        time_phase1_s: Tempo Fase 1 (segundos)
        time_phase2_s: Tempo Fase 2 (segundos)
        pareto_front: Frente de Pareto (para calcular métricas)
        final_hypervolume: Hipervolume final
        pareto_spread: Spread da frente de Pareto
        num_solutions_pareto: Número de soluções não-dominadas
        feasibility_rate_final: Taxa de viabilidade final (%)
        calc_aep_func: Função para calcular AEP
        cabling_func: Função para calcular cabeamento
        wind_data: Dados de vento (wind_dir, wind_freq, wind_speed)
        turb_data: Dados das turbinas (turb_ci, turb_co, rated_ws, rated_pwr, turb_diam)
    
    Returns:
        dict: Dicionário com todas as métricas para summary_results.csv
    """
    # Extrai coordenadas e parâmetros do indivíduo
    if is_sequential:
        n_grupos_norm = individual[0]
        sub_pos = np.array([individual[1], individual[2]])
        turb_coords = np.array(turbine_layout).reshape((turbine_count, 2))
    else:
        n_coords = turbine_count * 2
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        turb_coords = np.array(coords_flat).reshape((turbine_count, 2))
    
    # Converte número de grupos
    MIN_GRUPOS = 5
    MAX_GRUPOS = 64
    n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
    n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
    n_grupos = min(n_grupos, turbine_count)
    
    # Calcula cabeamento
    coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
    substation_idx = turbine_count
    
    try:
        plant, res = cabling_func(coords_all, sub=substation_idx, n_grupos=n_grupos)
    except Exception as e:
        # Em caso de erro, retorna métricas com valores padrão
        return {
            'Run_ID': run_id,
            'Seed': seed,
            'Method': method_name,
            'Turbine_Count': turbine_count,
            'Net_AEP_GWh': 0.0,
            'Total_Cost_USD': float('inf'),
            'LCOE_Proxy_USD_MWh': float('inf'),
            # ... outros campos com valores padrão
        }
    
    # Calcula AEP
    wind_dir, wind_freq, wind_speed = wind_data
    turb_diam = turb_data[4]
    aep_array = calc_aep_func(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                              turb_data[0], turb_data[1],
                              turb_data[2], turb_data[3])
    gross_aep_mwh = np.sum(aep_array)
    
    electrical_loss_mwh = res['perda_anual_mwh']
    net_aep_mwh = gross_aep_mwh - electrical_loss_mwh
    net_aep_gwh = net_aep_mwh / 1000.0
    gross_aep_gwh = gross_aep_mwh / 1000.0
    
    total_cost_usd = res['custo_total_usd']
    lcoe_proxy = calculate_lcoe_proxy(total_cost_usd, net_aep_gwh)
    
    # Calcula wake loss (aproximação)
    wake_loss_percentage = calculate_wake_loss_percentage(gross_aep_mwh, net_aep_mwh, electrical_loss_mwh)
    electrical_loss_percentage = (electrical_loss_mwh / gross_aep_mwh * 100.0) if gross_aep_mwh > 0 else 0.0
    
    # Calcula excentricidade
    substation_eccentricity = calculate_substation_eccentricity(turb_coords, sub_pos)
    
    # Estatísticas dos ramais
    string_stats = calculate_cable_strings_stats(plant)
    
    # Tipos de cabos
    cable_sections = get_cable_sections_used(plant)
    
    # Monta dicionário de métricas
    metrics = {
        # Identificação
        'Run_ID': run_id,
        'Seed': seed,
        'Method': method_name,
        'Turbine_Count': turbine_count,
        
        # Objetivos finais
        'Net_AEP_GWh': net_aep_gwh,
        'Total_Cost_USD': total_cost_usd,
        'LCOE_Proxy_USD_MWh': lcoe_proxy,
        
        # Breakdown de física e engenharia
        'Gross_AEP_GWh': gross_aep_gwh,
        'Wake_Loss_Percentage': wake_loss_percentage,
        'Electrical_Loss_Percentage': electrical_loss_percentage,
        'Electrical_Loss_MWh': electrical_loss_mwh,
        'Total_Cable_Length_km': res['comprimento_total_m'] / 1000.0,
        'Substation_Eccentricity_m': substation_eccentricity,
        
        # Topologia da rede elétrica
        'Num_Cable_Strings': string_stats['num_strings'],
        'Avg_Turbines_Per_String': string_stats['avg_turbines_per_string'],
        'Std_Turbines_Per_String': string_stats['std_turbines_per_string'],
        'Cable_Sections_Used': cable_sections,
        
        # Performance computacional e algorítmica
        'Time_Total_s': time_total_s,
        'Time_Phase1_s': time_phase1_s,
        'Time_Phase2_s': time_phase2_s,
        'Final_Hypervolume': final_hypervolume,
        'Pareto_Spread': pareto_spread,
        'Num_Solutions_Pareto': num_solutions_pareto,
        'Feasibility_Rate_Final': feasibility_rate_final,
    }
    
    return metrics

def collect_pareto_front_metrics(pareto_front, run_id, method_name,
                                calc_aep_func=None, cabling_func=None,
                                wind_data=None, turb_data=None, turbine_count=36):
    """
    Coleta métricas para all_pareto_fronts.csv (todas soluções Pareto).
    
    Args:
        pareto_front: Lista de soluções não-dominadas
        run_id: Número da execução
        method_name: 'Baseline', 'Proposed', ou 'Sequential'
        calc_aep_func: Função para calcular AEP
        cabling_func: Função para calcular cabeamento
        wind_data: Dados de vento
        turb_data: Dados das turbinas
        turbine_count: Número de turbinas
    
    Returns:
        list: Lista de dicionários, um para cada solução Pareto
    """
    metrics_list = []
    
    MIN_GRUPOS = 5
    MAX_GRUPOS = 64
    
    for solution_id, individual in enumerate(pareto_front):
        if not hasattr(individual, 'fitness') or not individual.fitness.valid:
            continue
        
        # Extrai coordenadas
        n_coords = turbine_count * 2
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        turb_coords = np.array(coords_flat).reshape((turbine_count, 2))
        
        # Converte número de grupos
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        n_grupos = min(n_grupos, turbine_count)
        
        # Calcula cabeamento
        coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
        substation_idx = turbine_count
        
        try:
            plant, res = cabling_func(coords_all, sub=substation_idx, n_grupos=n_grupos)
        except Exception:
            continue
        
        # Calcula AEP
        wind_dir, wind_freq, wind_speed = wind_data
        turb_diam = turb_data[4]
        aep_array = calc_aep_func(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                 turb_data[0], turb_data[1],
                                 turb_data[2], turb_data[3])
        gross_aep_mwh = np.sum(aep_array)
        
        electrical_loss_mwh = res['perda_anual_mwh']
        net_aep_mwh = gross_aep_mwh - electrical_loss_mwh
        net_aep_gwh = net_aep_mwh / 1000.0
        
        wake_loss_percentage = calculate_wake_loss_percentage(gross_aep_mwh, net_aep_mwh, electrical_loss_mwh)
        electrical_loss_percentage = (electrical_loss_mwh / gross_aep_mwh * 100.0) if gross_aep_mwh > 0 else 0.0
        
        metrics = {
            'Run_ID': run_id,
            'Method': method_name,
            'Solution_ID': solution_id,
            'Net_AEP_GWh': net_aep_gwh,
            'Total_Cost_USD': res['custo_total_usd'],
            'Wake_Loss_Percentage': wake_loss_percentage,
            'Electrical_Loss_Percentage': electrical_loss_percentage,
            'Num_Strings': len(plant.paths),
        }
        
        metrics_list.append(metrics)
    
    return metrics_list

def collect_convergence_metrics(run_id, method_name, generation, population,
                               pareto_front=None, ref_point=None,
                               hypervolume_func=None):
    """
    Coleta métricas para convergence_history.csv (dinâmica do algoritmo).
    
    Args:
        run_id: Número da execução
        method_name: 'Baseline', 'Proposed', ou 'Sequential'
        generation: Número da geração atual
        population: População atual
        pareto_front: Frente de Pareto atual (opcional)
        ref_point: Ponto de referência para hipervolume (opcional)
        hypervolume_func: Função para calcular hipervolume
    
    Returns:
        dict: Dicionário com métricas de convergência
    """
    # Calcula hipervolume
    hypervolume = 0.0
    if pareto_front is not None and len(pareto_front) > 0 and hypervolume_func is not None:
        hypervolume = hypervolume_func(pareto_front, ref_point)
    
    # Calcula taxa de viabilidade
    feasibility_rate = calculate_feasibility_rate(population)
    
    # Calcula diversidade
    pop_diversity_aep = calculate_population_diversity(population, objective_idx=0)
    pop_diversity_cost = calculate_population_diversity(population, objective_idx=1)
    
    metrics = {
        'Run_ID': run_id,
        'Method': method_name,
        'Generation': generation,
        'Hypervolume': hypervolume,
        'Pop_Feasibility_Rate': feasibility_rate,
        'Pop_Diversity_AEP': pop_diversity_aep,
        'Pop_Diversity_Cost': pop_diversity_cost,
    }
    
    return metrics

# =============================================================================
# FUNÇÕES DE SALVAMENTO EM CSV
# =============================================================================

def save_summary_results(metrics_dict, output_dir='.', append=True):
    """
    Salva métricas em summary_results.csv.
    
    Args:
        metrics_dict: Dicionário com métricas (ou lista de dicionários)
        output_dir: Diretório de saída
        append: Se True, adiciona ao arquivo existente
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'summary_results.csv')
    
    # Converte para DataFrame
    if isinstance(metrics_dict, dict):
        df = pd.DataFrame([metrics_dict])
    else:
        df = pd.DataFrame(metrics_dict)
    
    # Salva
    if append and os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False, float_format='%.6f')
    else:
        df.to_csv(filepath, mode='w', header=True, index=False, float_format='%.6f')
    
    print(f"✓ Métricas de resumo salvas em: {filepath}")

def save_pareto_fronts(metrics_list, output_dir='.', append=True):
    """
    Salva métricas em all_pareto_fronts.csv.
    
    Args:
        metrics_list: Lista de dicionários (uma por solução Pareto)
        output_dir: Diretório de saída
        append: Se True, adiciona ao arquivo existente
    """
    if len(metrics_list) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'all_pareto_fronts.csv')
    
    df = pd.DataFrame(metrics_list)
    
    if append and os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False, float_format='%.6f')
    else:
        df.to_csv(filepath, mode='w', header=True, index=False, float_format='%.6f')
    
    print(f"✓ Métricas de frente de Pareto salvas em: {filepath} ({len(metrics_list)} soluções)")

def save_convergence_history(metrics_dict, output_dir='.', append=True):
    """
    Salva métricas em convergence_history.csv.
    
    Args:
        metrics_dict: Dicionário com métricas (ou lista de dicionários)
        output_dir: Diretório de saída
        append: Se True, adiciona ao arquivo existente
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'convergence_history.csv')
    
    if isinstance(metrics_dict, dict):
        df = pd.DataFrame([metrics_dict])
    else:
        df = pd.DataFrame(metrics_dict)
    
    if append and os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False, float_format='%.6f')
    else:
        df.to_csv(filepath, mode='w', header=True, index=False, float_format='%.6f')
    
    print(f"✓ Métricas de convergência salvas em: {filepath}")

def collect_representative_solution_coords(individual, method_name, run_id, seed, turbine_count,
                                          is_sequential=False, turbine_layout=None,
                                          net_aep_gwh=0.0, total_cost_usd=0.0):
    """
    Coleta coordenadas de uma solução representativa (knee point) para plotagem de layouts.
    
    Args:
        individual: Indivíduo Baseline/Proposed (35 genes) ou Sequential (3 genes)
        method_name: 'Baseline', 'Proposed', ou 'Sequential'
        run_id: Número da execução
        seed: Semente aleatória
        turbine_count: Número de turbinas
        is_sequential: Se True, individual é da Fase 2 Sequential
        turbine_layout: Layout de turbinas para Sequential
        net_aep_gwh: AEP líquido em GWh (para identificação)
        total_cost_usd: Custo total em USD (para identificação)
    
    Returns:
        dict: Dicionário com coordenadas e metadados
    """
    # Extrai coordenadas
    if is_sequential:
        n_grupos_norm = individual[0]
        sub_pos = np.array([individual[1], individual[2]])
        turb_coords = np.array(turbine_layout).reshape((turbine_count, 2))
    else:
        n_coords = turbine_count * 2
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        turb_coords = np.array(coords_flat).reshape((turbine_count, 2))
    
    # Converte número de grupos
    MIN_GRUPOS = 5
    MAX_GRUPOS = 64
    n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
    n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
    n_grupos = min(n_grupos, turbine_count)
    
    # Serializa coordenadas das turbinas (lista de floats separados por vírgula)
    turb_x_str = ','.join([f'{x:.12f}' for x in turb_coords[:, 0]])
    turb_y_str = ','.join([f'{y:.12f}' for y in turb_coords[:, 1]])
    
    return {
        'Run_ID': run_id,
        'Seed': seed,
        'Method': method_name,
        'Turbine_Count': turbine_count,
        'Net_AEP_GWh': net_aep_gwh,
        'Total_Cost_USD': total_cost_usd,
        'Turbine_Coords_X': turb_x_str,
        'Turbine_Coords_Y': turb_y_str,
        'Substation_X': sub_pos[0],
        'Substation_Y': sub_pos[1],
        'N_Grupos': n_grupos
    }

def save_representative_solutions(metrics_dict, output_dir='.', append=True):
    """
    Salva coordenadas de soluções representativas em representative_solutions.csv.
    
    Args:
        metrics_dict: Dicionário com coordenadas (ou lista de dicionários)
        output_dir: Diretório de saída
        append: Se True, adiciona ao arquivo existente
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'representative_solutions.csv')
    
    # Converte para DataFrame
    if isinstance(metrics_dict, dict):
        df = pd.DataFrame([metrics_dict])
    else:
        df = pd.DataFrame(metrics_dict)
    
    # Salva
    if append and os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False, float_format='%.12f')
    else:
        df.to_csv(filepath, mode='w', header=True, index=False, float_format='%.12f')
    
    print(f"✓ Coordenadas de soluções representativas salvas em: {filepath}")
