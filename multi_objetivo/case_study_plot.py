"""
CÓDIGO DE PLOTAGEM DOS RESULTADOS DO ESTUDO DE CASO
==================================================
Lê os 3 CSVs gerados pelo case_study_comparison_original.py e gera todos os gráficos:
1. summary_results.csv - Uma linha por execução/método (melhor solução)
2. all_pareto_fronts.csv - Múltiplas linhas por execução (todas soluções Pareto)
3. convergence_history.csv - Uma linha a cada X gerações (dinâmica do algoritmo)

Gera:
- Comparação de frentes de Pareto
- Histórico de hipervolume
- Boxplots comparativos
- Gráficos de soluções individuais (opcional)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

# Configuração de fontes para publicação
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

# Configurações de tamanhos de fonte
FONT_SIZE_BASE = 13
FONT_SIZE_TITLE_MAIN = 22
FONT_SIZE_TITLE_PLOT = 20
FONT_SIZE_TITLE_SUBPLOT = 16
FONT_SIZE_LABEL_AXIS = 14
FONT_SIZE_LABEL_AXIS_SMALL = 13
FONT_SIZE_LEGEND = 13
FONT_SIZE_LEGEND_SMALL = 12
FONT_SIZE_TICK = 14
TURBINE_MARKER_SIZE = 40
TURBINE_EDGE_WIDTH = 1.0
SUBSTATION_MARKER_SIZE = 200
CABLE_LINEWIDTH = 1.0

# Importa módulos do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import multi_objetivo.cabling_v3 as cabling_v3

# Constantes para plotagem de layouts
CIRCLE_RADIUS = 5000  # Raio do círculo de restrição (metros)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calculate_c_metric(pareto_front_a, pareto_front_b):
    """
    Calcula C-metric: porcentagem de soluções de B dominadas por pelo menos uma solução de A.
    
    C(A, B) = |{b in B: existe a in A tal que a domina b}| / |B|
    
    Args:
        pareto_front_a: DataFrame com soluções do método A (Net_AEP_GWh, Total_Cost_USD)
        pareto_front_b: DataFrame com soluções do método B
    
    Returns:
        c_metric: Valor entre 0 e 1 (0 = nenhuma solução dominada, 1 = todas dominadas)
    """
    if len(pareto_front_b) == 0:
        return 0.0
    if len(pareto_front_a) == 0:
        return 0.0
    
    # Extrai AEP e custo
    aeps_a = pareto_front_a['Net_AEP_GWh'].values
    costs_a = pareto_front_a['Total_Cost_USD'].values
    
    # Conta quantas soluções de B são dominadas por pelo menos uma solução de A
    dominated_count = 0
    for _, row_b in pareto_front_b.iterrows():
        aep_b = row_b['Net_AEP_GWh']
        cost_b = row_b['Total_Cost_USD']
        
        # Verifica se existe alguma solução em A que domina esta solução de B
        is_dominated = False
        for i in range(len(pareto_front_a)):
            aep_a = aeps_a[i]
            cost_a = costs_a[i]
            
            # A domina B se: AEP_A >= AEP_B AND Cost_A <= Cost_B, com pelo menos uma desigualdade estrita
            if (aep_a >= aep_b and cost_a <= cost_b) and (aep_a > aep_b or cost_a < cost_b):
                is_dominated = True
                break
        
        if is_dominated:
            dominated_count += 1
    
    return dominated_count / len(pareto_front_b)

def find_knee_point_from_df(df_pareto):
    """
    Seleciona a melhor solução usando o método do knee point (ponto de joelho).
    
    Args:
        df_pareto: DataFrame com soluções Pareto (Net_AEP_GWh, Total_Cost_USD)
    
    Returns:
        best_row: Linha do DataFrame com melhor trade-off (knee point)
    """
    if len(df_pareto) == 0:
        return None
    
    if len(df_pareto) == 1:
        return df_pareto.iloc[0]
    
    # Extrai AEP e custo
    aeps = df_pareto['Net_AEP_GWh'].values
    costs = df_pareto['Total_Cost_USD'].values
    
    # Normaliza AEP e custo para [0, 1]
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
    ideal_point = np.array([1.0, 1.0])
    distances = []
    for i in range(len(df_pareto)):
        point = np.array([aep_norm[i], cost_norm[i]])
        dist = np.linalg.norm(ideal_point - point)
        distances.append(dist)
    
    # Retorna a solução com menor distância ao ideal (knee point)
    knee_idx = np.argmin(distances)
    return df_pareto.iloc[knee_idx]

# =============================================================================
# FUNÇÕES DE PLOTAGEM
# =============================================================================

def plot_pareto_fronts_comparison(df_summary, df_pareto, output_dir='.'):
    """
    Plota frentes de Pareto comparativas dos 3 métodos.
    
    Args:
        df_summary: DataFrame com summary_results.csv
        df_pareto: DataFrame com all_pareto_fronts.csv
        output_dir: Diretório de saída
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filtra dados por método
    methods = ['Baseline', 'Proposed', 'Sequential']
    colors = {'Baseline': 'blue', 'Proposed': 'green', 'Sequential': 'red'}
    
    # Para cada método, plota frente de Pareto
    for method in methods:
        if method == 'Sequential':
            # Sequential: apenas um ponto (melhor solução)
            df_method = df_summary[df_summary['Method'] == method]
            if len(df_method) > 0:
                # Usa a última execução como exemplo
                last_run = df_method['Run_ID'].max()
                df_last = df_method[df_method['Run_ID'] == last_run]
                if len(df_last) > 0:
                    row = df_last.iloc[0]
                    ax.scatter(row['Total_Cost_USD'] / 1000.0, row['Net_AEP_GWh'],
                              s=100, alpha=1.0, color=colors[method],
                              edgecolors='dark' + colors[method], linewidths=2.0,
                              label=f'{method} (Best Cost)', zorder=5, marker='s')
        else:
            # Baseline e Proposed: frente de Pareto completa
            df_method_pareto = df_pareto[df_pareto['Method'] == method]
            if len(df_method_pareto) > 0:
                # Usa a última execução como exemplo
                last_run = df_method_pareto['Run_ID'].max()
                df_last = df_method_pareto[df_method_pareto['Run_ID'] == last_run]
                
                if len(df_last) > 0:
                    costs = df_last['Total_Cost_USD'].values / 1000.0  # kUSD
                    aeps = df_last['Net_AEP_GWh'].values
                    
                    # Plota todos os pontos (transparentes)
                    ax.scatter(costs, aeps, s=40, alpha=0.4, color=colors[method],
                              edgecolors='dark' + colors[method], linewidths=0.5,
                              label=f'{method} Pareto Front (n={len(df_last)})', zorder=2)
                    
                    # Plota knee point (sólido)
                    knee = find_knee_point_from_df(df_last)
                    if knee is not None:
                        ax.scatter(knee['Total_Cost_USD'] / 1000.0, knee['Net_AEP_GWh'],
                                  s=100, alpha=1.0, color=colors[method],
                                  edgecolors='dark' + colors[method], linewidths=2.0,
                                  label=f'{method} Knee Point', zorder=4, marker='o')
    
    # Configurações do gráfico
    ax.set_xlabel('Total Cabling Cost (Thousands USD)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_ylabel('Net AEP (GWh)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_title('Pareto Fronts Comparison', fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(frameon=True, loc='lower right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    
    # Melhora a aparência geral
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'pareto_fronts_comparison.png')
    output_path_pdf = os.path.join(output_dir, 'pareto_fronts_comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comparação de Frentes de Pareto salva em: {output_path_png} e {output_path_pdf}")
    plt.close()

def plot_hypervolume_history(df_convergence, output_dir='.'):
    """
    Plota a evolução do Hipervolume ao longo das gerações.
    
    Args:
        df_convergence: DataFrame com convergence_history.csv
        output_dir: Diretório de saída
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['Baseline', 'Proposed']
    colors = {'Baseline': 'blue', 'Proposed': 'green'}
    markers = {'Baseline': 'o', 'Proposed': 's'}
    
    for method in methods:
        df_method = df_convergence[df_convergence['Method'] == method]
        if len(df_method) == 0:
            continue
        
        # Agrupa por geração e calcula média e desvio padrão
        grouped = df_method.groupby('Generation')['Hypervolume'].agg(['mean', 'std']).reset_index()
        
        gens = grouped['Generation'].values
        mean_hv = grouped['mean'].values
        std_hv = grouped['std'].values
        
        # Plota linha média
        ax.plot(gens, mean_hv, color=colors[method], linewidth=2.5,
               label=f'{method} (Mean)', marker=markers[method], markersize=6)
        
        # Plota banda de desvio padrão
        ax.fill_between(gens,
                        mean_hv - std_hv,
                        mean_hv + std_hv,
                        alpha=0.3, color=colors[method],
                        label=f'{method} (±1 Std Dev)')
    
    # Configurações do gráfico
    ax.set_xlabel('Generation', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_title('Hypervolume Evolution: Mean ± Standard Deviation',
                fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(frameon=True, loc='best', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    
    # Melhora a aparência geral
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'hypervolume_history.png')
    output_path_pdf = os.path.join(output_dir, 'hypervolume_history.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Histórico de Hipervolume salvo em: {output_path_png} e {output_path_pdf}")
    plt.close()

def plot_boxplots_comparison(df_summary, output_dir='.'):
    """
    Plota boxplots comparativos das métricas principais.
    
    Args:
        df_summary: DataFrame com summary_results.csv
        output_dir: Diretório de saída
    """
    if len(df_summary) == 0:
        print("AVISO: Nenhum dado disponível para boxplots")
        return
    
    # Prepara dados para boxplots
    methods = sorted(df_summary['Method'].unique())
    colors = ['#2E86AB', '#E63946', '#06A77D']  # Azul, Vermelho, Verde
    
    # Métricas para plotar
    metrics = {
        'Final_Hypervolume': {'data': [], 'ylabel': 'Hypervolume (Higher is Better)', 'title': 'Hypervolume Comparison'},
        'Pareto_Spread': {'data': [], 'ylabel': 'Spread (Lower is Better)', 'title': 'Solution Diversity (Spread)'},
        'Num_Solutions_Pareto': {'data': [], 'ylabel': 'Number of Pareto Solutions', 'title': 'Pareto Front Size'},
        'Time_Total_s': {'data': [], 'ylabel': 'Execution Time (seconds)', 'title': 'Computational Efficiency'}
    }
    
    # Agrupa dados por método
    for method in methods:
        df_method = df_summary[df_summary['Method'] == method]
        
        for metric_name in metrics.keys():
            if metric_name in df_method.columns:
                values = df_method[metric_name].dropna().values
                metrics[metric_name]['data'].append(values if len(values) > 0 else [0])
            else:
                metrics[metric_name]['data'].append([0])
    
    # Cria boxplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (metric_name, metric_info) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        data = metric_info['data']
        
        if len(data) > 0 and any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.6)
            
            # Colore as caixas
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Estiliza outros elementos
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.2)
            
            ax.set_ylabel(metric_info['ylabel'], fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
            ax.set_title(metric_info['title'], fontsize=FONT_SIZE_TITLE_SUBPLOT, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.suptitle('Case Study: Comparative Metrics', fontsize=FONT_SIZE_TITLE_MAIN, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'case_study_boxplots.png')
    output_path_pdf = os.path.join(output_dir, 'case_study_boxplots.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Boxplots salvos em: {output_path_png} e {output_path_pdf}")
    plt.close()
    
    # Boxplot dedicado de Hypervolume
    if 'Final_Hypervolume' in df_summary.columns:
        plt.figure(figsize=(10, 6))
        hv_data = []
        for method in methods:
            df_method = df_summary[df_summary['Method'] == method]
            values = df_method['Final_Hypervolume'].dropna().values
            hv_data.append(values if len(values) > 0 else [0])
        
        if len(hv_data) > 0:
            bp_hv = plt.boxplot(hv_data, tick_labels=methods, patch_artist=True, widths=0.6)
            
            colors_hv = ['#2E86AB', '#E63946', '#06A77D']
            for patch, color in zip(bp_hv['boxes'], colors_hv[:len(bp_hv['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp_hv[element], color='black', linewidth=1.2)
            
            plt.ylabel('Hypervolume (Higher is Better)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
            plt.title('Hypervolume Comparison - Case Study', fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            ax_hv = plt.gca()
            ax_hv.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
            ax_hv.spines['top'].set_visible(False)
            ax_hv.spines['right'].set_visible(False)
            
            plt.tight_layout()
            output_path_png = os.path.join(output_dir, 'case_study_hypervolume.png')
            output_path_pdf = os.path.join(output_dir, 'case_study_hypervolume.pdf')
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
            print(f"✓ Boxplot de Hypervolume salvo em: {output_path_png} e {output_path_pdf}")
            plt.close()

def plot_execution_time_comparison(df_summary, output_dir='.'):
    """
    Plota gráfico de barras comparando tempo de execução.
    
    Args:
        df_summary: DataFrame com summary_results.csv
        output_dir: Diretório de saída
    """
    if len(df_summary) == 0:
        print("AVISO: Nenhum dado disponível para gráfico de tempo de execução")
        return
    
    # Agrupa por Turbine_Count e Method
    turbine_counts = sorted(df_summary['Turbine_Count'].unique())
    methods = ['Baseline', 'Proposed', 'Sequential']
    colors = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(turbine_counts))
    bar_width = 0.28
    spacing = 0.05
    
    for i, method in enumerate(methods):
        times = []
        stds = []
        
        for tc in turbine_counts:
            df_subset = df_summary[(df_summary['Method'] == method) & (df_summary['Turbine_Count'] == tc)]
            if len(df_subset) > 0:
                time_vals = df_subset['Time_Total_s'].values / 60.0  # Converte para minutos
                times.append(np.mean(time_vals))
                stds.append(np.std(time_vals))
            else:
                times.append(0)
                stds.append(0)
        
        ax.bar(x + i * (bar_width + spacing) - (bar_width + spacing), times, bar_width,
               label=method, color=colors[method], yerr=stds, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Number of Turbines', fontsize=24, fontweight='bold')
    ax.set_ylabel('Average Execution Time (minutes)', fontsize=24, fontweight='bold')
    ax.set_title('Computational Efficiency Across Problem Scales', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{tc} Turbines' for tc in turbine_counts], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=18, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'execution_time_comparison.png')
    output_path_pdf = os.path.join(output_dir, 'execution_time_comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico de tempo de execução salvo em: {output_path_png} e {output_path_pdf}")
    plt.close()

def plot_aggregated_pareto_fronts(df_pareto, df_summary, output_dir='.'):
    """
    Plota frentes de Pareto agregadas (acumula todas as execuções).
    Calcula e destaca o knee point do aggregated pareto front para cada método.
    Similar ao pareto_compacto das imagens de referência.
    
    Args:
        df_pareto: DataFrame com all_pareto_fronts.csv
        df_summary: DataFrame com summary_results.csv (para Sequential)
        output_dir: Diretório de saída
    
    Returns:
        dict: Dicionário com knee points do aggregated pareto front por método
              {'Baseline': row, 'Proposed': row, 'Sequential': row}
    """
    if len(df_pareto) == 0:
        print("AVISO: Nenhum dado disponível para aggregated Pareto fronts")
        return {}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['Baseline', 'Proposed', 'Sequential']
    colors = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}
    markers = {'Baseline': 'o', 'Proposed': 's', 'Sequential': '^'}
    knee_points = {}
    
    # Para Baseline e Proposed: acumula todas as execuções
    for method in ['Baseline', 'Proposed']:
        df_method = df_pareto[df_pareto['Method'] == method]
        if len(df_method) > 0:
            costs = df_method['Total_Cost_USD'].values / 1000.0  # kUSD
            aeps = df_method['Net_AEP_GWh'].values
            
            # Ordena por custo para conectar pontos
            sorted_indices = np.argsort(costs)
            costs_sorted = costs[sorted_indices]
            aeps_sorted = aeps[sorted_indices]
            
            # Plota linha conectando pontos (frente de Pareto agregada)
            ax.plot(costs_sorted, aeps_sorted, '--', color=colors[method], 
                   linewidth=2.0, alpha=0.6, zorder=2)
            
            # Plota todos os pontos
            # Converte cor para versão mais escura
            dark_color = mcolors.to_rgb(colors[method])
            dark_color = tuple([max(0, c * 0.7) for c in dark_color])  # Escurece 30%
            
            ax.scatter(costs, aeps, s=50, alpha=0.7, color=colors[method],
                      edgecolors=dark_color, linewidths=1.0,
                      marker=markers[method], label=f'{method} (n={len(df_method)} solutions)', zorder=3)
            
            # Calcula knee point do aggregated pareto front
            knee = find_knee_point_from_df(df_method)
            if knee is not None:
                knee_points[method] = knee
                # Destaca knee point com círculo preto maior
                ax.scatter(knee['Total_Cost_USD'] / 1000.0, knee['Net_AEP_GWh'],
                          s=200, alpha=1.0, color=colors[method],
                          edgecolors='black', linewidths=3.0,
                          marker=markers[method], zorder=5, label=f'{method} Knee Point')
    
    # Para Sequential: acumula todas as execuções
    df_sequential = df_summary[df_summary['Method'] == 'Sequential']
    if len(df_sequential) > 0:
        costs_seq = df_sequential['Total_Cost_USD'].values / 1000.0
        aeps_seq = df_sequential['Net_AEP_GWh'].values
        
        # Ordena por custo
        sorted_indices = np.argsort(costs_seq)
        costs_seq_sorted = costs_seq[sorted_indices]
        aeps_seq_sorted = aeps_seq[sorted_indices]
        
        # Plota linha
        ax.plot(costs_seq_sorted, aeps_seq_sorted, '--', color=colors['Sequential'],
               linewidth=2.0, alpha=0.6, zorder=2)
        
        # Plota pontos
        dark_seq_color = mcolors.to_rgb(colors['Sequential'])
        dark_seq_color = tuple([max(0, c * 0.7) for c in dark_seq_color])  # Escurece 30%
        
        ax.scatter(costs_seq, aeps_seq, s=80, alpha=0.8, color=colors['Sequential'],
                  edgecolors=dark_seq_color, linewidths=1.5,
                  marker=markers['Sequential'], label=f'Sequential (n={len(df_sequential)} runs)', zorder=4)
        
        # Calcula knee point do aggregated pareto front (Sequential)
        # Para Sequential, usa summary_results como se fosse pareto front
        knee_seq = find_knee_point_from_df(df_sequential)
        if knee_seq is not None:
            knee_points['Sequential'] = knee_seq
            # Destaca knee point
            ax.scatter(knee_seq['Total_Cost_USD'] / 1000.0, knee_seq['Net_AEP_GWh'],
                      s=200, alpha=1.0, color=colors['Sequential'],
                      edgecolors='black', linewidths=3.0,
                      marker=markers['Sequential'], zorder=5, label=f'Sequential Knee Point')
    
    # Configurações
    ax.set_xlabel('Total Cabling Cost (kUSD)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_ylabel('Net Annual Energy Production (GWh)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_title('Aggregated Pareto Fronts', fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(frameon=True, loc='best', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    
    # Melhora aparência
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'aggregated_pareto_fronts.png')
    output_path_pdf = os.path.join(output_dir, 'aggregated_pareto_fronts.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Frentes de Pareto Agregadas salvas em: {output_path_png} e {output_path_pdf}")
    plt.close()
    
    return knee_points

def plot_qualitative_metrics(df_summary, output_dir='.'):
    """
    Plota métricas qualitativas: cable groups, cable length, electrical losses, substation distance.
    Similar às imagens qualitative_*.png.
    
    Args:
        df_summary: DataFrame com summary_results.csv
        output_dir: Diretório de saída
    """
    if len(df_summary) == 0:
        print("AVISO: Nenhum dado disponível para métricas qualitativas")
        return
    
    methods = sorted(df_summary['Method'].unique())
    colors = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}
    
    # Métricas qualitativas disponíveis
    metrics = {
        'Num_Cable_Strings': {
            'data': [], 
            'ylabel': 'Number of Cable Strings', 
            'title': 'Cable Group Distribution'
        },
        'Total_Cable_Length_km': {
            'data': [], 
            'ylabel': 'Total Cable Length (km)', 
            'title': 'Cable Length Comparison'
        },
        'Electrical_Loss_Percentage': {
            'data': [], 
            'ylabel': 'Electrical Loss Percentage (%)', 
            'title': 'Electrical Losses'
        },
        'Substation_Eccentricity_m': {
            'data': [], 
            'ylabel': 'Substation Eccentricity (m)', 
            'title': 'Substation Distance from Centroid'
        }
    }
    
    # Agrupa dados por método
    for method in methods:
        df_method = df_summary[df_summary['Method'] == method]
        
        for metric_name in metrics.keys():
            if metric_name in df_method.columns:
                values = df_method[metric_name].dropna().values
                metrics[metric_name]['data'].append(values if len(values) > 0 else [0])
            else:
                metrics[metric_name]['data'].append([0])
    
    # Cria subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (metric_name, metric_info) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        data = metric_info['data']
        
        if len(data) > 0 and any(len(d) > 0 for d in data):
            # Usa violin plot para melhor visualização da distribuição
            parts = ax.violinplot(data, positions=range(len(methods)), widths=0.6, 
                                 showmeans=True, showmedians=True)
            
            # Colore os violinos
            for pc, method in zip(parts['bodies'], methods):
                pc.set_facecolor(colors[method])
                pc.set_alpha(0.7)
            
            # Estiliza outros elementos
            for element in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                if element in parts:
                    parts[element].set_color('black')
                    parts[element].set_linewidth(1.2)
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, fontsize=FONT_SIZE_TICK)
            ax.set_ylabel(metric_info['ylabel'], fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
            ax.set_title(metric_info['title'], fontsize=FONT_SIZE_TITLE_SUBPLOT, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.suptitle('Qualitative Analysis of Solution Characteristics', 
                fontsize=FONT_SIZE_TITLE_MAIN, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, 'qualitative_metrics.png')
    output_path_pdf = os.path.join(output_dir, 'qualitative_metrics.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Métricas Qualitativas salvas em: {output_path_png} e {output_path_pdf}")
    plt.close()

def plot_individual_solution_layout(turb_coords, sub_pos, n_grupos, method_name, 
                                   aep_gwh, cost_kusd, cable_length_km, 
                                   electrical_loss_pct, n_strings, output_dir='.', 
                                   solution_id=0, run_id=1):
    """
    Plota layout individual de uma solução (turbinas, subestação, cabos).
    Similar às imagens baseline_solution_*.png.
    
    Args:
        turb_coords: Array numpy com coordenadas das turbinas (N, 2)
        sub_pos: Array numpy com posição da subestação (2,)
        n_grupos: Número de grupos de cabeamento
        method_name: Nome do método ('Baseline', 'Proposed', 'Sequential')
        aep_gwh: AEP líquido em GWh
        cost_kusd: Custo em kUSD
        cable_length_km: Comprimento total de cabos em km
        electrical_loss_pct: Percentual de perdas elétricas
        n_strings: Número de ramais de cabo
        output_dir: Diretório de saída
        solution_id: ID da solução
        run_id: ID da execução
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Calcula cabeamento
    coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
    substation_idx = len(turb_coords)
    
    try:
        plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=substation_idx, n_grupos=n_grupos)
    except Exception as e:
        print(f"ERRO ao calcular cabeamento para plot: {e}")
        return
    
    # Desenha círculo de restrição
    circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                   linewidth=2, color='black', alpha=0.5, zorder=1)
    ax.add_patch(circle)
    
    # Desenha cabeamento
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(plant.paths), 10)))
    for i, path in enumerate(plant.paths):
        if len(path) > 1:
            valid_path = [k for k in path if 0 <= k < len(coords_all)]
            if len(valid_path) > 1:
                x_path = [coords_all[k, 0] for k in valid_path]
                y_path = [coords_all[k, 1] for k in valid_path]
                ax.plot(x_path, y_path, '-', linewidth=CABLE_LINEWIDTH, 
                       color=colors[i % len(colors)], alpha=0.8, zorder=4)
    
    # Desenha turbinas
    ax.scatter(turb_coords[:, 0], turb_coords[:, 1], 
              s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
              linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
    
    # Desenha subestação
    ax.scatter(sub_pos[0], sub_pos[1],
              marker='*', s=SUBSTATION_MARKER_SIZE, c='gold', edgecolors='black',
              linewidths=TURBINE_EDGE_WIDTH, zorder=6, label='Substation')
    
    # Configurações
    ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
    
    # Título com métricas
    title_text = (
        f'{method_name} - Best Solution\n'
        f'AEP: {aep_gwh:.2f} GWh | Cost: ${cost_kusd:.0f}k USD\n'
        f'Cable: {cable_length_km:.2f} km | Losses: {electrical_loss_pct:.2f}% | Groups: {n_strings}'
    )
    ax.set_title(title_text, fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(frameon=True, loc='upper right', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    
    # Melhora aparência
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_basename = f'{method_name.lower()}_solution_run{run_id}_id{solution_id}'
    output_path_png = os.path.join(output_dir, f'{output_basename}.png')
    output_path_pdf = os.path.join(output_dir, f'{output_basename}.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Layout {method_name} salvo em: {output_path_png}")
    plt.close()

def plot_representative_solutions(df_summary, df_pareto, knee_points_aggregated, output_dir='.'):
    """
    Plota layouts das soluções representativas: knee points do aggregated pareto front.
    Apenas 3 layouts: um por método (Baseline, Proposed, Sequential).
    
    Args:
        df_summary: DataFrame com summary_results.csv
        df_pareto: DataFrame com all_pareto_fronts.csv
        knee_points_aggregated: Dicionário com knee points do aggregated pareto front
                              {'Baseline': row, 'Proposed': row, 'Sequential': row}
        output_dir: Diretório de saída
    """
    print("\n>>> Gerando layouts das soluções representativas (knee points do aggregated pareto front)...")
    
    if len(knee_points_aggregated) == 0:
        print("   AVISO: Nenhum knee point disponível. Execute plot_aggregated_pareto_fronts primeiro.")
        return
    
    # Carrega CSV de coordenadas
    coords_path = os.path.join(output_dir, 'representative_solutions.csv')
    if not os.path.exists(coords_path):
        print(f"   AVISO: Arquivo não encontrado: {coords_path}")
        print("   Execute o case_study_comparison_original.py primeiro para gerar coordenadas.")
        return
    
    df_coords = pd.read_csv(coords_path)
    if len(df_coords) == 0:
        print("   AVISO: Nenhuma coordenada disponível.")
        return
    
    # Para cada método, encontra a solução mais próxima do knee point do aggregated pareto front
    for method, knee_row in knee_points_aggregated.items():
        if knee_row is None:
            continue
        
        # Busca no CSV de coordenadas a solução mais próxima do knee point
        # Compara por AEP e Custo
        target_aep = knee_row['Net_AEP_GWh']
        target_cost = knee_row['Total_Cost_USD']
        
        df_method_coords = df_coords[df_coords['Method'] == method]
        if len(df_method_coords) == 0:
            print(f"   AVISO: Nenhuma coordenada disponível para {method}")
            continue
        
        # Calcula distância normalizada para encontrar solução mais próxima
        aeps = df_method_coords['Net_AEP_GWh'].values
        costs = df_method_coords['Total_Cost_USD'].values
        
        # Normaliza
        aep_range = aeps.max() - aeps.min() if aeps.max() != aeps.min() else 1.0
        cost_range = costs.max() - costs.min() if costs.max() != costs.min() else 1.0
        
        aep_norm = (aeps - aeps.min()) / aep_range
        cost_norm = (costs - costs.min()) / cost_range
        target_aep_norm = (target_aep - aeps.min()) / aep_range
        target_cost_norm = (target_cost - costs.min()) / cost_range
        
        # Calcula distâncias
        distances = np.sqrt((aep_norm - target_aep_norm)**2 + (cost_norm - target_cost_norm)**2)
        closest_idx = np.argmin(distances)
        closest_row = df_method_coords.iloc[closest_idx]
        
        # Deserializa coordenadas das turbinas
        try:
            turb_x = np.array([float(x) for x in closest_row['Turbine_Coords_X'].split(',')])
            turb_y = np.array([float(y) for y in closest_row['Turbine_Coords_Y'].split(',')])
            turb_coords = np.column_stack([turb_x, turb_y])
            
            sub_pos = np.array([closest_row['Substation_X'], closest_row['Substation_Y']])
            n_grupos = int(closest_row['N_Grupos'])
            
            # Obtém métricas do summary_results ou usa do knee point
            df_method_summary = df_summary[(df_summary['Method'] == method) & 
                                           (df_summary['Run_ID'] == closest_row['Run_ID'])]
            if len(df_method_summary) > 0:
                summary_row = df_method_summary.iloc[0]
                aep_gwh = summary_row['Net_AEP_GWh']
                cost_kusd = summary_row['Total_Cost_USD'] / 1000.0
                cable_length_km = summary_row.get('Total_Cable_Length_km', 0.0)
                electrical_loss_pct = summary_row.get('Electrical_Loss_Percentage', 0.0)
                n_strings = summary_row.get('Num_Cable_Strings', n_grupos)
            else:
                # Usa valores do knee point
                aep_gwh = target_aep
                cost_kusd = target_cost / 1000.0
                cable_length_km = 0.0
                electrical_loss_pct = 0.0
                n_strings = n_grupos
            
            # Plota layout (knee point do aggregated pareto front)
            plot_individual_solution_layout(
                turb_coords, sub_pos, n_grupos, method,
                aep_gwh, cost_kusd, cable_length_km,
                electrical_loss_pct, n_strings,
                output_dir=output_dir,
                solution_id=0, run_id=int(closest_row['Run_ID'])
            )
            
            print(f"   ✓ Layout {method} (knee point do aggregated pareto front) gerado")
            
        except Exception as e:
            print(f"   ERRO ao plotar solução {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✓ Layouts das soluções representativas gerados em: {output_dir}")

def calculate_and_save_success_rates(df_summary, output_dir='.'):
    """
    Calcula e salva taxa de sucesso para todos os métodos (Baseline, Proposed, Sequential).
    Uma solução é considerada sucesso se Net_AEP_GWh > 0 e Total_Cost_USD > 0.
    
    Args:
        df_summary: DataFrame com summary_results.csv
        output_dir: Diretório de saída
    """
    if len(df_summary) == 0:
        print("AVISO: Nenhum dado disponível para calcular taxa de sucesso")
        return
    
    methods = ['Baseline', 'Proposed', 'Sequential']
    success_stats = {}
    
    for method in methods:
        df_method = df_summary[df_summary['Method'] == method]
        if len(df_method) == 0:
            continue
        
        # Considera sucesso se Net_AEP_GWh > 0 e Total_Cost_USD > 0 e não infinito
        valid_solutions = df_method[
            (df_method['Net_AEP_GWh'] > 0) & 
            (df_method['Total_Cost_USD'] > 0) &
            (df_method['Total_Cost_USD'] != float('inf'))
        ]
        
        total_runs = len(df_method)
        success_count = len(valid_solutions)
        failure_count = total_runs - success_count
        success_rate = (success_count / total_runs * 100.0) if total_runs > 0 else 0.0
        
        success_stats[method] = {
            'total_runs': total_runs,
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_rate
        }
    
    # Salva estatísticas em arquivo de texto
    stats_file = os.path.join(output_dir, 'success_rates_all_methods.txt')
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TAXA DE SUCESSO - TODOS OS MÉTODOS\n")
        f.write("="*80 + "\n\n")
        
        for method in methods:
            if method in success_stats:
                stats = success_stats[method]
                f.write(f"{method.upper()}:\n")
                f.write(f"  Total de execuções: {stats['total_runs']}\n")
                f.write(f"  Sucessos (soluções válidas encontradas): {stats['success_count']}\n")
                f.write(f"  Falhas (nenhuma solução válida): {stats['failure_count']}\n")
                f.write(f"  Taxa de sucesso: {stats['success_rate']:.1f}%\n")
                f.write(f"  Taxa de falha: {(100 - stats['success_rate']):.1f}%\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("NOTA: Uma solução é considerada sucesso se Net_AEP_GWh > 0 e Total_Cost_USD > 0\n")
        f.write("="*80 + "\n")
    
    # Imprime no console também
    print("\n" + "="*80)
    print("TAXA DE SUCESSO - TODOS OS MÉTODOS")
    print("="*80)
    for method in methods:
        if method in success_stats:
            stats = success_stats[method]
            print(f"\n{method.upper()}:")
            print(f"  Total de execuções: {stats['total_runs']}")
            print(f"  Sucessos: {stats['success_count']}")
            print(f"  Falhas: {stats['failure_count']}")
            print(f"  Taxa de sucesso: {stats['success_rate']:.1f}%")
            print(f"  Taxa de falha: {(100 - stats['success_rate']):.1f}%")
    print("="*80)
    print(f"✓ Estatísticas de taxa de sucesso salvas em: {stats_file}")

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main(results_dir):
    """
    Função principal: lê os CSVs e gera todos os gráficos.
    
    Args:
        results_dir: Diretório onde estão os CSVs (padrão: 'results_36')
    """
    print("="*80)
    print("GERANDO GRÁFICOS DOS RESULTADOS DO ESTUDO DE CASO")
    print("="*80)
    
    # Caminhos dos arquivos CSV
    summary_path = os.path.join(results_dir, 'summary_results.csv')
    pareto_path = os.path.join(results_dir, 'all_pareto_fronts.csv')
    convergence_path = os.path.join(results_dir, 'convergence_history.csv')
    
    # Verifica se os arquivos existem
    if not os.path.exists(summary_path):
        print(f"ERRO: Arquivo não encontrado: {summary_path}")
        return
    
    # Carrega dados
    print(f"\n>>> Carregando dados de {summary_path}...")
    df_summary = pd.read_csv(summary_path)
    print(f"   {len(df_summary)} linhas carregadas")
    
    df_pareto = pd.DataFrame()
    if os.path.exists(pareto_path):
        print(f"\n>>> Carregando dados de {pareto_path}...")
        df_pareto = pd.read_csv(pareto_path)
        print(f"   {len(df_pareto)} linhas carregadas")
    else:
        print(f"\nAVISO: Arquivo não encontrado: {pareto_path} (pode não ter frentes de Pareto)")
    
    df_convergence = pd.DataFrame()
    if os.path.exists(convergence_path):
        print(f"\n>>> Carregando dados de {convergence_path}...")
        df_convergence = pd.read_csv(convergence_path)
        print(f"   {len(df_convergence)} linhas carregadas")
    else:
        print(f"\nAVISO: Arquivo não encontrado: {convergence_path} (pode não ter histórico de convergência)")
    
    # Gera gráficos
    print("\n" + "="*80)
    print("GERANDO GRÁFICOS")
    print("="*80)
    
    # 1. Comparação de frentes de Pareto
    if len(df_pareto) > 0:
        print("\n>>> Gerando comparação de frentes de Pareto...")
        plot_pareto_fronts_comparison(df_summary, df_pareto, output_dir=results_dir)
    else:
        print("\n>>> Pulando comparação de frentes de Pareto (sem dados)")
    
    # 2. Histórico de hipervolume
    if len(df_convergence) > 0:
        print("\n>>> Gerando histórico de hipervolume...")
        plot_hypervolume_history(df_convergence, output_dir=results_dir)
    else:
        print("\n>>> Pulando histórico de hipervolume (sem dados)")
    
    # 3. Boxplots comparativos
    print("\n>>> Gerando boxplots comparativos...")
    plot_boxplots_comparison(df_summary, output_dir=results_dir)
    
    # 4. Tempo de execução
    print("\n>>> Gerando gráfico de tempo de execução...")
    plot_execution_time_comparison(df_summary, output_dir=results_dir)
    
    # 5. Frentes de Pareto Agregadas (retorna knee points)
    knee_points_aggregated = {}
    if len(df_pareto) > 0:
        print("\n>>> Gerando frentes de Pareto agregadas...")
        knee_points_aggregated = plot_aggregated_pareto_fronts(df_pareto, df_summary, output_dir=results_dir)
    else:
        print("\n>>> Pulando frentes de Pareto agregadas (sem dados)")
    
    # 6. Métricas Qualitativas
    print("\n>>> Gerando métricas qualitativas...")
    plot_qualitative_metrics(df_summary, output_dir=results_dir)
    
    # 7. Layouts de soluções representativas (knee points do aggregated pareto front)
    print("\n>>> Gerando layouts das soluções representativas (knee points do aggregated pareto front)...")
    plot_representative_solutions(df_summary, df_pareto, knee_points_aggregated, output_dir=results_dir)
    
    # 8. Estatísticas de taxa de sucesso
    print("\n>>> Calculando taxa de sucesso para todos os métodos...")
    calculate_and_save_success_rates(df_summary, output_dir=results_dir)
    
    print("\n" + "="*80)
    print("TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
    print("="*80)
    print(f"✓ Gráficos salvos em: {os.path.abspath(results_dir)}")

if __name__ == "__main__":

    main(results_dir='results_16')
