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
from scipy import stats  # For statistical significance tests


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

# Cores para publicação (consistência em todos os gráficos)
COLORS_PUBLICATION = {
    'Proposed': '#2ca02c',    # Verde
    'Baseline': '#1f77b4',    # Azul
    'Sequential': '#d62728'   # Vermelho
}
MARKERS_PUBLICATION = {
    'Baseline': 'o',          # Círculo
    'Proposed': 's',          # Quadrado
    'Sequential': '^'         # Triângulo
}


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

def plot_scalability_metrics(results_dirs, output_dir='.'):
    """
    Gera gráficos de escalabilidade comparando múltiplas escalas (16, 36, 64 turbinas).
    Similar aos gráficos scalability_*.png.
    
    Args:
        results_dirs: Lista de tuplas (scale, dir_path) ou dict {scale: dir_path}
                     Ex: [('16', 'results_16'), ('36', 'results_36'), ('64', 'results_64')]
        output_dir: Diretório de saída
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para análise de escalabilidade")
        return
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para análise de escalabilidade")
        return
    
    colors = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}
    markers = {'Baseline': 'o', 'Proposed': 's', 'Sequential': '^'}
    
    # Métricas de escalabilidade
    metrics = {
        'net_aep': {
            'col': 'Net_AEP_GWh',
            'ylabel': 'Net AEP (GWh)',
            'title': 'Net AEP Scalability'
        },
        'capex': {
            'col': 'Total_Cost_USD',
            'ylabel': 'CAPEX (USD)',
            'title': 'CAPEX Scalability',
            'scale': 1000,  # Converte para kUSD
            'ylabel_scaled': 'CAPEX (kUSD)'
        },
        'execution_time': {
            'col': 'Time_Total_s',
            'ylabel': 'Execution Time (seconds)',
            'title': 'Computational Efficiency Scalability',
            'scale': 60,  # Converte para minutos
            'ylabel_scaled': 'Execution Time (minutes)'
        }
    }
    
    scales = sorted([int(s) for s in dfs.keys()])
    
    for metric_key, metric_info in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            means = []
            stds = []
            valid_scales = []
            
            for scale in scales:
                scale_str = str(scale)
                if scale_str in dfs:
                    df_scale = dfs[scale_str]
                    data = df_scale[df_scale['Method'] == method]
                    
                    if len(data) > 0 and metric_info['col'] in data.columns:
                        values = data[metric_info['col']].values
                        # Aplica escala se necessário
                        if 'scale' in metric_info:
                            values = values / metric_info['scale']
                        
                        if len(values) > 0:
                            means.append(np.mean(values))
                            stds.append(np.std(values))
                            valid_scales.append(scale)
            
            if len(means) > 0:
                ax.errorbar(valid_scales, means, yerr=stds,
                           label=method, color=colors[method],
                           marker=markers[method],
                           markersize=12, linewidth=3, capsize=8, capthick=2.5,
                           elinewidth=2.5)
        
        ax.set_xlabel('Number of Turbines', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
        ylabel = metric_info.get('ylabel_scaled', metric_info['ylabel'])
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
        ax.set_title(metric_info['title'], fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=1.5, length=6)
        ax.set_xticks(scales)
        
        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'scalability_{metric_key}.png')
        output_path_pdf = os.path.join(output_dir, f'scalability_{metric_key}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico gerado: {output_path}")
        plt.close()

def plot_qualitative_metrics_multi_scale(results_dirs, output_dir='.'):
    """
    Gera gráficos de características qualitativas comparando múltiplas escalas.
    Similar aos gráficos qualitative_*.png.
    
    Args:
        results_dirs: Lista de tuplas (scale, dir_path) ou dict {scale: dir_path}
        output_dir: Diretório de saída
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para análise qualitativa")
        return
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            # Calcula distância da subestação (se não existir)
            if 'Substation_Eccentricity_m' not in df.columns:
                # Tenta calcular se tiver coordenadas
                if 'Substation_X' in df.columns and 'Substation_Y' in df.columns:
                    df['Substation_Eccentricity_m'] = np.sqrt(
                        df['Substation_X']**2 + df['Substation_Y']**2
                    )
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para análise qualitativa")
        return
    
    colors = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}
    
    # Características qualitativas
    characteristics = {
        'substation_distance': {
            'col': 'Substation_Eccentricity_m',
            'ylabel': 'Substation Distance (m)',
            'title': 'Substation Distance by Method and Scale'
        },
        'cable_length': {
            'col': 'Total_Cable_Length_km',
            'ylabel': 'Total Cable Length (km)',
            'title': 'Cable Length by Method and Scale'
        },
        'cable_groups': {
            'col': 'Num_Cable_Strings',
            'ylabel': 'Number of Cable Groups',
            'title': 'Cable Groups by Method and Scale'
        },
        'electrical_losses': {
            'col': 'Electrical_Loss_Percentage',
            'ylabel': 'Electrical Losses (%)',
            'title': 'Electrical Losses by Method and Scale'
        }
    }
    
    scales = sorted([int(s) for s in dfs.keys()])
    scale_strs = [str(s) for s in scales]
    x = np.arange(len(scales))
    width = 0.25
    
    for char_key, char_info in characteristics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for i, method in enumerate(['Baseline', 'Proposed', 'Sequential']):
            means = []
            stds = []
            
            for scale_str in scale_strs:
                if scale_str in dfs:
                    df_scale = dfs[scale_str]
                    data = df_scale[df_scale['Method'] == method]
                    
                    if len(data) > 0 and char_info['col'] in data.columns:
                        values = data[char_info['col']].dropna().values
                        if len(values) > 0:
                            means.append(np.mean(values))
                            stds.append(np.std(values))
                        else:
                            means.append(0)
                            stds.append(0)
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i*width, means, width, label=method, color=colors[method],
                  alpha=0.8, yerr=stds, capsize=8, error_kw={'linewidth': 2.5, 'capthick': 2.5})
        
        ax.set_xlabel('Number of Turbines', fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
        ax.set_ylabel(char_info['ylabel'], fontsize=FONT_SIZE_LABEL_AXIS, fontweight='bold')
        ax.set_title(char_info['title'], fontsize=FONT_SIZE_TITLE_PLOT, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(s) for s in scales])
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y', linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=1.5, length=6)
        
        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'qualitative_{char_key}.png')
        output_path_pdf = os.path.join(output_dir, f'qualitative_{char_key}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico gerado: {output_path}")
        plt.close()

# =============================================================================
# FUNÇÕES DE PLOTAGEM SEPARADAS (PUBLICAÇÃO)
# =============================================================================

def plot_scalability_separate(results_dirs, output_dir='.'):
    """
    Gera gráficos separados para cada métrica de escalabilidade (estilo publicação).
    Similar ao generate_figures_tables.py mas integrado ao workflow existente.
    
    Args:
        results_dirs: Dict {scale: dir_path} Ex: {'16': 'results_16', '36': 'results_36'}
        output_dir: Diretório de saída
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para análise de escalabilidade")
        return
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para análise de escalabilidade")
        return
    
    scales = sorted([int(s) for s in dfs.keys()])
    
    # Métricas de escalabilidade
    metrics = {
        'net_aep': {
            'col': 'Net_AEP_GWh',
            'ylabel': 'Net AEP (GWh)',
            'title': 'Net AEP Scalability'
        },
        'capex': {
            'col': 'Total_Cost_USD',
            'ylabel': 'CAPEX (kUSD)',
            'title': 'CAPEX Scalability',
            'scale_factor': 1000  # USD -> kUSD
        },
        'execution_time': {
            'col': 'Time_Total_s',
            'ylabel': 'Execution Time (minutes)',
            'title': 'Computational Efficiency Scalability',
            'scale_factor': 60  # seconds -> minutes
        }
    }
    
    for metric_key, metric_info in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            means = []
            stds = []
            valid_scales = []
            
            for scale in scales:
                scale_str = str(scale)
                if scale_str in dfs:
                    df_scale = dfs[scale_str]
                    data = df_scale[df_scale['Method'] == method]
                    
                    if len(data) > 0 and metric_info['col'] in data.columns:
                        values = data[metric_info['col']].values
                        
                        # Aplica escala se necessário
                        if 'scale_factor' in metric_info:
                            values = values / metric_info['scale_factor']
                        
                        if len(values) > 0:
                            means.append(np.mean(values))
                            stds.append(np.std(values))
                            valid_scales.append(scale)
            
            if len(means) > 0:
                ax.errorbar(valid_scales, means, yerr=stds,
                           label=method, color=COLORS_PUBLICATION[method],
                           marker=MARKERS_PUBLICATION[method],
                           markersize=12, linewidth=3, capsize=8, capthick=2.5,
                           elinewidth=2.5)
        
        ax.set_xlabel('Number of Turbines', fontsize=20, fontweight='bold')
        ax.set_ylabel(metric_info['ylabel'], fontsize=20, fontweight='bold')
        ax.set_title(metric_info['title'], fontsize=24, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
        ax.legend(fontsize=18, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=18, width=1.5, length=6)
        ax.set_xticks(scales)
        
        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'scalability_{metric_key}.png')
        output_path_pdf = os.path.join(output_dir, f'scalability_{metric_key}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico de escalabilidade gerado: {output_path}")
        plt.close()

def plot_qualitative_separate(results_dirs, output_dir='.'):
    """
    Gera gráficos separados para características qualitativas (estilo publicação).
    Similar ao generate_figures_tables.py mas integrado ao workflow existente.
    
    Args:
        results_dirs: Dict {scale: dir_path}
        output_dir: Diretório de saída
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para análise qualitativa")
        return
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            # Calcula métrica derivada se não existir
            if 'Substation_Eccentricity_m' not in df.columns:
                if 'Substation_X' in df.columns and 'Substation_Y' in df.columns:
                    df['Substation_Eccentricity_m'] = np.sqrt(
                        df['Substation_X']**2 + df['Substation_Y']**2
                    )
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para análise qualitativa")
        return
    
    # Características qualitativas
    characteristics = {
        'substation_distance': {
            'col': 'Substation_Eccentricity_m',
            'ylabel': 'Substation Distance (m)',
            'title': 'Substation Distance by Method and Scale'
        },
        'cable_length': {
            'col': 'Total_Cable_Length_km',
            'ylabel': 'Total Cable Length (km)',
            'title': 'Cable Length by Method and Scale'
        },
        'cable_groups': {
            'col': 'Num_Cable_Strings',
            'ylabel': 'Number of Cable Groups',
            'title': 'Cable Groups by Method and Scale'
        },
        'electrical_losses': {
            'col': 'Electrical_Loss_Percentage',
            'ylabel': 'Electrical Losses (%)',
            'title': 'Electrical Losses by Method and Scale'
        }
    }
    
    scales = sorted([int(s) for s in dfs.keys()])
    scale_strs = [str(s) for s in scales]
    x = np.arange(len(scales))
    width = 0.25
    
    for char_key, char_info in characteristics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for i, method in enumerate(['Baseline', 'Proposed', 'Sequential']):
            means = []
            stds = []
            
            for scale_str in scale_strs:
                if scale_str in dfs:
                    df_scale = dfs[scale_str]
                    data = df_scale[df_scale['Method'] == method]
                    
                    if len(data) > 0 and char_info['col'] in data.columns:
                        values = data[char_info['col']].dropna().values
                        if len(values) > 0:
                            means.append(np.mean(values))
                            stds.append(np.std(values))
                        else:
                            means.append(0)
                            stds.append(0)
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i*width, means, width, label=method, color=COLORS_PUBLICATION[method],
                  alpha=0.8, yerr=stds, capsize=8, error_kw={'linewidth': 2.5, 'capthick': 2.5})
        
        ax.set_xlabel('Number of Turbines', fontsize=20, fontweight='bold')
        ax.set_ylabel(char_info['ylabel'], fontsize=20, fontweight='bold')
        ax.set_title(char_info['title'], fontsize=24, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(s) for s in scales])
        ax.legend(fontsize=18, loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y', linewidth=1.5)
        ax.tick_params(labelsize=18, width=1.5, length=6)
        
        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'qualitative_{char_key}.png')
        output_path_pdf = os.path.join(output_dir, f'qualitative_{char_key}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico qualitativo gerado: {output_path}")
        plt.close()

def generate_latex_tables(results_dirs, output_dir='.'):
    """
    Gera tabelas LaTeX com dados resumidos para múltiplas escalas.
    Similar ao generate_figures_tables.py mas integrado ao workflow existente.
    
    Args:
        results_dirs: Dict {scale: dir_path}
        output_dir: Diretório de saída
    
    Returns:
        tuple: (latex_quant, latex_qual) - Strings com código LaTeX das tabelas
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para gerar tabelas LaTeX")
        return None, None
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            # Calcula métricas derivadas
            if 'Substation_Eccentricity_m' not in df.columns:
                if 'Substation_X' in df.columns and 'Substation_Y' in df.columns:
                    df['Substation_Eccentricity_m'] = np.sqrt(
                        df['Substation_X']**2 + df['Substation_Y']**2
                    )
            if 'Electrical_Loss_Percentage' not in df.columns:
                if 'Perdas_Joule_MWh' in df.columns and 'AEP_Bruto_MWh' in df.columns:
                    df['Electrical_Loss_Percentage'] = (
                        df['Perdas_Joule_MWh'] / df['AEP_Bruto_MWh'] * 100
                    )
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para gerar tabelas LaTeX")
        return None, None
    
    scales_sorted = sorted(dfs.keys(), key=lambda x: int(x))
    
    # Tabela de métricas quantitativas
    latex_quant = """\\begin{table}[htbp]
\\caption{Quantitative performance metrics across problem scales (mean $\\pm$ std).}
\\label{tab:quantitative_metrics}
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{clccc}
\\toprule
\\textbf{Scale} & \\textbf{Method} & \\textbf{Net AEP} & \\textbf{CAPEX} & \\textbf{Time} \\\\
 &  & [GWh] & [kUSD] & [min] \\\\
\\midrule
"""
    
    for scale in scales_sorted:
        df_scale = dfs[scale]
        latex_quant += f"\\multirow{{3}}{{*}}{{{scale}}}\n"
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            data = df_scale[df_scale['Method'] == method]
            if len(data) > 0:
                # Extrai métricas com conversões de unidades
                aep_mean = data['Net_AEP_GWh'].mean() if 'Net_AEP_GWh' in data.columns else 0
                aep_std = data['Net_AEP_GWh'].std() if 'Net_AEP_GWh' in data.columns else 0
                
                cost_mean = (data['Total_Cost_USD'].mean() / 1000.0) if 'Total_Cost_USD' in data.columns else 0
                cost_std = (data['Total_Cost_USD'].std() / 1000.0) if 'Total_Cost_USD' in data.columns else 0
                
                time_mean = (data['Time_Total_s'].mean() / 60.0) if 'Time_Total_s' in data.columns else 0
                time_std = (data['Time_Total_s'].std() / 60.0) if 'Time_Total_s' in data.columns else 0
                
                latex_quant += f" & {method:12s} & ${aep_mean:.2f} \\pm {aep_std:.2f}$ & ${cost_mean:.0f} \\pm {cost_std:.0f}$ & ${time_mean:.1f} \\pm {time_std:.1f}$ \\\\\n"
        
        if scale != scales_sorted[-1]:
            latex_quant += "\\midrule\n"
    
    latex_quant += """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    
    # Tabela de características qualitativas
    latex_qual = """\\begin{table*}[htbp]
\\caption{Qualitative characteristics across problem scales (mean $\\pm$ std).}
\\label{tab:qualitative_metrics}
\\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{clcccc}
\\toprule
\\textbf{Scale} & \\textbf{Method} & \\textbf{Subst. Dist.} & \\textbf{Cable Length} & \\textbf{Groups} & \\textbf{Losses} \\\\
 &  & [m] & [km] &  & [\\%] \\\\
\\midrule
"""
    
    for scale in scales_sorted:
        df_scale = dfs[scale]
        latex_qual += f"\\multirow{{3}}{{*}}{{{scale}}}\n"
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            data = df_scale[df_scale['Method'] == method]
            if len(data) > 0:
                subst_mean = data['Substation_Eccentricity_m'].mean() if 'Substation_Eccentricity_m' in data.columns else 0
                subst_std = data['Substation_Eccentricity_m'].std() if 'Substation_Eccentricity_m' in data.columns else 0
                
                cable_mean = data['Total_Cable_Length_km'].mean() if 'Total_Cable_Length_km' in data.columns else 0
                cable_std = data['Total_Cable_Length_km'].std() if 'Total_Cable_Length_km' in data.columns else 0
                
                groups_mean = data['Num_Cable_Strings'].mean() if 'Num_Cable_Strings' in data.columns else 0
                groups_std = data['Num_Cable_Strings'].std() if 'Num_Cable_Strings' in data.columns else 0
                
                losses_mean = data['Electrical_Loss_Percentage'].mean() if 'Electrical_Loss_Percentage' in data.columns else 0
                losses_std = data['Electrical_Loss_Percentage'].std() if 'Electrical_Loss_Percentage' in data.columns else 0
                
                latex_qual += f" & {method:12s} & ${subst_mean:.0f} \\pm {subst_std:.0f}$ & ${cable_mean:.2f} \\pm {cable_std:.2f}$ & ${groups_mean:.1f} \\pm {groups_std:.1f}$ & ${losses_mean:.2f} \\pm {losses_std:.2f}$ \\\\\n"
        
        if scale != scales_sorted[-1]:
            latex_qual += "\\midrule\n"
    
    latex_qual += """\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""
    
    # Salvar tabelas
    os.makedirs(output_dir, exist_ok=True)
    quant_path = os.path.join(output_dir, 'table_quantitative_metrics.tex')
    qual_path = os.path.join(output_dir, 'table_qualitative_metrics.tex')
    
    with open(quant_path, 'w') as f:
        f.write(latex_quant)
    print(f"✓ Tabela LaTeX gerada: {quant_path}")
    
    with open(qual_path, 'w') as f:
        f.write(latex_qual)
    print(f"✓ Tabela LaTeX gerada: {qual_path}")
    
    return latex_quant, latex_qual

def perform_statistical_tests(results_dirs, output_dir='.'):
    """
    Realiza testes de significância estatística comparando métodos.
    
    Testes aplicados:
    - Mann-Whitney U test (comparação entre 2 grupos independentes)
    - Compara Proposed vs Baseline e Sequential vs Baseline
    
    Args:
        results_dirs: Dict {scale: dir_path} com dados de múltiplas escalas
        output_dir: Diretório de saída para relatórios
    
    Returns:
        dict: Resultados dos testes estatísticos
    """
    if isinstance(results_dirs, dict):
        scales_data = results_dirs
    else:
        scales_data = {scale: dir_path for scale, dir_path in results_dirs}
    
    if len(scales_data) == 0:
        print("AVISO: Nenhum diretório fornecido para testes estatísticos")
        return {}
    
    # Carrega dados de todas as escalas
    dfs = {}
    for scale_str, dir_path in scales_data.items():
        summary_path = os.path.join(dir_path, 'summary_results.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Scale'] = int(scale_str)
            dfs[scale_str] = df
        else:
            print(f"AVISO: {summary_path} não encontrado, pulando escala {scale_str}")
    
    if len(dfs) == 0:
        print("AVISO: Nenhum dado disponível para testes estatísticos")
        return {}
    
    # Métricas a testar
    metrics = {
        'Final_Hypervolume': {
            'name': 'Hypervolume',
            'better': 'higher',
            'unit': '×10¹²'
        },
        'Net_AEP_GWh': {
            'name': 'Net AEP',
            'better': 'higher',
            'unit': 'GWh'
        },
        'Total_Cost_USD': {
            'name': 'Cabling Cost',
            'better': 'lower',
            'unit': 'USD',
            'scale': 1000  # Convert to kUSD
        },
        'Time_Total_s': {
            'name': 'Execution Time',
            'better': 'lower',
            'unit': 'seconds',
            'scale': 60  # Convert to minutes
        }
    }
    
    all_results = {}
    
    # Abre arquivo de saída
    stats_file = os.path.join(output_dir, 'statistical_tests.txt')
    latex_file = os.path.join(output_dir, 'statistical_tests.tex')
    
    with open(stats_file, 'w') as f_txt, open(latex_file, 'w') as f_tex:
        # Cabeçalho texto
        f_txt.write("="*80 + "\n")
        f_txt.write("STATISTICAL SIGNIFICANCE TESTS - MANN-WHITNEY U TEST\n")
        f_txt.write("="*80 + "\n")
        f_txt.write("Null Hypothesis (H0): The two methods have the same distribution\n")
        f_txt.write("Alternative Hypothesis (H1): The two methods have different distributions\n")
        f_txt.write("Significance Level: α = 0.05\n")
        f_txt.write("p < 0.05: Reject H0 (statistically significant difference)\n")
        f_txt.write("p ≥ 0.05: Fail to reject H0 (no significant difference)\n")
        f_txt.write("="*80 + "\n\n")
        
        # Cabeçalho LaTeX
        f_tex.write("% Statistical Significance Tests - LaTeX Table\n")
        f_tex.write("\\begin{table*}[htbp]\n")
        f_tex.write("\\caption{Statistical significance tests (Mann-Whitney U) comparing optimization methods.}\n")
        f_tex.write("\\label{tab:statistical_tests}\n")
        f_tex.write("\\centering\n")
        f_tex.write("\\resizebox{\\textwidth}{!}{%\n")
        f_tex.write("\\begin{tabular}{llccccl}\n")
        f_tex.write("\\toprule\n")
        f_tex.write("\\textbf{Scale} & \\textbf{Metric} & \\textbf{Baseline} & \\textbf{Proposed} & \\textbf{Sequential} & \\textbf{p-value} & \\textbf{Conclusion} \\\\\n")
        f_tex.write("\\midrule\n")
        
        # Para cada escala
        for scale_str in sorted(dfs.keys(), key=lambda x: int(x)):
            df_scale = dfs[scale_str]
            scale = int(scale_str)
            
            f_txt.write(f"\n{'='*80}\n")
            f_txt.write(f"SCALE: {scale} TURBINES\n")
            f_txt.write(f"{'='*80}\n\n")
            
            scale_results = {}
            
            # Para cada métrica
            for metric_key, metric_info in metrics.items():
                if metric_key not in df_scale.columns:
                    continue
                
                f_txt.write(f"\n{'-'*80}\n")
                f_txt.write(f"Metric: {metric_info['name']}\n")
                f_txt.write(f"{'-'*80}\n")
                
                # Extrai dados por método
                baseline_data = df_scale[df_scale['Method'] == 'Baseline'][metric_key].dropna().values
                proposed_data = df_scale[df_scale['Method'] == 'Proposed'][metric_key].dropna().values
                sequential_data = df_scale[df_scale['Method'] == 'Sequential'][metric_key].dropna().values
                
                # Aplica escala se necessário
                scale_factor = metric_info.get('scale', 1)
                baseline_data = baseline_data / scale_factor
                proposed_data = proposed_data / scale_factor
                sequential_data = sequential_data / scale_factor
                
                # Calcula estatísticas descritivas
                baseline_mean = np.mean(baseline_data) if len(baseline_data) > 0 else 0
                baseline_std = np.std(baseline_data) if len(baseline_data) > 0 else 0
                proposed_mean = np.mean(proposed_data) if len(proposed_data) > 0 else 0
                proposed_std = np.std(proposed_data) if len(proposed_data) > 0 else 0
                sequential_mean = np.mean(sequential_data) if len(sequential_data) > 0 else 0
                sequential_std = np.std(sequential_data) if len(sequential_data) > 0 else 0
                
                f_txt.write(f"Baseline:   {baseline_mean:.4f} ± {baseline_std:.4f} (n={len(baseline_data)})\n")
                f_txt.write(f"Proposed:   {proposed_mean:.4f} ± {proposed_std:.4f} (n={len(proposed_data)})\n")
                f_txt.write(f"Sequential: {sequential_mean:.4f} ± {sequential_std:.4f} (n={len(sequential_data)})\n\n")
                
                # Teste 1: Proposed vs Baseline
                if len(proposed_data) > 0 and len(baseline_data) > 0:
                    try:
                        statistic_pb, p_value_pb = stats.mannwhitneyu(
                            proposed_data, baseline_data, alternative='two-sided'
                        )
                        
                        # Determina se é significativo
                        is_significant_pb = p_value_pb < 0.05
                        significance_pb = "***" if p_value_pb < 0.001 else "**" if p_value_pb < 0.01 else "*" if p_value_pb < 0.05 else "ns"
                        
                        # Determina qual é melhor
                        if metric_info['better'] == 'higher':
                            winner_pb = "Proposed" if proposed_mean > baseline_mean else "Baseline"
                        else:
                            winner_pb = "Proposed" if proposed_mean < baseline_mean else "Baseline"
                        
                        f_txt.write(f"Test: Proposed vs Baseline\n")
                        f_txt.write(f"  Mann-Whitney U statistic: {statistic_pb:.4f}\n")
                        f_txt.write(f"  p-value: {p_value_pb:.6f} {significance_pb}\n")
                        f_txt.write(f"  Significant: {'YES' if is_significant_pb else 'NO'}\n")
                        f_txt.write(f"  Better method: {winner_pb}\n\n")
                        
                        # Salva resultados
                        if scale_str not in scale_results:
                            scale_results[scale_str] = {}
                        scale_results[scale_str][f"{metric_key}_proposed_vs_baseline"] = {
                            'p_value': p_value_pb,
                            'significant': is_significant_pb,
                            'winner': winner_pb
                        }
                        
                        # Adiciona à tabela LaTeX
                        unit_str = f"[{metric_info['unit']}]" if 'scale' in metric_info else f"[{metric_info['unit']}]"
                        conclusion = f"\\textbf{{{winner_pb}}}" if is_significant_pb else "No diff."
                        f_tex.write(f"{scale} & {metric_info['name']} {unit_str} & "
                                  f"${baseline_mean:.2f}\\pm{baseline_std:.2f}$ & "
                                  f"${proposed_mean:.2f}\\pm{proposed_std:.2f}$ & "
                                  f"${sequential_mean:.2f}\\pm{sequential_std:.2f}$ & "
                                  f"{p_value_pb:.4f}{significance_pb} & {conclusion} \\\\\n")
                        
                    except Exception as e:
                        f_txt.write(f"  ERROR: {e}\n\n")
                
                # Teste 2: Sequential vs Baseline
                if len(sequential_data) > 0 and len(baseline_data) > 0:
                    try:
                        statistic_sb, p_value_sb = stats.mannwhitneyu(
                            sequential_data, baseline_data, alternative='two-sided'
                        )
                        
                        is_significant_sb = p_value_sb < 0.05
                        significance_sb = "***" if p_value_sb < 0.001 else "**" if p_value_sb < 0.01 else "*" if p_value_sb < 0.05 else "ns"
                        
                        if metric_info['better'] == 'higher':
                            winner_sb = "Sequential" if sequential_mean > baseline_mean else "Baseline"
                        else:
                            winner_sb = "Sequential" if sequential_mean < baseline_mean else "Baseline"
                        
                        f_txt.write(f"Test: Sequential vs Baseline\n")
                        f_txt.write(f"  Mann-Whitney U statistic: {statistic_sb:.4f}\n")
                        f_txt.write(f"  p-value: {p_value_sb:.6f} {significance_sb}\n")
                        f_txt.write(f"  Significant: {'YES' if is_significant_sb else 'NO'}\n")
                        f_txt.write(f"  Better method: {winner_sb}\n\n")
                        
                        if scale_str not in scale_results:
                            scale_results[scale_str] = {}
                        scale_results[scale_str][f"{metric_key}_sequential_vs_baseline"] = {
                            'p_value': p_value_sb,
                            'significant': is_significant_sb,
                            'winner': winner_sb
                        }
                        
                    except Exception as e:
                        f_txt.write(f"  ERROR: {e}\n\n")
                
                # Teste 3: Proposed vs Sequential
                if len(proposed_data) > 0 and len(sequential_data) > 0:
                    try:
                        statistic_ps, p_value_ps = stats.mannwhitneyu(
                            proposed_data, sequential_data, alternative='two-sided'
                        )
                        
                        is_significant_ps = p_value_ps < 0.05
                        significance_ps = "***" if p_value_ps < 0.001 else "**" if p_value_ps < 0.01 else "*" if p_value_ps < 0.05 else "ns"
                        
                        if metric_info['better'] == 'higher':
                            winner_ps = "Proposed" if proposed_mean > sequential_mean else "Sequential"
                        else:
                            winner_ps = "Proposed" if proposed_mean < sequential_mean else "Sequential"
                        
                        f_txt.write(f"Test: Proposed vs Sequential\n")
                        f_txt.write(f"  Mann-Whitney U statistic: {statistic_ps:.4f}\n")
                        f_txt.write(f"  p-value: {p_value_ps:.6f} {significance_ps}\n")
                        f_txt.write(f"  Significant: {'YES' if is_significant_ps else 'NO'}\n")
                        f_txt.write(f"  Better method: {winner_ps}\n\n")
                        
                        if scale_str not in scale_results:
                            scale_results[scale_str] = {}
                        scale_results[scale_str][f"{metric_key}_proposed_vs_sequential"] = {
                            'p_value': p_value_ps,
                            'significant': is_significant_ps,
                            'winner': winner_ps
                        }
                        
                    except Exception as e:
                        f_txt.write(f"  ERROR: {e}\n\n")
            
            all_results[scale_str] = scale_results
        
        # Fecha tabela LaTeX
        f_tex.write("\\bottomrule\n")
        f_tex.write("\\end{tabular}%\n")
        f_tex.write("}\n")
        f_tex.write("\\begin{tablenotes}\n")
        f_tex.write("\\small\n")
        f_tex.write("\\item Note: Significance levels: *** p\u003c0.001, ** p\u003c0.01, * p\u003c0.05, ns = not significant.\n")
        f_tex.write("\\end{tablenotes}\n")
        f_tex.write("\\end{table*}\n")
    
    print(f"✓ Testes estatísticos salvos em: {stats_file}")
    print(f"✓ Tabela LaTeX salva em: {latex_file}")
    
    return all_results

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main(results_dir, multi_scale_dirs=None):
    """
    Função principal: lê os CSVs e gera todos os gráficos.
    
    Args:
        results_dir: Diretório onde estão os CSVs (ex: 'results_36')
        multi_scale_dirs: Opcional. Dict {scale: dir_path} para análise multi-escala.
                         Ex: {'16': 'results_16', '36': 'results_36', '64': 'results_64'}
                         Se fornecido, gera também gráficos de escalabilidade e qualitativos multi-escala.
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
    
    # 9. Análise multi-escala (se múltiplos diretórios fornecidos)
    if multi_scale_dirs is not None and len(multi_scale_dirs) > 1:
        print("\n>>> Gerando gráficos de escalabilidade multi-escala...")
        plot_scalability_metrics(multi_scale_dirs, output_dir=results_dir)
        
        print("\n>>> Gerando gráficos qualitativos multi-escala...")
        plot_qualitative_metrics_multi_scale(multi_scale_dirs, output_dir=results_dir)
        
        # 10. Gráficos separados estilo publicação (novos)
        print("\n>>> Gerando gráficos de escalabilidade separados (publicação)...")
        plot_scalability_separate(multi_scale_dirs, output_dir=results_dir)
        
        print("\n>>> Gerando gráficos qualitativos separados (publicação)...")
        plot_qualitative_separate(multi_scale_dirs, output_dir=results_dir)
        
        # 11. Tabelas LaTeX (novas)
        print("\n>>> Gerando tabelas LaTeX...")
        generate_latex_tables(multi_scale_dirs, output_dir=results_dir)
        
        # 12. Testes estatísticos (novos)
        print("\n>>> Realizando testes de significância estatística...")
        perform_statistical_tests(multi_scale_dirs, output_dir=results_dir)

    
    print("\n" + "="*80)
    print("TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
    print("="*80)
    print(f"✓ Gráficos salvos em: {os.path.abspath(results_dir)}")

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURAÇÃO HARDCODED - Edite aqui os diretórios de resultados
    # =========================================================================
    
    # Diretório principal onde serão salvos os gráficos e testes estatísticos
    RESULTS_DIR = 'results_36'
    
    # Para análise multi-escala, defina os diretórios aqui:
    # Formato: {escala: caminho_do_diretório}
    # Se você não quer análise multi-escala, deixe como None
    MULTI_SCALE_DIRS = {
        '16': 'results_16',
        '36': 'results_36',
        # '64': 'results_64',  # Descomente se tiver resultados para 64 turbinas
    }
    
    # Se você quer apenas uma única escala (sem comparações multi-escala),
    # use a linha abaixo ao invés da definição acima:
    # MULTI_SCALE_DIRS = None
    
    # =========================================================================
    # Executa o script
    # =========================================================================
    
    print("="*80)
    print("CONFIGURAÇÃO:")
    print(f"  Diretório de saída: {RESULTS_DIR}")
    if MULTI_SCALE_DIRS:
        print(f"  Análise multi-escala: {list(MULTI_SCALE_DIRS.keys())} turbinas")
    else:
        print("  Análise de escala única")
    print("="*80)
    
    main(results_dir=RESULTS_DIR, multi_scale_dirs=MULTI_SCALE_DIRS)
