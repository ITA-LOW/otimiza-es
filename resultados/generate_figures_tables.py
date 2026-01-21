import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- CONFIGURAÇÃO ACM ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

# Estilos com letras maiores
COLORS = {'Proposed': '#2ca02c', 'Baseline': '#1f77b4', 'Sequential': '#d62728'}
FONT_SIZE_LABEL = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 18

# Carregar todos os dados
dfs = {}
for scale in ['16', '36', '64']:
    df = pd.read_csv(f'results_{scale}/case_study_results.csv')
    df['Scale'] = int(scale)
    df['AEP_Liquido_GWh'] = df['AEP_Liquido_MWh'] / 1000
    df['Custo_Total_kUSD'] = df['Custo_Total_USD'] / 1000
    df['Time_Minutes'] = df['Time_Total_seconds'] / 60
    df['Substation_Dist'] = np.sqrt(df['Substation_X_m']**2 + df['Substation_Y_m']**2)
    df['Losses_Pct'] = (df['Perdas_Joule_MWh'] / df['AEP_Bruto_MWh'] * 100)
    dfs[scale] = df

# =============================================================================
# GRÁFICOS SEPARADOS DE ESCALABILIDADE
# =============================================================================
def plot_scalability_separate():
    """Gráficos separados para cada métrica de escalabilidade"""
    scales = [16, 36, 64]
    
    metrics = {
        'net_aep': {
            'col': 'AEP_Liquido_GWh',
            'ylabel': 'Net AEP (GWh)',
            'title': 'Net AEP Scalability'
        },
        'capex': {
            'col': 'Custo_Total_kUSD',
            'ylabel': 'CAPEX (kUSD)',
            'title': 'CAPEX Scalability'
        },
        'execution_time': {
            'col': 'Time_Minutes',
            'ylabel': 'Execution Time (minutes)',
            'title': 'Computational Efficiency Scalability'
        }
    }
    
    for metric_key, metric_info in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            means = []
            stds = []
            valid_scales = []
            
            for scale in scales:
                df_scale = dfs[str(scale)]
                data = df_scale[df_scale['Method'] == method]
                
                if len(data) > 0:
                    values = data[metric_info['col']].values
                    if len(values) > 0:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                        valid_scales.append(scale)
            
            if len(means) > 0:
                markers = {'Baseline': 'o', 'Proposed': 's', 'Sequential': '^'}
                ax.errorbar(valid_scales, means, yerr=stds, 
                           label=method, color=COLORS[method], 
                           marker=markers[method],
                           markersize=12, linewidth=3, capsize=8, capthick=2.5,
                           elinewidth=2.5)
        
        ax.set_xlabel('Number of Turbines', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel(metric_info['ylabel'], fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(metric_info['title'], fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
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
        plt.savefig(f'scalability_{metric_key}.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico gerado: scalability_{metric_key}.png")
        plt.close()

# =============================================================================
# GRÁFICOS DE CARACTERÍSTICAS QUALITATIVAS SEPARADOS
# =============================================================================
def plot_qualitative_separate():
    """Gráficos separados para características qualitativas"""
    characteristics = {
        'substation_distance': {
            'col': 'Substation_Dist',
            'ylabel': 'Substation Distance (m)',
            'title': 'Substation Distance by Method and Scale'
        },
        'cable_length': {
            'col': 'Comprimento_Total_km',
            'ylabel': 'Total Cable Length (km)',
            'title': 'Cable Length by Method and Scale'
        },
        'cable_groups': {
            'col': 'N_Grupos',
            'ylabel': 'Number of Cable Groups',
            'title': 'Cable Groups by Method and Scale'
        },
        'electrical_losses': {
            'col': 'Losses_Pct',
            'ylabel': 'Electrical Losses (%)',
            'title': 'Electrical Losses by Method and Scale'
        }
    }
    
    scales = ['16', '36', '64']
    x = np.arange(len(scales))
    width = 0.25
    
    for char_key, char_info in characteristics.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for i, method in enumerate(['Baseline', 'Proposed', 'Sequential']):
            means = []
            stds = []
            
            for scale in scales:
                df_scale = dfs[scale]
                data = df_scale[df_scale['Method'] == method]
                
                if len(data) > 0 and char_info['col'] in data.columns:
                    values = data[char_info['col']].values
                    if len(values) > 0:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i*width, means, width, label=method, color=COLORS[method], 
                  alpha=0.8, yerr=stds, capsize=8, error_kw={'linewidth': 2.5, 'capthick': 2.5})
        
        ax.set_xlabel('Number of Turbines', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel(char_info['ylabel'], fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(char_info['title'], fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scales)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y', linewidth=1.5)
        ax.tick_params(labelsize=FONT_SIZE_TICK, width=1.5, length=6)
        
        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        plt.savefig(f'qualitative_{char_key}.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráfico gerado: qualitative_{char_key}.png")
        plt.close()

# =============================================================================
# GERAR TABELAS EM LATEX
# =============================================================================
def generate_latex_tables():
    """Gera tabelas LaTeX com dados resumidos"""
    
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
    
    for scale in ['16', '36', '64']:
        df_scale = dfs[scale]
        latex_quant += f"\\multirow{{3}}{{*}}{{{scale}}}\n"
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            data = df_scale[df_scale['Method'] == method]
            if len(data) > 0:
                aep_mean = data['AEP_Liquido_GWh'].mean()
                aep_std = data['AEP_Liquido_GWh'].std()
                cost_mean = data['Custo_Total_kUSD'].mean()
                cost_std = data['Custo_Total_kUSD'].std()
                time_mean = data['Time_Minutes'].mean()
                time_std = data['Time_Minutes'].std()
                
                latex_quant += f" & {method:12s} & ${aep_mean:.2f} \\pm {aep_std:.2f}$ & ${cost_mean:.0f} \\pm {cost_std:.0f}$ & ${time_mean:.1f} \\pm {time_std:.1f}$ \\\\\n"
        
        if scale != '64':
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
    
    for scale in ['16', '36', '64']:
        df_scale = dfs[scale]
        latex_qual += f"\\multirow{{3}}{{*}}{{{scale}}}\n"
        
        for method in ['Baseline', 'Proposed', 'Sequential']:
            data = df_scale[df_scale['Method'] == method]
            if len(data) > 0:
                subst_mean = data['Substation_Dist'].mean()
                subst_std = data['Substation_Dist'].std()
                cable_mean = data['Comprimento_Total_km'].mean()
                cable_std = data['Comprimento_Total_km'].std()
                groups_mean = data['N_Grupos'].mean()
                groups_std = data['N_Grupos'].std()
                losses_mean = data['Losses_Pct'].mean()
                losses_std = data['Losses_Pct'].std()
                
                latex_qual += f" & {method:12s} & ${subst_mean:.0f} \\pm {subst_std:.0f}$ & ${cable_mean:.2f} \\pm {cable_std:.2f}$ & ${groups_mean:.1f} \\pm {groups_std:.1f}$ & ${losses_mean:.2f} \\pm {losses_std:.2f}$ \\\\\n"
        
        if scale != '64':
            latex_qual += "\\midrule\n"
    
    latex_qual += """\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""
    
    # Salvar tabelas
    with open('table_quantitative_metrics.tex', 'w') as f:
        f.write(latex_quant)
    print("✓ Tabela gerada: table_quantitative_metrics.tex")
    
    with open('table_qualitative_metrics.tex', 'w') as f:
        f.write(latex_qual)
    print("✓ Tabela gerada: table_qualitative_metrics.tex")
    
    return latex_quant, latex_qual

# =============================================================================
# EXECUÇÃO
# =============================================================================
print("=" * 80)
print("GERANDO GRÁFICOS E TABELAS")
print("=" * 80)
print()

print("1. Gerando gráficos de escalabilidade separados...")
plot_scalability_separate()
print()

print("2. Gerando gráficos de características qualitativas separados...")
plot_qualitative_separate()
print()

print("3. Gerando tabelas LaTeX...")
generate_latex_tables()
print()

print("=" * 80)
print("✓ TODOS OS ARQUIVOS GERADOS COM SUCESSO!")
print("=" * 80)

