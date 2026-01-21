import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# --- CONFIGURAÇÃO ACM ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

# Estilos
COLORS = {'Proposed': '#2ca02c', 'Baseline': '#1f77b4', 'Sequential': '#d62728'}
FONT_SIZE_LABEL = 18
FONT_SIZE_TITLE = 20
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 18

# --- DADOS ---
path_csv = 'results_16/case_study_results.csv'
df = pd.read_csv(path_csv)

def get_pareto(df_sub):
    costs = df_sub['Custo_Total_USD'].values
    aeps = df_sub['AEP_Liquido_MWh'].values
    pareto_indices = []
    for i in range(len(df_sub)):
        is_dominated = np.any((costs <= costs[i]) & (aeps >= aeps[i]) & ((costs < costs[i]) | (aeps > aeps[i])))
        if not is_dominated: pareto_indices.append(i)
    return df_sub.iloc[pareto_indices].sort_values('Custo_Total_USD')

# Criar a figura com dois subplots (lados) compartilhando o eixo Y
fig = plt.figure(figsize=(10, 5))
# O subplot da direita (high cost) pode ser um pouco maior se tiver mais pontos
gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.05) 
ax1 = fig.add_subplot(gs[0]) # Lado esquerdo (Baseline)
ax2 = fig.add_subplot(gs[1], sharey=ax1) # Lado direito (Proposed/Seq)

for method_name in ['Baseline', 'Proposed', 'Sequential']:
    df_m = df[df['Method'] == method_name]
    pareto = get_pareto(df_m)
    
    x = pareto['Custo_Total_USD'] / 1000
    y = pareto['AEP_Liquido_MWh'] / 1000
    
    # Plotar em ambos os eixos (eles só aparecerão onde o limite permitir)
    for ax in [ax1, ax2]:
        ax.plot(x, y, color=COLORS[method_name], linestyle='--', alpha=0.3)
        ax.scatter(x, y, color=COLORS[method_name], s=30, alpha=0.6, label=method_name if ax == ax2 else "")
        # Knee point (aproximado para o plot)
        knee_idx = np.argmin(np.sqrt(((x-x.min())/(x.max()-x.min()+1))**2 + (1-(y-y.min())/(y.max()-y.min()+1))**2))
        ax.scatter(x.iloc[knee_idx], y.iloc[knee_idx], color=COLORS[method_name], s=100, edgecolors='black', zorder=5)

# --- CONFIGURAR O "CORTE" DO EIXO ---
# Limites para esconder o vazio (ajuste conforme seus dados)
ax1.set_xlim(140, 250) # Foco na Baseline
ax2.set_xlim(500, 850) # Foco no Proposed/Seq

# Definir ticks específicos para evitar sobreposição nas bordas do corte
ax1.set_xticks([150, 175, 200, 225])
ax2.set_xticks([550, 600, 650, 700, 750, 800, 850])

# Esconder as bordas internas
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
#ax2.yaxis.tick_right() # Opcional: manter ticks só na esquerda
ax2.tick_params(labelleft=False, left=False)

# Adicionar as marcas de "corte" (slashes //)
d = .015 
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # Top-right diagonal
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)    # Bottom-right diagonal

kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d/2, +d/2), (-d, +d), **kwargs)      # Top-left diagonal
ax2.plot((-d/2, +d/2), (1-d, 1+d), **kwargs)  # Bottom-left diagonal

# Labels e Título
fig.text(0.5, 0.01, 'Total Cabling Cost (kUSD)', ha='center', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Net Annual Energy Production (GWh)', fontsize=FONT_SIZE_LABEL)
plt.suptitle('Aggregated Pareto Fronts', fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.98)

# Configurar tamanho dos ticks
ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK)

ax2.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, linestyle=':', alpha=0.5)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.savefig('pareto_compacto.pdf', format='pdf', bbox_inches='tight')
plt.savefig('pareto_compacto.png', dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- CONFIGURAÇÃO ACM (FONTES TYPE 42) ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

# Configurações de Tamanho (Aumentadas para legibilidade em subplots)
LBL_SIZE = 18
TICK_SIZE = 16
LEG_SIZE = 14
TITLE_SIZE = 18

def plot_hv_comparison(csv_path, output_name, title):
    df = pd.read_csv(csv_path)
    
    # Se houver uma coluna para filtrar o número de turbinas, use-a. 
    # Caso contrário, assumimos que você está passando o CSV específico da escala.
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Cores padrão do artigo
    colors = {'Proposed': '#d62728', 'Baseline': '#1f77b4'}
    
    for method in ['Proposed', 'Baseline']:
        df_method = df[df['Method'] == method]
        
        # Agrupar por geração para calcular Média e Desvio Padrão entre os Runs
        # Nota: Ajuste 'N_Generations' se o nome da coluna for diferente
        stats = df_method.groupby('N_Generations')['Hypervolume'].agg(['mean', 'std']).reset_index()
        
        # Se for Proposed, talvez queira plotar apenas até a geração 1000 
        # (conforme discutimos que a P1 não conta HV)
        x = stats['N_Generations']
        y_mean = stats['mean']
        y_std = stats['std']

        # Plot da Média
        plt.plot(x, y_mean, label=f'{method} (Mean)', color=colors[method], lw=2.5)
        
        # Plot do Desvio Padrão (Sombra)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, 
                         color=colors[method], alpha=0.2, label=f'{method} ±1 Std Dev')

    # Configurações de Eixo
    plt.title(title, fontsize=TITLE_SIZE, fontweight='bold')
    plt.xlabel('Generation (Phase 2)', fontsize=LBL_SIZE)
    plt.ylabel('Hypervolume', fontsize=LBL_SIZE)
    
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    
    # Formatação científica no eixo Y (comum para HV que tem valores altos)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=LEG_SIZE, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- EXECUÇÃO ---
# Se você tiver 3 arquivos diferentes:
plot_hv_comparison('results_16/case_study_results.csv', 'hv_16', '16 Turbines')
# plot_hv_comparison('results_36/case_study_results.csv', 'hv_36', '36 Turbines')
# plot_hv_comparison('results_64/case_study_results.csv', 'hv_64', '64 Turbines')   