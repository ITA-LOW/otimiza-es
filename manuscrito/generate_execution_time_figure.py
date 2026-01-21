import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# --- CONFIGURAÇÃO ACM ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

# Dados (média em minutos)
scales = [16, 36, 64]
baseline_times = [10.1, 83.7, 165.1]
proposed_times = [6.9, 49.1, 90.5]
sequential_times = [5.2, 43.3, 84.2]

# Cores
COLORS = {'Baseline': '#1f77b4', 'Proposed': '#2ca02c', 'Sequential': '#d62728'}

# Configuração de fontes maiores para coluna única
FONT_SIZE_LABEL = 18
FONT_SIZE_TITLE = 20
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16

# Criar figura
fig, ax = plt.subplots(figsize=(8, 6))

# Posições dos grupos
x = np.arange(len(scales))
bar_width = 0.28  # Barras mais "gordinhas"
spacing = 0.05  # Espaçamento menor entre grupos (colunas mais próximas)

# Ajustar posições para colunas mais próximas
x_baseline = x - bar_width - spacing/2
x_proposed = x
x_sequential = x + bar_width + spacing/2

# Criar barras
bars1 = ax.bar(x_baseline, baseline_times, bar_width, label='Baseline', 
               color=COLORS['Baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x_proposed, proposed_times, bar_width, label='Proposed', 
               color=COLORS['Proposed'], alpha=0.8, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x_sequential, sequential_times, bar_width, label='Sequential', 
               color=COLORS['Sequential'], alpha=0.8, edgecolor='black', linewidth=1.2)

# Adicionar valores acima das barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Configurar eixos
ax.set_xlabel('Number of Turbines', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax.set_ylabel('Average Execution Time (minutes)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax.set_title('Computational Efficiency Across Problem Scales', 
             fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'{s} Turbines' for s in scales], fontsize=FONT_SIZE_TICK)
ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK, width=1.5, length=6)
ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK, width=1.5, length=6)

# Grid
#ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1)

# Legenda completa com os 3 métodos
ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left', framealpha=0.9, 
          edgecolor='black', frameon=True, fancybox=False)

# Remover bordas superior e direita
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Ajustar layout
plt.tight_layout()

# Salvar
plt.savefig('img/execution_time_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('img/execution_time_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Gráfico gerado: img/execution_time_comparison.png e .pdf")
