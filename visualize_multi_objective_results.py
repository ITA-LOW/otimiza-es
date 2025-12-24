
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Adiciona o diretório pai ao path para encontrar o módulo multi_objetivo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from multi_objetivo.cabling import analisar_layout_completo, plotar_cabeamento

# Create output directory if it doesn't exist
output_dir = 'output/multi_objective_sub'
os.makedirs(output_dir, exist_ok=True)

# Load the data from the JSON file
file_path = f'{output_dir}/pareto_solutions.json'
# Corrigir o caminho do arquivo para ser relativo ao script
pareto_json_path = os.path.join(os.path.dirname(__file__), 'output', 'multi_objective_sub', 'pareto_solutions.json')

with open(pareto_json_path, 'r') as f:
    data = json.load(f)

# Create a pandas DataFrame
df = pd.DataFrame(data)

# --- 1. Pareto Front Plot ---
plt.figure(figsize=(12, 8))
# Convert AEP to GWh for better readability
df['aep_gwh'] = df['aep'] / 1000
scatter = plt.scatter(df['cost_usd'] / 1e6, df['aep_gwh'], c=df['rank'], cmap='viridis_r', s=50, alpha=0.7)
plt.colorbar(scatter, label='Rank')
plt.title('Fronteira de Pareto: Produção de Energia (AEP) vs. Custo', fontsize=16)
plt.xlabel('Custo do Cabeamento (Milhões de USD)', fontsize=12)
plt.ylabel('Produção Anual de Energia (GWh)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Find and highlight interesting points
# Normalize objectives (cost is to be minimized, aep is to be maximized)
# Avoid division by zero if all values are the same
cost_range = df['cost_usd'].max() - df['cost_usd'].min()
aep_range = df['aep_gwh'].max() - df['aep_gwh'].min()

norm_cost = (df['cost_usd'] - df['cost_usd'].min()) / cost_range if cost_range > 0 else np.zeros(len(df))
norm_aep = (df['aep_gwh'] - df['aep_gwh'].min()) / aep_range if aep_range > 0 else np.zeros(len(df))


# The "utopian point" is where cost is min (0) and aep is max (1)
utopian_point = np.array([0, 1])
# Handle cases where norm_cost or norm_aep are NaN (if there's only one solution)
if np.isnan(norm_cost).any() or np.isnan(norm_aep).any():
    compromise_idx = 0
else:
    distances = np.sqrt((norm_cost - utopian_point[0])**2 + (norm_aep - utopian_point[1])**2)
    compromise_idx = distances.idxmin()

max_aep_idx = df['aep_gwh'].idxmax()
min_cost_idx = df['cost_usd'].idxmin()

# Highlight points
plt.scatter(df.loc[max_aep_idx, 'cost_usd'] / 1e6, df.loc[max_aep_idx, 'aep_gwh'],
            color='red', s=150, edgecolors='black', zorder=5, label=f'Maior AEP (ID: {max_aep_idx})')
plt.scatter(df.loc[min_cost_idx, 'cost_usd'] / 1e6, df.loc[min_cost_idx, 'aep_gwh'],
            color='blue', s=150, edgecolors='black', zorder=5, label=f'Menor Custo (ID: {min_cost_idx})')
plt.scatter(df.loc[compromise_idx, 'cost_usd'] / 1e6, df.loc[compromise_idx, 'aep_gwh'],
            color='green', s=150, edgecolors='black', zorder=5, label=f'Solução de Compromisso (ID: {compromise_idx})')

plt.legend(fontsize=12)
pareto_plot_path = os.path.join(output_dir, 'pareto_front_accurate.png')
plt.savefig(pareto_plot_path)
plt.close()

print(f"Gráfico da fronteira de Pareto salvo em: {pareto_plot_path}")

# --- 2. Layout Plots with Accurate Cabling ---
def plot_accurate_layout(solution, solution_id, title_prefix, filename):
    """
    Analyzes and plots a layout with its real cabling route.
    """
    # A subestação é a ÚLTIMA na lista de coordenadas do 'individual' original.
    # No nosso JSON, ela está separada. Vamos juntá-las para a análise.
    turb_coords = np.array(solution['layout_turbines'])
    sub_coords = np.array(solution['layout_substation'])
    
    # O script de cabeamento espera um array onde a subestação é um dos pontos.
    # Vamos adicionar a subestação no início e usar o índice 0.
    all_coords = np.vstack([sub_coords.reshape(1, 2), turb_coords])
    substation_idx = 0
    num_turbines = len(turb_coords)
    
    # Executa a análise de cabeamento
    # Precisamos de alguns parâmetros que estavam no script de otimização
    n_grupos = int(np.sqrt(num_turbines))
    rated_pwr = 3.35e6 # Valor padrão do IEA37, usado no script de otimização
    
    planta, resultados = analisar_layout_completo(
        coordenadas=all_coords,
        substation_idx=substation_idx,
        n_grupos=n_grupos,
        P_turbina=rated_pwr
    )
    
    # Gera o título para o gráfico
    titulo_plot = (
        f'{title_prefix} (ID: {solution_id})\n'
        f'AEP: {solution["aep_gwh"]:.2f} GWh, Custo: ${solution["cost_usd"]/1e6:.2f}M\n'
        f'Custo Recalculado: ${resultados["custo_total_usd"]/1e6:.2f}M | '
        f'Perda Anual: {resultados["perda_anual_mwh"]:.2f} MWh'
    )
    
    # Plota o cabeamento real
    plotar_cabeamento(planta, all_coords, substation_idx, titulo=titulo_plot, output_filename=filename)

# Plot layout for the 3 selected solutions with accurate cabling
solution_max_aep = df.loc[max_aep_idx]
plot_accurate_layout(solution_max_aep, max_aep_idx,
            'Layout para Maior AEP',
            os.path.join(output_dir, 'layout_max_aep_accurate.png'))

solution_min_cost = df.loc[min_cost_idx]
plot_accurate_layout(solution_min_cost, min_cost_idx,
            'Layout para Menor Custo',
            os.path.join(output_dir, 'layout_min_cost_accurate.png'))

solution_compromise = df.loc[compromise_idx]
plot_accurate_layout(solution_compromise, compromise_idx,
            'Layout para Solução de Compromisso',
            os.path.join(output_dir, 'layout_compromise_accurate.png'))

# --- 3. Parallel Coordinates Plot ---
plt.figure(figsize=(12, 8))
# Normalizando os dados para o plot de coordenadas paralelas para melhor visualização
df_norm = df.copy()
df_norm['AEP (GWh)'] = (df['aep_gwh'] - df['aep_gwh'].min()) / (df['aep_gwh'].max() - df['aep_gwh'].min())
df_norm['Custo (M$)'] = (df['cost_usd'] - df['cost_usd'].min()) / (df['cost_usd'].max() - df['cost_usd'].min())


pd.plotting.parallel_coordinates(
    df_norm[['AEP (GWh)', 'Custo (M$)', 'rank']],
    'rank',
    colormap='viridis_r'
)
plt.title('Gráfico de Coordenadas Paralelas das Soluções (Normalizado)', fontsize=16)
plt.xlabel('Objetivos', fontsize=12)
plt.ylabel('Valores Normalizados', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
parallel_plot_path = os.path.join(output_dir, 'parallel_coordinates_accurate.png')
plt.savefig(parallel_plot_path)
plt.close()

print(f"Gráfico de coordenadas paralelas salvo em: {parallel_plot_path}")
