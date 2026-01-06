"""
Script para gerar figuras de resultados (Seção 4) do artigo.
Gera figuras conforme requisitos do template ACM.

Autor: [Seu Nome]
Data: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# =============================================================================
# CONFIGURAÇÃO TÉCNICA OBRIGATÓRIA (ACM Template)
# =============================================================================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Usa estilo seaborn-v0_8-paper (disponível no matplotlib)
plt.style.use('seaborn-v0_8-paper')

# Configurações adicionais para qualidade de publicação
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 300

# =============================================================================
# IMPORTAÇÕES DE MÓDULOS DO PROJETO
# =============================================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from multi_objetivo.cabling_v3 import analisar_layout_completo

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES
# =============================================================================
# Raio do círculo para cada cenário (em metros)
CIRCLE_RADIUS = {
    16: 1300,
    36: 2000,
    64: 3000
}

# Cores para as strings de cabos (paleta distinta e variada)
def get_cable_colors(n_groups):
    """
    Retorna lista de cores distintas para os grupos de cabos.
    Combina múltiplas paletas e garante que não há repetições.
    """
    # Define cores manualmente para garantir máxima distinção visual
    # Cores escolhidas para serem facilmente distinguíveis
    distinct_colors = [
        '#1f77b4',  # Azul
        '#ff7f0e',  # Laranja
        '#2ca02c',  # Verde
        '#d62728',  # Vermelho
        '#9467bd',  # Roxo
        '#8c564b',  # Marrom
        '#e377c2',  # Rosa
        '#7f7f7f',  # Cinza
        '#bcbd22',  # Amarelo-esverdeado
        '#17becf',  # Ciano
        '#ffbb78',  # Pêssego
        '#98df8a',  # Verde claro
        '#ff9896',  # Rosa claro
        '#c5b0d5',  # Lavanda
        '#c49c94',  # Bege
        '#f7b6d3',  # Rosa pastel
        '#c7c7c7',  # Cinza claro
        '#dbdb8d',  # Amarelo claro
        '#9edae5',  # Azul claro
        '#ffed6f',  # Amarelo
    ]
    
    # Se precisar de mais cores, gera usando HSV uniformemente espaçado
    if n_groups > len(distinct_colors):
        additional = n_groups - len(distinct_colors)
        hsv_colors = plt.cm.hsv(np.linspace(0, 0.9, additional))  # Evita voltar ao vermelho
        from matplotlib.colors import to_hex
        distinct_colors.extend([to_hex(c) for c in hsv_colors])
    
    return distinct_colors[:n_groups]

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_solution_file(filepath):
    """
    Carrega arquivo de solução e retorna coordenadas, n_grupos e posição da subestação.
    
    Args:
        filepath: Caminho para o arquivo solution_*_coords.txt
        
    Returns:
        coords: Array numpy com coordenadas das turbinas (N, 2)
        n_grupos: Número de grupos de cabeamento
        substation_pos: Array numpy com posição da subestação (2,)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extrai coordenadas
    xc_line = [line for line in lines if line.startswith('xc:')][0]
    yc_line = [line for line in lines if line.startswith('yc:')][0]
    
    xc = np.array(eval(xc_line.split(":")[1].strip()))
    yc = np.array(eval(yc_line.split(":")[1].strip()))
    coords = np.column_stack([xc, yc])
    
    # Extrai n_grupos e subestação
    n_grupos = None
    substation_x = None
    substation_y = None
    
    for line in lines:
        if line.startswith('n_grupos:'):
            n_grupos = int(line.split(':')[1].strip())
        elif line.startswith('substation_x:'):
            substation_x = float(line.split(':')[1].strip())
        elif line.startswith('substation_y:'):
            substation_y = float(line.split(':')[1].strip())
    
    substation_pos = None
    if substation_x is not None and substation_y is not None:
        substation_pos = np.array([substation_x, substation_y])
    
    return coords, n_grupos, substation_pos


def knee_point(df):
    """
    Retorna índice do knee point (ponto de joelho) usando método de distância normalizada.
    Prioriza soluções com alto AEP.
    """
    cost = df["Custo_USD"].values
    aep = df["AEP_Liquido_MWh"].values
    
    # Normaliza valores entre 0 e 1
    cost_min, cost_max = cost.min(), cost.max()
    aep_min, aep_max = aep.min(), aep.max()
    
    if cost_max == cost_min:
        cost_n = np.ones_like(cost)
    else:
        cost_n = (cost_max - cost) / (cost_max - cost_min)  # Quanto menor melhor
    
    if aep_max == aep_min:
        aep_n = np.ones_like(aep)
    else:
        aep_n = (aep - aep_min) / (aep_max - aep_min)  # Quanto maior melhor
    
    # Ponto ideal: (AEP máximo normalizado, Custo mínimo normalizado) = (1.0, 1.0)
    ideal_point = np.array([1.0, 1.0])
    
    # Peso: AEP é 3x mais importante que custo
    weight_aep = 3.0
    weight_cost = 1.0
    
    distances = []
    for i in range(len(df)):
        point = np.array([aep_n[i], cost_n[i]])
        dist_aep = weight_aep * abs(ideal_point[0] - point[0])
        dist_cost = weight_cost * abs(ideal_point[1] - point[1])
        dist = np.sqrt(dist_aep**2 + dist_cost**2)
        distances.append(dist)
    
    return np.argmin(distances)


def load_pareto_data(scenario_dir, n_turbines):
    """
    Carrega dados da frente de Pareto para um cenário específico.
    
    Args:
        scenario_dir: Diretório com os dados do cenário
        n_turbines: Número de turbinas (16, 36 ou 64)
        
    Returns:
        df: DataFrame com dados da frente de Pareto
    """
    csv_path = os.path.join(scenario_dir, 'pareto_summary.csv')
    
    if not os.path.exists(csv_path):
        print(f"AVISO: Arquivo {csv_path} não encontrado. Pulando cenário de {n_turbines} turbinas.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Remove duplicatas e valores inválidos
    df = df.dropna(subset=['Custo_USD', 'AEP_Liquido_MWh'])
    df = df[(df['Custo_USD'] > 0) & (df['AEP_Liquido_MWh'] > 0)]
    df = df.drop_duplicates(subset=['Custo_USD', 'AEP_Liquido_MWh'])
    
    return df


# =============================================================================
# FUNÇÃO PRINCIPAL: GRÁFICO DA FRENTE DE PARETO (1x3)
# =============================================================================

def plot_pareto_fronts(base_dir, output_dir):
    """
    Gera figura 1x3 com frentes de Pareto para os 3 cenários.
    Por enquanto, apenas 16 turbinas tem dados disponíveis.
    
    Args:
        base_dir: Diretório base onde estão os dados (pareto_front_results)
        output_dir: Diretório onde salvar as figuras
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scenarios = [16, 36, 64]
    
    for idx, n_turbines in enumerate(scenarios):
        ax = axes[idx]
        
        # Carrega dados do cenário
        # Primeiro tenta diretório específico, depois diretório padrão
        scenario_dir = os.path.join(base_dir, f'pareto_front_results_{n_turbines}')
        if not os.path.exists(scenario_dir):
            scenario_dir = base_dir
        
        df = load_pareto_data(scenario_dir, n_turbines)
        
        if df is None or len(df) == 0:
            # Mostra mensagem para cenários sem dados
            ax.text(0.5, 0.5, f'Data not available\nfor {n_turbines} turbines',
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=11, style='italic', color='gray')
            ax.set_title(f'{n_turbines} Turbines', fontsize=12, fontweight='bold')
            ax.set_xlabel('Total Cabling Cost (Thousands USD)', fontsize=11)
            ax.set_ylabel('Net AEP (GWh)', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.3)
            continue
        
        # Converte AEP para GWh e custo para milhares de USD
        aep_gwh = df["AEP_Liquido_MWh"].values / 1000.0
        cost_thousands = df["Custo_USD"].values / 1000.0
        
        # Plota todas as soluções da frente de Pareto
        ax.scatter(cost_thousands, aep_gwh, s=30, alpha=0.7, 
                  color='navy', edgecolors='darkblue', linewidths=0.3,
                  label=f'Pareto Solutions (n={len(df)})')
        
        # Identifica e destaca o knee point
        knee_idx = knee_point(df)
        knee_cost = cost_thousands[knee_idx]
        knee_aep = aep_gwh[knee_idx]
        
        ax.scatter(knee_cost, knee_aep, s=300, alpha=1.0,
                  color='gold', edgecolors='black', linewidths=2,
                  marker='*', zorder=10, label='Knee Point')
        
        # Configurações do gráfico
        ax.set_xlabel('Total Cabling Cost (Thousands USD)', fontsize=11)
        ax.set_ylabel('Net AEP (GWh)', fontsize=11)
        ax.set_title(f'{n_turbines} Turbines', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=9, frameon=True)
    
    plt.tight_layout()
    
    # Salva em PDF e EPS
    pdf_path = os.path.join(output_dir, 'pareto_fronts_3scenarios.pdf')
    eps_path = os.path.join(output_dir, 'pareto_fronts_3scenarios.eps')
    
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(eps_path, bbox_inches='tight', dpi=300)
    print(f"Figura salva: {pdf_path}")
    print(f"Figura salva: {eps_path}")
    
    plt.close()


# =============================================================================
# FUNÇÃO PRINCIPAL: LAYOUTS OTIMIZADOS (Knee Point)
# =============================================================================

def plot_knee_point_layout(base_dir, output_dir, n_turbines):
    """
    Gera figura do layout otimizado para o knee point de um cenário.
    
    Args:
        base_dir: Diretório base onde estão os dados
        output_dir: Diretório onde salvar as figuras
        n_turbines: Número de turbinas (16, 36 ou 64)
    """
    # Carrega dados do cenário
    scenario_dir = os.path.join(base_dir, f'pareto_front_results_{n_turbines}')
    if not os.path.exists(scenario_dir):
        scenario_dir = base_dir
    
    df = load_pareto_data(scenario_dir, n_turbines)
    
    if df is None or len(df) == 0:
        print(f"AVISO: Nenhum dado disponível para {n_turbines} turbinas.")
        return
    
    # Identifica knee point
    knee_idx = knee_point(df)
    knee_solution = df.iloc[knee_idx]
    knee_file_path = knee_solution["File"]
    
    # Handle relative paths - o CSV já contém "pareto_front_results/" no caminho
    if not os.path.isabs(knee_file_path):
        # O caminho no CSV já é relativo a partir de pareto_front_results/
        # Então precisamos ir um nível acima do base_dir
        script_dir = os.path.dirname(os.path.abspath(__file__))
        knee_file_path = os.path.join(script_dir, knee_file_path)
    
    if not os.path.exists(knee_file_path):
        print(f"ERRO: Arquivo {knee_file_path} não encontrado.")
        # Tenta caminho alternativo: apenas o nome do arquivo no base_dir
        filename = os.path.basename(knee_file_path)
        alt_path = os.path.join(base_dir, filename)
        if os.path.exists(alt_path):
            knee_file_path = alt_path
            print(f"     Usando caminho alternativo: {knee_file_path}")
        else:
            print(f"ERRO: Não foi possível encontrar o arquivo.")
            return
    
    # Carrega dados da solução
    coords, n_grupos, substation_pos = load_solution_file(knee_file_path)
    
    if substation_pos is None:
        print(f"AVISO: Posição da subestação não encontrada. Usando turbina mais próxima do centro.")
        # Fallback: encontra turbina mais próxima do centro
        dist_to_center = np.linalg.norm(coords, axis=1)
        substation_idx = np.argmin(dist_to_center)
        coords_with_sub = coords
    else:
        # Adiciona subestação como ponto extra
        coords_with_sub = np.vstack([coords, substation_pos.reshape(1, 2)])
        substation_idx = len(coords)
    
    # Usa n_grupos do arquivo ou valor padrão
    if n_grupos is None:
        n_grupos = int(np.sqrt(len(coords)))
    
    # Calcula cabeamento
    try:
        planta, resultados = analisar_layout_completo(
            coords_with_sub, sub=substation_idx, n_grupos=n_grupos)
    except Exception as e:
        print(f"ERRO ao calcular cabeamento: {e}")
        return
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Desenha círculo de restrição
    circle_radius = CIRCLE_RADIUS.get(n_turbines, 1300)
    circle = Circle((0, 0), circle_radius, fill=False, linestyle='--',
                   linewidth=2, color='black', label='Park Limit')
    ax.add_patch(circle)
    
    # Desenha cabos (cada string com cor diferente)
    # Obtém cores distintas para todos os grupos
    cable_colors = get_cable_colors(len(planta.paths))
    
    for i, path in enumerate(planta.paths):
        if len(path) > 1:
            valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
            if len(valid_path) > 1:
                x_path = [coords_with_sub[k, 0] for k in valid_path]
                y_path = [coords_with_sub[k, 1] for k in valid_path]
                color = cable_colors[i] if i < len(cable_colors) else plt.cm.hsv(i / len(planta.paths))
                ax.plot(x_path, y_path, '-', linewidth=2.5, color=color,
                       alpha=0.8, zorder=4, label=f'Group {i+1}' if i < 15 else '')
    
    # Desenha turbinas em vermelho com números
    ax.scatter(coords[:, 0], coords[:, 1], s=150, c='red',
              edgecolors='black', linewidths=1.5, zorder=5, label='Turbines')
    
    # Adiciona labels com números das turbinas
    for i in range(len(coords)):
        ax.annotate(f'T{i}', 
                   (coords[i, 0], coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.7),
                   zorder=7)
    
    # Desenha subestação (estrela menor)
    if substation_pos is not None:
        ax.scatter(substation_pos[0], substation_pos[1],
                  marker='*', s=300, c='gold', edgecolors='black',
                  linewidths=2, zorder=6, label='Offshore Substation')
        
        # Adiciona label "SUB" na subestação
        ax.annotate('SUB', 
                   (substation_pos[0], substation_pos[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', 
                           edgecolor='black', alpha=0.8),
                   zorder=7)
    
    # Configurações do gráfico
    ax.set_xlim(-1.2 * circle_radius, 1.2 * circle_radius)
    ax.set_ylim(-1.2 * circle_radius, 1.2 * circle_radius)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    
    # Título com custo em thousands
    cost_thousands = knee_solution["Custo_USD"] / 1000.0
    ax.set_title(f'Optimized Layout - {n_turbines} Turbines (Knee Point)\n'
                f'AEP: {knee_solution["AEP_Liquido_MWh"]/1000:.2f} GWh | '
                f'Cost: ${cost_thousands:.0f}k USD | '
                f'Groups: {n_grupos}',
                fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2, frameon=True)
    
    plt.tight_layout()
    
    # Salva em PDF e EPS
    pdf_path = os.path.join(output_dir, f'layout_knee_point_{n_turbines}turbines.pdf')
    eps_path = os.path.join(output_dir, f'layout_knee_point_{n_turbines}turbines.eps')
    
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(eps_path, bbox_inches='tight', dpi=300)
    print(f"Layout salvo: {pdf_path}")
    print(f"Layout salvo: {eps_path}")
    
    plt.close()


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """
    Função principal: gera todas as figuras do artigo.
    """
    # Define diretórios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'pareto_front_results')
    output_dir = os.path.join(script_dir, 'article_figures')
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GERAÇÃO DE FIGURAS PARA O ARTIGO")
    print("=" * 70)
    
    # 1. Gera gráfico da frente de Pareto (1x3)
    print("\n1. Gerando gráfico da frente de Pareto (3 cenários)...")
    plot_pareto_fronts(base_dir, output_dir)
    
    # 2. Gera layouts otimizados para cada cenário (apenas os disponíveis)
    print("\n2. Gerando layouts otimizados (knee point)...")
    available_scenarios = []
    for n_turbines in [16, 36, 64]:
        scenario_dir = os.path.join(base_dir, f'pareto_front_results_{n_turbines}')
        if not os.path.exists(scenario_dir):
            scenario_dir = base_dir
        df = load_pareto_data(scenario_dir, n_turbines)
        if df is not None and len(df) > 0:
            available_scenarios.append(n_turbines)
            print(f"   Processando cenário de {n_turbines} turbinas...")
            plot_knee_point_layout(base_dir, output_dir, n_turbines)
        else:
            print(f"   Pulando cenário de {n_turbines} turbinas (sem dados)")
    
    if len(available_scenarios) == 0:
        print("   AVISO: Nenhum cenário com dados disponível!")
    
    print("\n" + "=" * 70)
    print("TODAS AS FIGURAS FORAM GERADAS COM SUCESSO!")
    print(f"Figuras salvas em: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

