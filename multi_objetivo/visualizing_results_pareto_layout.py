import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.animation as animation
import sys
import os
import glob
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from multi_objetivo.cabling_v3 import analisar_layout_completo

# ============================
# CONFIGURAÇÕES DE ANIMAÇÃO
# ============================

CIRCLE_RADIUS = 1300
IND_SIZE = 16
EVOLUTION_DIR_PHASE1 = "pareto_front_results/evolution_phase1"
EVOLUTION_DIR_PHASE2 = "pareto_front_results/evolution"
SUBSTATION_CONTINENT = np.array([[-1.0, -1350.0]])

# ============================
# FUNÇÕES DE ANIMAÇÃO
# ============================

def load_generation_file(filename):
    """Carrega coordenadas e métricas de um arquivo de geração."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extrai coordenadas
    xc_line = [line for line in lines if line.startswith('xc:')][0]
    yc_line = [line for line in lines if line.startswith('yc:')][0]
    
    xc = np.array(eval(xc_line.split(":")[1].strip()))
    yc = np.array(eval(yc_line.split(":")[1].strip()))
    coords = np.column_stack([xc, yc])
    
    # Extrai métricas
    aep = None
    cost = None
    phase = 1  # Default
    n_grupos = None
    substation_x = None
    substation_y = None
    for line in lines:
        if line.startswith('aep:'):
            aep = float(line.split(':')[1].strip())
        elif line.startswith('cost:'):
            cost = float(line.split(':')[1].strip())
        elif line.startswith('phase:'):
            phase = int(line.split(':')[1].strip())
        elif line.startswith('n_grupos:'):
            n_grupos = int(line.split(':')[1].strip())
        elif line.startswith('substation_x:'):
            substation_x = float(line.split(':')[1].strip())
        elif line.startswith('substation_y:'):
            substation_y = float(line.split(':')[1].strip())
    
    substation_pos = None
    if substation_x is not None and substation_y is not None:
        substation_pos = np.array([substation_x, substation_y])
    
    return coords, aep, cost, phase, n_grupos, substation_pos

def get_generation_files(project_root):
    """Retorna lista de arquivos de geração ordenados (Fase 1 + Fase 2)."""
    files = []
    
    # Adiciona arquivos da Fase 1
    phase1_dir = os.path.join(project_root, EVOLUTION_DIR_PHASE1)
    if os.path.exists(phase1_dir):
        pattern_phase1 = os.path.join(phase1_dir, "gen_*.txt")
        files.extend(glob.glob(pattern_phase1))
    
    # Adiciona arquivos da Fase 2
    phase2_dir = os.path.join(project_root, EVOLUTION_DIR_PHASE2)
    if os.path.exists(phase2_dir):
        pattern_phase2 = os.path.join(phase2_dir, "gen_*.txt")
        files.extend(glob.glob(pattern_phase2))
    
    # Ordena por número da geração
    def get_gen_number(filename):
        match = re.search(r'gen_(\d+)_', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_gen_number)
    return files

def create_evolution_animation(project_root):
    """Cria e mostra animação da evolução."""
    files = get_generation_files(project_root)
    
    if len(files) == 0:
        print(f"AVISO: Nenhum arquivo de evolução encontrado em {EVOLUTION_DIR_PHASE1} ou {EVOLUTION_DIR_PHASE2}")
        print("Pulando animação...")
        return None
    
    print(f"Encontrados {len(files)} arquivos de geração")
    print("Carregando dados e calculando cabeamento para Fase 2...")
    
    # Carrega todos os dados
    all_coords = []
    all_aep = []
    all_cost = []
    all_gen_numbers = []
    all_phases = []
    all_n_grupos = []
    all_cabling_paths = []
    all_substation_pos = []  # Armazena posições das subestações otimizadas pelo GA
    all_coords_with_sub = []  # Armazena coordenadas incluindo subestação (para plot do cabeamento)
    
    for filename in files:
        coords, aep, cost, phase, n_grupos, substation_pos = load_generation_file(filename)
        all_coords.append(coords)
        all_aep.append(aep)
        all_cost.append(cost)
        all_phases.append(phase)
        all_n_grupos.append(n_grupos)
        all_substation_pos.append(substation_pos)  # Armazena posição da subestação
        
        # Extrai número da geração do nome do arquivo
        match = re.search(r'gen_(\d+)_', filename)
        gen_num = int(match.group(1)) if match else 0
        all_gen_numbers.append(gen_num)
        
        # Calcula cabeamento para Fase 2
        cabling_paths = None
        coords_with_sub = coords  # Default: sem subestação extra
        if phase == 2:
            try:
                # Adiciona a subestação otimizada pelo GA como um ponto extra no array de coordenadas
                # Isso permite que o cabeamento vá até a posição exata da subestação
                if substation_pos is not None:
                    # Cria array com turbinas + subestação
                    coords_with_sub = np.vstack([coords, substation_pos.reshape(1, 2)])
                    # O índice da subestação será o último (len(coords))
                    substation_idx = len(coords)
                else:
                    # Fallback: encontra turbina mais próxima do continente (arquivos antigos)
                    distancias_ao_continente = np.linalg.norm(coords - SUBSTATION_CONTINENT, axis=1)
                    ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
                    coords_with_sub = coords
                    substation_idx = ponto_de_coleta_idx
                
                # Usa n_grupos do arquivo ou valor padrão
                if n_grupos is not None:
                    n_grupos_to_use = n_grupos
                else:
                    n_grupos_to_use = int(np.sqrt(IND_SIZE))
                
                # Calcula cabeamento usando a subestação como ponto final
                planta, _ = analisar_layout_completo(
                    coords_with_sub, sub=substation_idx, n_grupos=n_grupos_to_use)
                cabling_paths = planta.paths
            except Exception as e:
                print(f"Erro ao calcular cabeamento: {e}")
                cabling_paths = None
        
        all_cabling_paths.append(cabling_paths)
        all_coords_with_sub.append(coords_with_sub)  # Armazena para uso no plot
        
        # Mostra progresso a cada 100 arquivos
        if len(all_coords) % 100 == 0:
            print(f"  Processados {len(all_coords)}/{len(files)} arquivos...")
    
    # Configura figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Evolução do Algoritmo Genético', fontsize=16, fontweight='bold')
    
    # Configuração do gráfico de layout
    ax1.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax1.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Círculo de restrição
    circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                    linewidth=2, color='black', label='Limite do Parque')
    ax1.add_patch(circle)
    
    # Configuração do gráfico de métricas
    ax2.set_xlabel('Geração', fontsize=12)
    ax2.set_ylabel('Valor', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Evolução das Métricas', fontsize=14)
    
    # Inicializa elementos
    scatter = ax1.scatter([], [], s=150, c='red', edgecolors='black', 
                          linewidths=1.5, zorder=5, label='Turbinas')
    title_text = ax1.set_title('', fontsize=14, fontweight='bold')
    
    # Lista para armazenar linhas de cabeamento e marcadores de subestação
    cabling_lines_container = []
    substation_markers_container = []
    
    # Linhas de evolução
    line_aep, = ax2.plot([], [], 'b-', linewidth=2, label='AEP (GWh)', marker='o', markersize=4)
    
    # Cria eixos secundários para custo
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Custo (USD)', fontsize=12, color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # Linha de custo no eixo secundário
    line_cost, = ax2_twin.plot([], [], 'r-', linewidth=2, label='Custo (USD)', marker='s', markersize=4, color='red')
    
    # Legenda combinada
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Calcula limites reais do custo para escala correta
    phase2_costs = [c for c, p in zip(all_cost, all_phases) if p == 2 and c is not None]
    if len(phase2_costs) > 0:
        cost_min_real = min(phase2_costs)
        cost_max_real = max(phase2_costs)
        cost_range = cost_max_real - cost_min_real
        cost_ymin = cost_min_real - cost_range * 0.02
        cost_ymax = cost_max_real + cost_range * 0.02
    else:
        cost_ymin = None
        cost_ymax = None
    
    def animate(frame):
        """Função de animação para cada frame."""
        if frame >= len(all_coords):
            return scatter, title_text, line_aep, line_cost
        
        coords = all_coords[frame]
        coords_with_sub = all_coords_with_sub[frame]
        aep = all_aep[frame]
        cost = all_cost[frame]
        gen_num = all_gen_numbers[frame]
        phase = all_phases[frame]
        cabling_paths = all_cabling_paths[frame]
        substation_pos = all_substation_pos[frame]
        
        # Remove linhas de cabeamento anteriores
        for line in cabling_lines_container:
            try:
                line.remove()
            except:
                pass
        cabling_lines_container.clear()
        
        # Remove marcadores de subestação anteriores
        for marker in substation_markers_container:
            try:
                marker.remove()
            except:
                pass
        substation_markers_container.clear()
        
        # Desenha subestação otimizada pelo GA (apenas Fase 2)
        if phase == 2 and substation_pos is not None:
            substation_marker = ax1.scatter(
                substation_pos[0], substation_pos[1],
                marker='*', s=400, c='gold', edgecolors='black',
                linewidths=2, zorder=6, label='Subestação (GA)'
            )
            substation_markers_container.append(substation_marker)
        
        # Desenha linhas de cabeamento (apenas Fase 2)
        # Usa coords_with_sub para incluir a subestação nos paths
        if phase == 2 and cabling_paths is not None and len(cabling_paths) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(cabling_paths), 10)))
            for i, path in enumerate(cabling_paths):
                if len(path) > 1:
                    # Valida índices considerando coords_with_sub (que inclui a subestação)
                    valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
                    if len(valid_path) > 1:
                        x_path = [coords_with_sub[k, 0] for k in valid_path]
                        y_path = [coords_with_sub[k, 1] for k in valid_path]
                        line, = ax1.plot(x_path, y_path, '-', linewidth=2.5, 
                                        color=colors[i % len(colors)], alpha=0.8, zorder=4)
                        cabling_lines_container.append(line)
        
        # Atualiza layout
        scatter.set_offsets(coords)
        
        # Título muda conforme a fase
        if phase == 1:
            title_text.set_text(f'Fase 1 - Geração {gen_num} | AEP Bruto: {aep/1000:.2f} GWh')
        else:
            n_grupos_str = f" | Grupos: {all_n_grupos[frame]}" if all_n_grupos[frame] is not None else ""
            cost_str = f"${cost:,.0f}" if cost is not None else "N/A"
            title_text.set_text(f'Fase 2 - Geração {gen_num} | AEP: {aep/1000:.2f} GWh | Custo: {cost_str}{n_grupos_str}')
        
        # Atualiza gráfico de evolução
        gens_so_far = all_gen_numbers[:frame+1]
        phases_so_far = all_phases[:frame+1]
        aeps_so_far = [a/1000 for a in all_aep[:frame+1]]
        
        # AEP sempre visível
        line_aep.set_data(gens_so_far, aeps_so_far)
        
        # Custo apenas na Fase 2
        phase2_indices = [i for i, p in enumerate(phases_so_far) if p == 2]
        if len(phase2_indices) > 0:
            phase2_gens = []
            phase2_costs = []
            for i in phase2_indices:
                if all_cost[i] is not None:
                    phase2_gens.append(gens_so_far[i])
                    phase2_costs.append(all_cost[i])
            
            if len(phase2_costs) > 0:
                line_cost.set_data(phase2_gens, phase2_costs)
                line_cost.set_visible(True)
                if cost_ymin is not None and cost_ymax is not None:
                    ax2_twin.set_ylim(cost_ymin, cost_ymax)
            else:
                line_cost.set_data([], [])
                line_cost.set_visible(False)
        else:
            line_cost.set_data([], [])
            line_cost.set_visible(False)
        
        # Ajusta limites dos eixos
        if len(gens_so_far) > 0:
            ax2.set_xlim(0, max(all_gen_numbers) + 10)
            if len(aeps_so_far) > 0:
                ax2.set_ylim(min(aeps_so_far) * 0.99, max(aeps_so_far) * 1.01)
        
        # Atualiza legenda quando necessário
        # Na Fase 2, mostra turbinas e subestação; na Fase 1, apenas turbinas
        if phase == 2 and substation_pos is not None:
            # Remove legenda antiga se existir
            if ax1.get_legend() is not None:
                ax1.get_legend().remove()
            # Cria nova legenda com turbinas e subestação
            ax1.legend(loc='upper right', fontsize=9)
        elif phase == 1:
            # Remove legenda antiga se existir
            if ax1.get_legend() is not None:
                ax1.get_legend().remove()
            # Cria legenda apenas com turbinas
            ax1.legend(loc='upper right', fontsize=9)
        
        return scatter, title_text, line_aep, line_cost
    
    # Cria animação
    print("Criando animação...")
    anim = animation.FuncAnimation(fig, animate, frames=len(all_coords), 
                                   interval=100, blit=False, repeat=True)
    
    return anim, fig

# ============================
# CONFIGURAÇÕES DE ESTILO (PAPER)
# ============================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_solution(fname):
    """Carrega coordenadas de um arquivo de solução."""
    with open(fname) as f:
        lines = f.readlines()
    xc = np.array(eval(lines[0].split(":")[1]))
    yc = np.array(eval(lines[1].split(":")[1]))
    return np.column_stack([xc, yc])

def load_solution_with_substation(fname):
    """Carrega coordenadas, número de grupos e posição da subestação de um arquivo de solução."""
    with open(fname) as f:
        lines = f.readlines()
    
    xc = np.array(eval(lines[0].split(":")[1]))
    yc = np.array(eval(lines[1].split(":")[1]))
    coords = np.column_stack([xc, yc])
    
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
    Retorna índice do knee point (ponto de joelho).
    Prioriza AEP: se uma solução tem AEP significativamente maior (>5%),
    ela é preferida mesmo com custo um pouco maior.
    Caso contrário, usa distância normalizada ponderada (AEP 3x mais importante).
    """
    cost = df["Custo_USD"].values
    aep = df["AEP_Liquido_MWh"].values

    # Encontra melhor AEP e menor custo
    best_aep_idx = np.argmax(aep)
    best_aep = aep[best_aep_idx]
    min_cost = cost.min()
    
    # Se há solução com AEP muito próximo do máximo (dentro de 5%), prioriza ela
    # mesmo que tenha custo um pouco maior
    threshold_aep = best_aep * 0.95  # 95% do melhor AEP
    
    candidates_high_aep = []
    for i in range(len(df)):
        if aep[i] >= threshold_aep:
            # Calcula custo relativo: quanto % mais caro que o mínimo
            cost_increase_pct = (cost[i] - min_cost) / min_cost if min_cost > 0 else 0
            # Calcula AEP relativo: quanto % do melhor AEP
            aep_pct = aep[i] / best_aep if best_aep > 0 else 0
            
            # Score: prioriza AEP alto, penaliza custo alto
            # Se custo aumenta menos de 20%, vale a pena pelo AEP alto
            if cost_increase_pct < 0.20:  # Custo até 20% maior que o mínimo
                score = aep_pct - cost_increase_pct * 0.5  # Penaliza custo, mas menos
            else:
                score = aep_pct - cost_increase_pct  # Penaliza mais se custo muito alto
            
            candidates_high_aep.append((score, i, aep[i], cost[i]))
    
    # Se há candidatos com AEP alto, escolhe o melhor score
    if len(candidates_high_aep) > 0:
        candidates_high_aep.sort(reverse=True)  # Ordena por score (maior primeiro)
        return candidates_high_aep[0][1]  # Retorna índice do melhor
    
    # Fallback: usa método de distância normalizada ponderada
    cost_min, cost_max = cost.min(), cost.max()
    aep_min, aep_max = aep.min(), aep.max()
    
    if cost_max == cost_min:
        cost_n = np.ones_like(cost)
    else:
        cost_n = (cost_max - cost) / (cost_max - cost_min)
    
    if aep_max == aep_min:
        aep_n = np.ones_like(aep)
    else:
        aep_n = (aep - aep_min) / (aep_max - aep_min)

    ideal_point = np.array([1.0, 1.0])
    weight_aep = 3.0  # AEP é 3x mais importante que custo
    weight_cost = 1.0
    
    distances = []
    for i in range(len(df)):
        point = np.array([aep_n[i], cost_n[i]])
        dist_aep = weight_aep * abs(ideal_point[0] - point[0])
        dist_cost = weight_cost * abs(ideal_point[1] - point[1])
        dist = np.sqrt(dist_aep**2 + dist_cost**2)
        distances.append(dist)
    
    return np.argmin(distances)


def plot_layout(ax, coords, title):
    SUB = np.array([[-1350, 0]])
    sub_idx = np.argmin(np.linalg.norm(coords - SUB, axis=1))

    plant, _ = analisar_layout_completo(coords, sub=sub_idx)

    cmap = plt.colormaps["tab10"]
    for i, path in enumerate(plant.paths):
        x = coords[path, 0]
        y = coords[path, 1]
        ax.plot(x, y, "-o", lw=1.5, ms=4, color=cmap(i))

    ax.scatter(
        coords[sub_idx, 0], coords[sub_idx, 1],
        marker="*", s=120, c="gold", edgecolor="black", zorder=5
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

# ============================
# LOAD DOS RESULTADOS
# ============================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of multi_objetivo)
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "pareto_front_results", "pareto_summary.csv")

df = pd.read_csv(csv_path)
print(f"Total de linhas no CSV: {len(df)}")
print(f"Colunas disponíveis: {df.columns.tolist()}")

# Remove linhas com valores NaN ou inválidos
df_clean = df.dropna(subset=["Custo_USD", "AEP_Liquido_MWh"])
print(f"Após remover NaN: {len(df_clean)} linhas")

df_clean = df_clean[(df_clean["Custo_USD"] > 0) & (df_clean["AEP_Liquido_MWh"] > 0)]
print(f"Após remover valores <= 0: {len(df_clean)} linhas")

# Remove duplicatas baseadas em Custo_USD e AEP_Liquido_MWh
# Mantém apenas soluções únicas na frente de Pareto
df_unique = df_clean.drop_duplicates(subset=["Custo_USD", "AEP_Liquido_MWh"])
print(f"Após remover duplicatas (mesmo Custo e AEP): {len(df_unique)} soluções únicas")

# Verifica valores únicos
unique_costs = df_unique["Custo_USD"].nunique()
unique_aeps = df_unique["AEP_Liquido_MWh"].nunique()
print(f"Valores únicos de Custo: {unique_costs}")
print(f"Valores únicos de AEP: {unique_aeps}")

# Mostra estatísticas
print(f"\nEstatísticas dos dados:")
print(f"  Custo - Min: {df_unique['Custo_USD'].min():.2f}, Max: {df_unique['Custo_USD'].max():.2f}")
print(f"  AEP - Min: {df_unique['AEP_Liquido_MWh'].min():.2f}, Max: {df_unique['AEP_Liquido_MWh'].max():.2f}")

df = df_unique
print(f"\nTotal de soluções únicas a serem plotadas: {len(df)}")

# ============================
# FIGURA FINAL - Apenas Frente de Pareto
# ============================

fig, ax = plt.subplots(figsize=(10, 7))

# Plota TODAS as soluções da frente de Pareto
# Usa valores únicos para evitar sobreposição visual
x_vals = df["Custo_USD"] / 1e6
y_vals = df["AEP_Liquido_MWh"]

print(f"\nPlotando {len(x_vals)} pontos...")
print(f"  X range: [{x_vals.min():.6f}, {x_vals.max():.6f}]")
print(f"  Y range: [{y_vals.min():.2f}, {y_vals.max():.2f}]")

# Calcula e destaca o knee point
idx_knee = knee_point(df)
knee_cost = x_vals.iloc[idx_knee]
knee_aep = y_vals.iloc[idx_knee]

print(f"\nKnee Point identificado:")
print(f"  Índice: {idx_knee}")
print(f"  AEP: {knee_aep:.2f} MWh ({knee_aep/1000:.2f} GWh)")
print(f"  Custo: ${knee_cost*1e6:,.0f} USD (${knee_cost:.3f}M)")

# Plota todas as soluções
ax.scatter(
    x_vals,
    y_vals,
    s=30,
    alpha=0.7,
    color="navy",
    edgecolors="darkblue",
    linewidths=0.3,
    label=f"Pareto Solutions (n={len(df)} unique)"
)

# Destaca o knee point
ax.scatter(
    knee_cost,
    knee_aep,
    s=200,
    alpha=1.0,
    color="orange",
    edgecolors="red",
    linewidths=2,
    marker="*",
    zorder=10,
    label=f"Knee Point (AEP: {knee_aep/1000:.2f} GWh, Cost: ${knee_cost:.3f}M)"
)

ax.set_xlabel("Total Cabling Cost (Million USD)", fontsize=12)
ax.set_ylabel("Net AEP (MWh/year)", fontsize=12)
ax.set_title("Pareto Front: Net AEP vs. Cabling Cost", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(frameon=True, loc="best", fontsize=10)

plt.tight_layout()
output_path = os.path.join(project_root, "pareto_front_complete.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {output_path}")

# ============================
# VISUALIZAÇÃO DO KNEE POINT COM CABEAMENTO
# ============================

print("\n" + "="*60)
print("CARREGANDO E VISUALIZANDO KNEE POINT COM CABEAMENTO")
print("="*60)

# Carrega dados do knee point
knee_solution = df.iloc[idx_knee]
knee_file_path = knee_solution["File"]

# Handle relative paths
if not os.path.isabs(knee_file_path):
    knee_file_path = os.path.join(project_root, knee_file_path)

print(f"Carregando solução do knee point: {knee_file_path}")

try:
    coords_knee, n_grupos_knee, substation_pos_knee = load_solution_with_substation(knee_file_path)
    
    # Calcula cabeamento usando a subestação otimizada
    if substation_pos_knee is not None:
        # Adiciona subestação como ponto extra
        coords_with_sub = np.vstack([coords_knee, substation_pos_knee.reshape(1, 2)])
        substation_idx = len(coords_knee)
    else:
        # Fallback: encontra turbina mais próxima do continente
        distancias_ao_continente = np.linalg.norm(coords_knee - SUBSTATION_CONTINENT, axis=1)
        substation_idx = np.argmin(distancias_ao_continente)
        coords_with_sub = coords_knee
    
    # Usa n_grupos do arquivo ou valor padrão
    if n_grupos_knee is not None:
        n_grupos_to_use = n_grupos_knee
    else:
        n_grupos_to_use = int(np.sqrt(IND_SIZE))
    
    print(f"  Número de grupos: {n_grupos_to_use}")
    print(f"  Posição da subestação: {substation_pos_knee}")
    
    # Calcula cabeamento
    planta_knee, resultados_knee = analisar_layout_completo(
        coords_with_sub, sub=substation_idx, n_grupos=n_grupos_to_use)
    
    print(f"  Custo calculado: ${resultados_knee['custo_total_usd']:,.0f} USD")
    print(f"  Perdas: {resultados_knee['perda_anual_mwh']:.2f} MWh")
    
    # Imprime cada string (path) do cabeamento
    print(f"\n  Strings do Cabeamento (Total: {len(planta_knee.paths)} grupos):")
    print("  " + "="*70)
    for i, path in enumerate(planta_knee.paths):
        # Converte índices para coordenadas para melhor visualização
        path_coords = [(coords_with_sub[idx, 0], coords_with_sub[idx, 1]) for idx in path]
        path_str = " -> ".join([f"T{idx}" if idx < len(coords_knee) else "SUB" for idx in path])
        print(f"  Grupo {i+1}: {path_str}")
        print(f"           Coordenadas: {path_coords}")
    print("  " + "="*70)
    
    # Cria figura com layout do knee point
    fig_knee, ax_knee = plt.subplots(figsize=(12, 10))
    
    # Desenha círculo de restrição
    circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                    linewidth=2, color='black', label='Limite do Parque')
    ax_knee.add_patch(circle)
    
    # Desenha cabeamento
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(planta_knee.paths), 10)))
    for i, path in enumerate(planta_knee.paths):
        if len(path) > 1:
            valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
            if len(valid_path) > 1:
                x_path = [coords_with_sub[k, 0] for k in valid_path]
                y_path = [coords_with_sub[k, 1] for k in valid_path]
                ax_knee.plot(x_path, y_path, '-', linewidth=2.5, 
                            color=colors[i % len(colors)], alpha=0.8, zorder=4,
                            label=f'Grupo {i+1}' if i < 10 else '')
    
    # Desenha turbinas
    ax_knee.scatter(coords_knee[:, 0], coords_knee[:, 1], 
                   s=150, c='red', edgecolors='black', 
                   linewidths=1.5, zorder=5, label='Turbinas')
    
    # Adiciona labels com números das turbinas
    for i in range(len(coords_knee)):
        ax_knee.annotate(f'T{i}', 
                        (coords_knee[i, 0], coords_knee[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.7),
                        zorder=7)
    
    # Desenha subestação
    if substation_pos_knee is not None:
        ax_knee.scatter(substation_pos_knee[0], substation_pos_knee[1],
                       marker='*', s=400, c='gold', edgecolors='black',
                       linewidths=2, zorder=6, label='Subestação (GA)')
        # Adiciona label da subestação
        ax_knee.annotate('SUB', 
                        (substation_pos_knee[0], substation_pos_knee[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', 
                                edgecolor='black', alpha=0.8),
                        zorder=7)
    
    ax_knee.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax_knee.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax_knee.set_aspect('equal')
    ax_knee.set_xlabel('X (m)', fontsize=12)
    ax_knee.set_ylabel('Y (m)', fontsize=12)
    ax_knee.grid(True, linestyle='--', alpha=0.6)
    ax_knee.set_title(
        f'Knee Point Layout\n'
        f'AEP: {knee_aep/1000:.2f} GWh | Custo: ${knee_cost:.3f}M | '
        f'Grupos: {n_grupos_to_use} | Perdas: {resultados_knee["perda_anual_mwh"]:.2f} MWh',
        fontsize=14, fontweight='bold'
    )
    ax_knee.legend(loc='upper right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    output_path_knee = os.path.join(project_root, "knee_point_layout.png")
    plt.savefig(output_path_knee, dpi=300, bbox_inches="tight")
    print(f"\nLayout do Knee Point salvo em: {output_path_knee}")
    
except Exception as e:
    print(f"ERRO ao carregar/visualizar knee point: {e}")
    import traceback
    traceback.print_exc()

# ============================
# MOSTRA ANIMAÇÃO E FIGURA ESTÁTICA
# ============================

# Primeiro mostra a animação
print("\n" + "="*60)
print("MOSTRANDO ANIMAÇÃO DA EVOLUÇÃO")
print("="*60)
anim_result = create_evolution_animation(project_root)
if anim_result is not None:
    anim, fig_anim = anim_result
    print("\nAnimações: Feche a janela da animação para continuar...")
    plt.show(block=True)  # Bloqueia até fechar a janela
    plt.close(fig_anim)

# Depois mostra a figura estática
print("\n" + "="*60)
print("MOSTRANDO FRENTE DE PARETO COMPLETA")
print("="*60)
plt.show()
