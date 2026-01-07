import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.animation as animation
import sys
import os
import glob
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =============================================================================
# CONFIGURAÇÃO DE FONTES PARA PUBLICAÇÃO (Type 42 = TrueType)
# =============================================================================
# ACM/GECCO requerem Type 1/TrueType fonts (Type 42). Type 3 fonts NÃO são aceitos.
# Type 42 é o padrão para PDF/PS com fontes TrueType embutidas.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from multi_objetivo.cabling_v3 import analisar_layout_completo

# ============================
# CONFIGURAÇÕES - MODIFIQUE AQUI PARA DIFERENTES CENÁRIOS
# ============================
# Para 16 turbinas: CIRCLE_RADIUS = 5000, IND_SIZE = 16
# Para 36 turbinas: CIRCLE_RADIUS = 2000, IND_SIZE = 36
# Para 64 turbinas: CIRCLE_RADIUS = 3000, IND_SIZE = 64

CIRCLE_RADIUS = 5000  # Raio do círculo de restrição (metros)
IND_SIZE = 64         # Número de turbinas
EVOLUTION_DIR_PHASE1 = "pareto_front_results/evolution_phase1"  # Diretório Fase 1
EVOLUTION_DIR_PHASE2 = "pareto_front_results/evolution"        # Diretório Fase 2
SUBSTATION_CONTINENT = np.array([[-1.0, -1350.0]])  # Posição de referência (não usado na Fase 2)

# Tamanhos para visualização (proporcionais à escala)
TURBINE_MARKER_SIZE = 40  # Tamanho do marcador das turbinas (ajuste para proporcionalidade)
TURBINE_EDGE_WIDTH = 1.0  # Largura da borda das turbinas
SUBSTATION_MARKER_SIZE = 200  # Tamanho do marcador da subestação (estrela)
CABLE_LINEWIDTH = 1.0  # Espessura das linhas de cabeamento (reduzida para melhor visualização)

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
        phase1_files = glob.glob(pattern_phase1)
        files.extend(phase1_files)
        print(f"Encontrados {len(phase1_files)} arquivos da Fase 1 em: {phase1_dir}")
    else:
        print(f"AVISO: Diretório da Fase 1 não encontrado: {phase1_dir}")
    
    # Adiciona arquivos da Fase 2
    phase2_dir = os.path.join(project_root, EVOLUTION_DIR_PHASE2)
    if os.path.exists(phase2_dir):
        pattern_phase2 = os.path.join(phase2_dir, "gen_*.txt")
        phase2_files = glob.glob(pattern_phase2)
        files.extend(phase2_files)
        print(f"Encontrados {len(phase2_files)} arquivos da Fase 2 em: {phase2_dir}")
    else:
        print(f"AVISO: Diretório da Fase 2 não encontrado: {phase2_dir}")
    
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
    fig.suptitle('Genetic Algorithm Evolution', fontsize=16, fontweight='bold')
    
    # Configuração do gráfico de layout
    ax1.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax1.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Círculo de restrição
    circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                    linewidth=2, color='black', label='Park Limit')
    ax1.add_patch(circle)
    
    # Configuração do gráfico de métricas
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Metrics Evolution', fontsize=14)
    
    # Inicializa elementos
    scatter = ax1.scatter([], [], s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
                          linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
    title_text = ax1.set_title('', fontsize=14, fontweight='bold')
    
    # Lista para armazenar linhas de cabeamento e marcadores de subestação
    cabling_lines_container = []
    substation_markers_container = []
    
    # Linhas de evolução
    line_aep, = ax2.plot([], [], 'b-', linewidth=2, label='AEP (GWh)', marker='o', markersize=4)
    
    # Cria eixos secundários para custo
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Cost (USD)', fontsize=12, color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # Linha de custo no eixo secundário
    line_cost, = ax2_twin.plot([], [], 'r-', linewidth=2, label='Cost (USD)', marker='s', markersize=4, color='red')
    
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
                marker='*', s=SUBSTATION_MARKER_SIZE, c='gold', edgecolors='black',
                linewidths=TURBINE_EDGE_WIDTH, zorder=6, label='Offshore Substation'
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
                        line, = ax1.plot(x_path, y_path, '-', linewidth=CABLE_LINEWIDTH, 
                                        color=colors[i % len(colors)], alpha=0.8, zorder=4)
                        cabling_lines_container.append(line)
        
        # Atualiza layout
        scatter.set_offsets(coords)
        
        # Título muda conforme a fase
        if phase == 1:
            title_text.set_text(f'Phase 1 - Generation {gen_num} | Gross AEP: {aep/1000:.2f} GWh')
        else:
            n_grupos_str = f" | Groups: {all_n_grupos[frame]}" if all_n_grupos[frame] is not None else ""
            cost_str = f"${cost:,.0f}" if cost is not None else "N/A"
            title_text.set_text(f'Phase 2 - Generation {gen_num} | Net AEP: {aep/1000:.2f} GWh | Cost: {cost_str}{n_grupos_str}')
        
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
# CONFIGURAÇÕES DE ESTILO (PAPER/ACM)
# ============================

# Configurações para qualidade de publicação ACM
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,  # Alta resolução para publicação
    "savefig.dpi": 300,  # DPI para salvamento
    "savefig.bbox": "tight",  # Bbox tight para melhor layout
    "savefig.pad_inches": 0.1,  # Padding ao salvar
})

# Usa estilo seaborn-paper se disponível (melhor para artigos)
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        pass  # Continua com estilo padrão se não disponível

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

def plot_layout_with_cabling(coords, n_grupos, substation_pos, title, aep_gwh, cost_kusd, 
                             n_grupos_to_use, output_basename, project_root):
    """
    Função auxiliar para plotar layout com cabeamento.
    Retorna a figura criada e os caminhos dos arquivos salvos.
    """
    # Garante que a subestação fique dentro do círculo
    if substation_pos is not None:
        dist_sub = np.linalg.norm(substation_pos)
        if dist_sub > CIRCLE_RADIUS:
            angle = np.arctan2(substation_pos[1], substation_pos[0])
            substation_pos[0] = CIRCLE_RADIUS * np.cos(angle)
            substation_pos[1] = CIRCLE_RADIUS * np.sin(angle)
    
    # Calcula cabeamento usando a subestação otimizada
    if substation_pos is not None:
        coords_with_sub = np.vstack([coords, substation_pos.reshape(1, 2)])
        substation_idx = len(coords)
    else:
        distancias_ao_continente = np.linalg.norm(coords - SUBSTATION_CONTINENT, axis=1)
        substation_idx = np.argmin(distancias_ao_continente)
        coords_with_sub = coords
    
    # Calcula cabeamento
    planta, resultados = analisar_layout_completo(
        coords_with_sub, sub=substation_idx, n_grupos=n_grupos_to_use)
    
    losses_gwh = resultados["perda_anual_mwh"] / 1000.0  # Converte perdas para GWh
    comprimento_total_km = resultados["comprimento_total_m"] / 1000.0  # Converte para km
    secao_cabo_mm2 = resultados.get("secao_cabo_mm2") or resultados.get("secao_mm2", 0)  # Bitola do cabo
    # Usa o custo recalculado para manter consistência com comprimento e bitola
    cost_recalculated_kusd = resultados["custo_total_usd"] / 1000.0
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Desenha círculo de restrição
    circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                   linewidth=2, color='black', label='Park Limit')
    ax.add_patch(circle)
    
    # Desenha cabeamento
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(planta.paths), 10)))
    for i, path in enumerate(planta.paths):
        if len(path) > 1:
            valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
            if len(valid_path) > 1:
                x_path = [coords_with_sub[k, 0] for k in valid_path]
                y_path = [coords_with_sub[k, 1] for k in valid_path]
                ax.plot(x_path, y_path, '-', linewidth=CABLE_LINEWIDTH, 
                       color=colors[i % len(colors)], alpha=0.8, zorder=4,
                       label=f'Group {i+1}' if i < 10 else '')
    
    # Desenha turbinas
    ax.scatter(coords[:, 0], coords[:, 1], 
              s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
              linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
    
    # Adiciona labels com números das turbinas
    for i in range(len(coords)):
        ax.annotate(f'T{i}', 
                   (coords[i, 0], coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.7),
                   zorder=7)
    
    # Desenha subestação
    if substation_pos is not None:
        ax.scatter(substation_pos[0], substation_pos[1],
                  marker='*', s=SUBSTATION_MARKER_SIZE, c='gold', edgecolors='black',
                  linewidths=TURBINE_EDGE_WIDTH, zorder=6, label='Offshore Substation')
        # Adiciona label da subestação
        ax.annotate('SUB', 
                   (substation_pos[0], substation_pos[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', 
                           edgecolor='black', alpha=0.8),
                   zorder=7)
    
    ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_title(
        f'{title}\n'
        f'AEP: {aep_gwh:.2f} GWh | Cost: ${cost_recalculated_kusd:.0f}k USD | Groups: {n_grupos_to_use}\n'
        f'Losses: {losses_gwh:.2f} GWh | Cable: {secao_cabo_mm2:.0f} mm² | Length: {comprimento_total_km:.2f} km',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=9, ncol=2, frameon=True)
    
    plt.tight_layout()
    # Salva em múltiplos formatos
    output_path_png = os.path.join(project_root, f"{output_basename}.png")
    output_path_pdf = os.path.join(project_root, f"{output_basename}.pdf")
    output_path_eps = os.path.join(project_root, f"{output_basename}.eps")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_eps, dpi=300, bbox_inches="tight", facecolor='white')
    
    return fig, output_path_png, output_path_pdf, output_path_eps

# ============================
# LOAD DOS RESULTADOS
# ============================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of multi_objetivo)
project_root = os.path.dirname(script_dir)

# Determina o diretório de resultados baseado em EVOLUTION_DIR_PHASE2
# Ex: "pareto_front_results/evolution" -> "pareto_front_results"
# Ex: "pareto_front_results_36/evolution" -> "pareto_front_results_36"
results_dir_name = os.path.dirname(EVOLUTION_DIR_PHASE2)
if results_dir_name == "":
    results_dir_name = "pareto_front_results"  # Fallback

csv_path = os.path.join(project_root, results_dir_name, "pareto_summary.csv")
print(f"Procurando CSV em: {csv_path}")

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

df = df_unique.reset_index(drop=True)  # Reseta índice para evitar problemas com índices antigos
print(f"\nTotal de soluções únicas a serem plotadas: {len(df)}")

# ============================
# FIGURA FINAL - Frente de Pareto da Fase 2
# ============================

fig, ax = plt.subplots(figsize=(10, 7))

# Plota TODAS as soluções da frente de Pareto da Fase 2
# Converte custo para milhares de USD e AEP para GWh para melhor legibilidade
x_vals = df["Custo_USD"] / 1000.0  # Milhares de USD
y_vals = df["AEP_Liquido_MWh"] / 1000.0  # GWh

print(f"\nPlotando {len(x_vals)} soluções da Fase 2...")
print(f"  Custo range: [{x_vals.min():.2f}, {x_vals.max():.2f}] kUSD")
print(f"  AEP range: [{y_vals.min():.2f}, {y_vals.max():.2f}] GWh")

# Calcula e destaca o knee point, menor custo e maior AEP
idx_knee = knee_point(df)
# Usa idxmin/idxmax que retorna o índice do DataFrame (já resetado)
idx_min_cost = df["Custo_USD"].idxmin()  # Índice da solução com menor custo
idx_max_aep = df["AEP_Liquido_MWh"].idxmax()  # Índice da solução com maior AEP

knee_cost = x_vals.iloc[idx_knee]
knee_aep = y_vals.iloc[idx_knee]
min_cost_val = x_vals.iloc[idx_min_cost]
min_cost_aep = y_vals.iloc[idx_min_cost]
max_aep_val = y_vals.iloc[idx_max_aep]
max_aep_cost = x_vals.iloc[idx_max_aep]

print(f"\nSoluções identificadas:")
print(f"  Menor Custo - Índice: {idx_min_cost}, AEP: {min_cost_aep:.2f} GWh, Custo: ${min_cost_val:.0f}k USD")
print(f"  Knee Point - Índice: {idx_knee}, AEP: {knee_aep:.2f} GWh, Custo: ${knee_cost:.0f}k USD")
print(f"  Maior AEP  - Índice: {idx_max_aep}, AEP: {max_aep_val:.2f} GWh, Custo: ${max_aep_cost:.0f}k USD")

# Plota todas as soluções da frente de Pareto (Fase 2)
ax.scatter(
    x_vals,
    y_vals,
    s=40,
    alpha=0.7,
    color="navy",
    edgecolors="darkblue",
    linewidths=0.5,
    label=f"Pareto Solutions - Phase 2 (n={len(df)})",
    zorder=3
)

# Destaca o knee point com estrela dourada
ax.scatter(
    knee_cost,
    knee_aep,
    s=300,
    alpha=1.0,
    color="gold",
    edgecolors="black",
    linewidths=2,
    marker="*",
    zorder=10,
    label=f"Knee Point (AEP: {knee_aep:.2f} GWh, Cost: ${knee_cost:.0f}k USD)"
)

# Configurações do gráfico
ax.set_xlabel("Total Cabling Cost (Thousands USD)", fontsize=12, fontweight='bold')
ax.set_ylabel("Net AEP (GWh)", fontsize=12, fontweight='bold')
ax.set_title("Pareto Front - Phase 2: Net AEP vs. Cabling Cost", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
ax.legend(frameon=True, loc="best", fontsize=10, framealpha=0.9)

# Melhora a aparência geral
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
output_path = os.path.join(project_root, "pareto_front_complete.png")
# Salva em PNG (alta qualidade)
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
# Salva também em PDF e EPS (Type 42 fonts para ACM)
output_path_pdf = os.path.join(project_root, "pareto_front_complete.pdf")
output_path_eps = os.path.join(project_root, "pareto_front_complete.eps")
plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor='white')
plt.savefig(output_path_eps, dpi=300, bbox_inches="tight", facecolor='white')
print(f"Figure saved to: {output_path}")
print(f"Figure saved to: {output_path_pdf}")
print(f"Figure saved to: {output_path_eps}")

# ============================
# FUNÇÃO AUXILIAR PARA CARREGAR E PLOTAR SOLUÇÃO
# ============================

def load_and_plot_solution(df_idx, df, title, output_basename, project_root, results_dir_name):
    """Carrega e plota uma solução específica da Fase 2."""
    solution = df.iloc[df_idx]
    file_path = solution["File"]
    aep_gwh = solution["AEP_Liquido_MWh"] / 1000.0
    cost_kusd = solution["Custo_USD"] / 1000.0
    
    # Handle relative paths
    if not os.path.isabs(file_path):
        if results_dir_name in file_path:
            file_path = os.path.join(project_root, file_path)
        else:
            file_path = os.path.join(project_root, results_dir_name, os.path.basename(file_path))
    
    print(f"\nCarregando solução: {os.path.basename(file_path)}")
    print(f"  AEP: {aep_gwh:.2f} GWh, Custo: ${cost_kusd:.0f}k USD")
    
    try:
        coords, n_grupos, substation_pos = load_solution_with_substation(file_path)
        
        # Usa n_grupos do arquivo ou valor padrão
        if n_grupos is not None:
            n_grupos_to_use = n_grupos
        else:
            n_grupos_to_use = int(np.sqrt(IND_SIZE))
        
        # Plota usando função auxiliar (ela calcula cabeamento e perdas internamente)
        fig, png_path, pdf_path, eps_path = plot_layout_with_cabling(
            coords, n_grupos, substation_pos, title, aep_gwh, cost_kusd,
            n_grupos_to_use, output_basename, project_root)
        
        print(f"  Layout salvo em: {png_path}")
        return fig
        
    except Exception as e:
        print(f"ERRO ao carregar/visualizar solução: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================
# VISUALIZAÇÃO DAS SOLUÇÕES COM CABEAMENTO
# ============================

print("\n" + "="*60)
print("CARREGANDO E VISUALIZANDO SOLUÇÕES COM CABEAMENTO")
print("="*60)

# Plota menor custo
print("\n" + "-"*60)
print("1. SOLUÇÃO COM MENOR CUSTO")
print("-"*60)
load_and_plot_solution(
    idx_min_cost, df, "Optimized Layout - Minimum Cost",
    "min_cost_layout", project_root, results_dir_name)

# Plota knee point
print("\n" + "-"*60)
print("2. SOLUÇÃO KNEE POINT")
print("-"*60)
load_and_plot_solution(
    idx_knee, df, "Optimized Layout - Knee Point",
    "knee_point_layout", project_root, results_dir_name)

# Plota maior AEP
print("\n" + "-"*60)
print("3. SOLUÇÃO COM MAIOR AEP")
print("-"*60)
load_and_plot_solution(
    idx_max_aep, df, "Optimized Layout - Maximum AEP",
    "max_aep_layout", project_root, results_dir_name)

# ============================
# VISUALIZAÇÃO DO INDIVÍDUO INICIAL DA FASE 1
# ============================

print("\n" + "="*60)
print("CARREGANDO E VISUALIZANDO INDIVÍDUO INICIAL DA FASE 1")
print("="*60)

# Carrega o primeiro indivíduo da Fase 1 (geração 0)
gen0_file = os.path.join(project_root, EVOLUTION_DIR_PHASE1, "gen_0000_best.txt")
print(f"Carregando indivíduo inicial: {gen0_file}")

try:
    if os.path.exists(gen0_file):
        coords_initial, aep_initial, _, _, _, _ = load_generation_file(gen0_file)
        aep_initial_gwh = aep_initial / 1000.0 if aep_initial else 0.0
        
        print(f"  Coordenadas carregadas: {len(coords_initial)} turbinas")
        print(f"  AEP: {aep_initial_gwh:.2f} GWh")
        
        # Cria figura com layout inicial
        fig_initial, ax_initial = plt.subplots(figsize=(12, 10))
        
        # Desenha círculo de restrição
        circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                       linewidth=2, color='black', label='Park Limit')
        ax_initial.add_patch(circle)
        
        # Desenha turbinas
        ax_initial.scatter(coords_initial[:, 0], coords_initial[:, 1], 
                          s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
                          linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
        
        # Adiciona labels com números das turbinas
        for i in range(len(coords_initial)):
            ax_initial.annotate(f'T{i}', 
                              (coords_initial[i, 0], coords_initial[i, 1]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                      edgecolor='black', alpha=0.7),
                              zorder=7)
        
        ax_initial.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
        ax_initial.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
        ax_initial.set_aspect('equal')
        ax_initial.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax_initial.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax_initial.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax_initial.set_title(
            f'Initial Layout - Phase 1 (Generation 0)\n'
            f'AEP: {aep_initial_gwh:.2f} GWh | Turbines: {IND_SIZE}',
            fontsize=14, fontweight='bold'
        )
        ax_initial.legend(loc='upper right', fontsize=9, frameon=True)
        
        plt.tight_layout()
        # Salva em múltiplos formatos (alta qualidade, Type 42 fonts)
        output_path_initial_png = os.path.join(project_root, "phase1_initial_layout.png")
        output_path_initial_pdf = os.path.join(project_root, "phase1_initial_layout.pdf")
        output_path_initial_eps = os.path.join(project_root, "phase1_initial_layout.eps")
        plt.savefig(output_path_initial_png, dpi=300, bbox_inches="tight", facecolor='white')
        plt.savefig(output_path_initial_pdf, dpi=300, bbox_inches="tight", facecolor='white')
        plt.savefig(output_path_initial_eps, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"\nLayout Inicial da Fase 1 salvo em:")
        print(f"  PNG: {output_path_initial_png}")
        print(f"  PDF: {output_path_initial_pdf}")
        print(f"  EPS: {output_path_initial_eps}")
    else:
        print(f"AVISO: Arquivo não encontrado: {gen0_file}")
        print("  Pulando plot do indivíduo inicial...")
        
except Exception as e:
    print(f"ERRO ao carregar/visualizar indivíduo inicial: {e}")
    import traceback
    traceback.print_exc()

# ============================
# VISUALIZAÇÃO DO ÚLTIMO INDIVÍDUO DA FASE 1 (MAIOR AEP)
# ============================

print("\n" + "="*60)
print("CARREGANDO E VISUALIZANDO ÚLTIMO INDIVÍDUO DA FASE 1 (MAIOR AEP)")
print("="*60)

# Encontra o último arquivo da Fase 1 (maior número de geração)
phase1_dir = os.path.join(project_root, EVOLUTION_DIR_PHASE1)
print(f"Procurando arquivos da Fase 1 em: {phase1_dir}")

try:
    if os.path.exists(phase1_dir):
        pattern_phase1 = os.path.join(phase1_dir, "gen_*.txt")
        phase1_files = glob.glob(pattern_phase1)
        
        if len(phase1_files) > 0:
            # Ordena por número da geração
            def get_gen_number(filename):
                match = re.search(r'gen_(\d+)_', filename)
                return int(match.group(1)) if match else 0
            
            phase1_files.sort(key=get_gen_number)
            last_phase1_file = phase1_files[-1]  # Último arquivo (maior geração)
            gen_num = get_gen_number(last_phase1_file)
            
            print(f"Carregando último indivíduo da Fase 1: {os.path.basename(last_phase1_file)} (Geração {gen_num})")
            
            coords_best_p1, aep_best_p1, _, _, _, _ = load_generation_file(last_phase1_file)
            aep_best_p1_gwh = aep_best_p1 / 1000.0 if aep_best_p1 else 0.0
            
            print(f"  Coordenadas carregadas: {len(coords_best_p1)} turbinas")
            print(f"  AEP: {aep_best_p1_gwh:.2f} GWh")
            print(f"  Geração: {gen_num}")
            
            # Cria figura com layout do melhor indivíduo da Fase 1
            fig_best_p1, ax_best_p1 = plt.subplots(figsize=(12, 10))
            
            # Desenha círculo de restrição
            circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                           linewidth=2, color='black', label='Park Limit')
            ax_best_p1.add_patch(circle)
            
            # Desenha turbinas
            ax_best_p1.scatter(coords_best_p1[:, 0], coords_best_p1[:, 1], 
                              s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
                              linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
            
            # Adiciona labels com números das turbinas
            for i in range(len(coords_best_p1)):
                ax_best_p1.annotate(f'T{i}', 
                                  (coords_best_p1[i, 0], coords_best_p1[i, 1]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=9, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                          edgecolor='black', alpha=0.7),
                                  zorder=7)
            
            ax_best_p1.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax_best_p1.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax_best_p1.set_aspect('equal')
            ax_best_p1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax_best_p1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax_best_p1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            
            ax_best_p1.set_title(
                f'Best Layout - Phase 1 (Generation {gen_num})\n'
                f'AEP: {aep_best_p1_gwh:.2f} GWh | Turbines: {IND_SIZE}',
                fontsize=14, fontweight='bold'
            )
            ax_best_p1.legend(loc='upper right', fontsize=9, frameon=True)
            
            plt.tight_layout()
            # Salva em múltiplos formatos (alta qualidade, Type 42 fonts)
            output_path_best_p1_png = os.path.join(project_root, "phase1_best_layout.png")
            output_path_best_p1_pdf = os.path.join(project_root, "phase1_best_layout.pdf")
            output_path_best_p1_eps = os.path.join(project_root, "phase1_best_layout.eps")
            plt.savefig(output_path_best_p1_png, dpi=300, bbox_inches="tight", facecolor='white')
            plt.savefig(output_path_best_p1_pdf, dpi=300, bbox_inches="tight", facecolor='white')
            plt.savefig(output_path_best_p1_eps, dpi=300, bbox_inches="tight", facecolor='white')
            print(f"\nLayout do Melhor Indivíduo da Fase 1 salvo em:")
            print(f"  PNG: {output_path_best_p1_png}")
            print(f"  PDF: {output_path_best_p1_pdf}")
            print(f"  EPS: {output_path_best_p1_eps}")
        else:
            print(f"AVISO: Nenhum arquivo da Fase 1 encontrado em: {phase1_dir}")
    else:
        print(f"AVISO: Diretório da Fase 1 não encontrado: {phase1_dir}")
        
except Exception as e:
    print(f"ERRO ao carregar/visualizar melhor indivíduo da Fase 1: {e}")
    import traceback
    traceback.print_exc()

# ============================
# SUGESTÃO 1: FIGURA COMPARATIVA LADO A LADO (1x3)
# ============================

print("\n" + "="*60)
print("GERANDO FIGURA COMPARATIVA LADO A LADO (1x3)")
print("="*60)

def plot_comparative_layouts_1x3(df, idx_min_cost, idx_knee, idx_max_aep, 
                                  project_root, results_dir_name):
    """Plota os 3 layouts principais lado a lado em uma figura 1x3."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    solutions_info = [
        (idx_min_cost, "Minimum Cost", "min_cost"),
        (idx_knee, "Knee Point", "knee_point"),
        (idx_max_aep, "Maximum AEP", "max_aep")
    ]
    
    for ax_idx, (df_idx, title_suffix, _) in enumerate(solutions_info):
        solution = df.iloc[df_idx]
        file_path = solution["File"]
        aep_gwh = solution["AEP_Liquido_MWh"] / 1000.0
        cost_kusd = solution["Custo_USD"] / 1000.0
        
        # Handle relative paths
        if not os.path.isabs(file_path):
            if results_dir_name in file_path:
                file_path = os.path.join(project_root, file_path)
            else:
                file_path = os.path.join(project_root, results_dir_name, os.path.basename(file_path))
        
        try:
            coords, n_grupos, substation_pos = load_solution_with_substation(file_path)
            
            if n_grupos is not None:
                n_grupos_to_use = n_grupos
            else:
                n_grupos_to_use = int(np.sqrt(IND_SIZE))
            
            # Garante que a subestação fique dentro do círculo
            if substation_pos is not None:
                dist_sub = np.linalg.norm(substation_pos)
                if dist_sub > CIRCLE_RADIUS:
                    angle = np.arctan2(substation_pos[1], substation_pos[0])
                    substation_pos[0] = CIRCLE_RADIUS * np.cos(angle)
                    substation_pos[1] = CIRCLE_RADIUS * np.sin(angle)
            
            # Calcula cabeamento
            if substation_pos is not None:
                coords_with_sub = np.vstack([coords, substation_pos.reshape(1, 2)])
                substation_idx = len(coords)
            else:
                distancias_ao_continente = np.linalg.norm(coords - SUBSTATION_CONTINENT, axis=1)
                substation_idx = np.argmin(distancias_ao_continente)
                coords_with_sub = coords
            
            planta, resultados = analisar_layout_completo(
                coords_with_sub, sub=substation_idx, n_grupos=n_grupos_to_use)
            
            losses_gwh = resultados["perda_anual_mwh"] / 1000.0
            comprimento_total_km = resultados["comprimento_total_m"] / 1000.0
            secao_cabo_mm2 = resultados.get("secao_cabo_mm2") or resultados.get("secao_mm2", 0)
            # Usa o custo recalculado para manter consistência com comprimento e bitola
            cost_recalculated_kusd = resultados["custo_total_usd"] / 1000.0
            ax = axes[ax_idx]
            
            # Desenha círculo de restrição
            circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                           linewidth=2, color='black', label='Park Limit')
            ax.add_patch(circle)
            
            # Desenha cabeamento
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(planta.paths), 10)))
            for i, path in enumerate(planta.paths):
                if len(path) > 1:
                    valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
                    if len(valid_path) > 1:
                        x_path = [coords_with_sub[k, 0] for k in valid_path]
                        y_path = [coords_with_sub[k, 1] for k in valid_path]
                        ax.plot(x_path, y_path, '-', linewidth=CABLE_LINEWIDTH, 
                               color=colors[i % len(colors)], alpha=0.7, zorder=4)
            
            # Desenha turbinas
            ax.scatter(coords[:, 0], coords[:, 1], 
                      s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', 
                      linewidths=TURBINE_EDGE_WIDTH, zorder=5, label='Turbines')
            
            # Desenha subestação
            if substation_pos is not None:
                ax.scatter(substation_pos[0], substation_pos[1],
                          marker='*', s=SUBSTATION_MARKER_SIZE, c='gold', edgecolors='black',
                          linewidths=TURBINE_EDGE_WIDTH, zorder=6, label='Substation')
            
            ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            
            ax.set_title(
                f'{title_suffix}\n'
                f'AEP: {aep_gwh:.2f} GWh | Cost: ${cost_recalculated_kusd:.0f}k USD\n'
                f'Cable: {secao_cabo_mm2:.0f} mm² | Length: {comprimento_total_km:.2f} km',
                fontsize=12, fontweight='bold'
            )
            
        except Exception as e:
            print(f"  ERRO ao carregar {title_suffix}: {e}")
            axes[ax_idx].text(0.5, 0.5, f'Error loading\n{title_suffix}', 
                            ha='center', va='center', transform=axes[ax_idx].transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(project_root, "comparative_layouts_1x3.png")
    output_path_pdf = os.path.join(project_root, "comparative_layouts_1x3.pdf")
    output_path_eps = os.path.join(project_root, "comparative_layouts_1x3.eps")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_eps, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"Figura comparativa 1x3 salva em:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {output_path_pdf}")
    print(f"  EPS: {output_path_eps}")
    plt.close(fig)

plot_comparative_layouts_1x3(df, idx_min_cost, idx_knee, idx_max_aep, 
                              project_root, results_dir_name)

# ============================
# SUGESTÃO 2: GRÁFICO DE EVOLUÇÃO DAS MÉTRICAS
# ============================

print("\n" + "="*60)
print("GERANDO GRÁFICO DE EVOLUÇÃO DAS MÉTRICAS")
print("="*60)

def plot_metrics_evolution(project_root):
    """Plota evolução de AEP e Custo ao longo das gerações (Fase 1 e Fase 2)."""
    files = get_generation_files(project_root)
    
    if len(files) == 0:
        print("AVISO: Nenhum arquivo de evolução encontrado. Pulando gráfico de evolução...")
        return
    
    # Carrega dados de todas as gerações
    gens_p1, aeps_p1 = [], []
    gens_p2, aeps_p2, costs_p2 = [], [], []
    
    for filename in files:
        coords, aep, cost, phase, n_grupos, substation_pos = load_generation_file(filename)
        match = re.search(r'gen_(\d+)_', filename)
        gen_num = int(match.group(1)) if match else 0
        
        if phase == 1 and aep is not None:
            gens_p1.append(gen_num)
            aeps_p1.append(aep / 1000.0)  # Converte para GWh
        elif phase == 2 and aep is not None and cost is not None:
            gens_p2.append(gen_num)
            aeps_p2.append(aep / 1000.0)  # Converte para GWh
            costs_p2.append(cost / 1000.0)  # Converte para kUSD
    
    if len(gens_p1) == 0 and len(gens_p2) == 0:
        print("AVISO: Nenhum dado válido encontrado. Pulando gráfico de evolução...")
        return
    
    # Prepara dados de custo para começar na gen 0 junto com a Fase 1
    # Na Fase 1, o custo não é calculado, então usamos NaN para não plotar
    gens_cost = []
    costs_all = []
    
    # Adiciona valores NaN para as gerações da Fase 1 (para alinhar com AEP)
    if len(gens_p1) > 0:
        for gen in gens_p1:
            gens_cost.append(gen)
            costs_all.append(float('nan'))  # NaN para Fase 1 (não calcula custo)
    
    # Adiciona valores reais de custo para a Fase 2
    if len(gens_p2) > 0:
        for gen, cost in zip(gens_p2, costs_p2):
            gens_cost.append(gen)
            costs_all.append(cost)
    
    # Ordena por geração
    if len(gens_cost) > 0:
        sorted_data = sorted(zip(gens_cost, costs_all))
        gens_cost, costs_all = zip(*sorted_data)
        gens_cost = list(gens_cost)
        costs_all = list(costs_all)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico de AEP (Fase 1 e Fase 2)
    if len(gens_p1) > 0:
        ax1.plot(gens_p1, aeps_p1, 'b-o', linewidth=2, markersize=4, 
                label='Phase 1: Gross AEP', alpha=0.7)
    if len(gens_p2) > 0:
        ax1.plot(gens_p2, aeps_p2, 'g-s', linewidth=2, markersize=4, 
                label='Phase 2: Net AEP', alpha=0.7)
    
    # Calcula limites do eixo X para ambos os gráficos
    all_gens = (gens_p1 if len(gens_p1) > 0 else []) + (gens_p2 if len(gens_p2) > 0 else [])
    if len(all_gens) > 0:
        x_min = 0  # Sempre começa em 0
        x_max = max(all_gens)
    else:
        x_min = 0
        x_max = 1000
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AEP (GWh)', fontsize=12, fontweight='bold')
    ax1.set_title('AEP Evolution Across Generations', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlim(left=x_min, right=x_max)  # Força começar em 0
    
    # Gráfico de Custo (começa na gen 0 junto com Fase 1, mas só mostra valores na Fase 2)
    if len(gens_cost) > 0:
        ax2.plot(gens_cost, costs_all, 'r-^', linewidth=2, markersize=4, 
                label='Phase 2: Cabling Cost', alpha=0.7)
        ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cost (Thousands USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Cabling Cost Evolution (Phase 2)', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.set_xlim(left=x_min, right=x_max)  # Força começar em 0 e alinha com gráfico de cima
    else:
        ax2.text(0.5, 0.5, 'No Phase 2 data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Cabling Cost Evolution (Phase 2)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, "metrics_evolution.png")
    output_path_pdf = os.path.join(project_root, "metrics_evolution.pdf")
    output_path_eps = os.path.join(project_root, "metrics_evolution.eps")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_eps, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"Gráfico de evolução salvo em:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {output_path_pdf}")
    print(f"  EPS: {output_path_eps}")
    plt.close(fig)

plot_metrics_evolution(project_root)

# ============================
# SUGESTÃO 3: COMPARAÇÃO FASE 1 VS FASE 2
# ============================

print("\n" + "="*60)
print("GERANDO COMPARAÇÃO FASE 1 VS FASE 2")
print("="*60)

def plot_phase1_vs_phase2_comparison(project_root, df, idx_knee, results_dir_name):
    """Plota comparação: Layout inicial Fase 1, Melhor Fase 1, e Knee Point Fase 2."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Layout inicial da Fase 1
    gen0_file = os.path.join(project_root, EVOLUTION_DIR_PHASE1, "gen_0000_best.txt")
    if os.path.exists(gen0_file):
        try:
            coords_initial, aep_initial, _, _, _, _ = load_generation_file(gen0_file)
            aep_initial_gwh = aep_initial / 1000.0 if aep_initial else 0.0
            
            ax = axes[0]
            circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                          linewidth=2, color='black', label='Park Limit')
            ax.add_patch(circle)
            ax.scatter(coords_initial[:, 0], coords_initial[:, 1], 
                      s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', linewidths=TURBINE_EDGE_WIDTH, zorder=5)
            ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_title(f'Phase 1 - Initial\nAEP: {aep_initial_gwh:.2f} GWh', 
                        fontsize=12, fontweight='bold')
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Error loading\ninitial layout', 
                        ha='center', va='center', transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'Initial layout\nnot found', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # 2. Melhor layout da Fase 1
    phase1_dir = os.path.join(project_root, EVOLUTION_DIR_PHASE1)
    if os.path.exists(phase1_dir):
        pattern_phase1 = os.path.join(phase1_dir, "gen_*.txt")
        phase1_files = glob.glob(pattern_phase1)
        if len(phase1_files) > 0:
            def get_gen_number(filename):
                match = re.search(r'gen_(\d+)_', filename)
                return int(match.group(1)) if match else 0
            phase1_files.sort(key=get_gen_number)
            last_phase1_file = phase1_files[-1]
            
            try:
                coords_best_p1, aep_best_p1, _, _, _, _ = load_generation_file(last_phase1_file)
                aep_best_p1_gwh = aep_best_p1 / 1000.0 if aep_best_p1 else 0.0
                
                ax = axes[1]
                circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                              linewidth=2, color='black', label='Park Limit')
                ax.add_patch(circle)
                ax.scatter(coords_best_p1[:, 0], coords_best_p1[:, 1], 
                          s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', linewidths=TURBINE_EDGE_WIDTH, zorder=5)
                ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
                ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
                ax.set_aspect('equal')
                ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
                ax.set_title(f'Phase 1 - Best\nAEP: {aep_best_p1_gwh:.2f} GWh', 
                            fontsize=12, fontweight='bold')
            except Exception as e:
                axes[1].text(0.5, 0.5, f'Error loading\nbest Phase 1', 
                            ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, 'Best Phase 1\nnot found', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'Phase 1 directory\nnot found', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # 3. Knee Point da Fase 2 (com cabeamento)
    knee_solution = df.iloc[idx_knee]
    knee_file_path = knee_solution["File"]
    knee_aep_gwh = knee_solution["AEP_Liquido_MWh"] / 1000.0
    knee_cost_kusd = knee_solution["Custo_USD"] / 1000.0
    
    if not os.path.isabs(knee_file_path):
        if results_dir_name in knee_file_path:
            knee_file_path = os.path.join(project_root, knee_file_path)
        else:
            knee_file_path = os.path.join(project_root, results_dir_name, os.path.basename(knee_file_path))
    
    try:
        coords_knee, n_grupos_knee, substation_pos_knee = load_solution_with_substation(knee_file_path)
        
        if n_grupos_knee is not None:
            n_grupos_to_use = n_grupos_knee
        else:
            n_grupos_to_use = int(np.sqrt(IND_SIZE))
        
        if substation_pos_knee is not None:
            dist_sub = np.linalg.norm(substation_pos_knee)
            if dist_sub > CIRCLE_RADIUS:
                angle = np.arctan2(substation_pos_knee[1], substation_pos_knee[0])
                substation_pos_knee[0] = CIRCLE_RADIUS * np.cos(angle)
                substation_pos_knee[1] = CIRCLE_RADIUS * np.sin(angle)
        
        if substation_pos_knee is not None:
            coords_with_sub = np.vstack([coords_knee, substation_pos_knee.reshape(1, 2)])
            substation_idx = len(coords_knee)
        else:
            distancias_ao_continente = np.linalg.norm(coords_knee - SUBSTATION_CONTINENT, axis=1)
            substation_idx = np.argmin(distancias_ao_continente)
            coords_with_sub = coords_knee
        
        planta, resultados = analisar_layout_completo(
            coords_with_sub, sub=substation_idx, n_grupos=n_grupos_to_use)
        
        comprimento_total_km = resultados["comprimento_total_m"] / 1000.0
        secao_cabo_mm2 = resultados.get("secao_cabo_mm2") or resultados.get("secao_mm2", 0)
        # Usa o custo recalculado para manter consistência com comprimento e bitola
        cost_recalculated_kusd = resultados["custo_total_usd"] / 1000.0
        
        ax = axes[2]
        circle = Circle((0, 0), CIRCLE_RADIUS, fill=False, linestyle='--', 
                      linewidth=2, color='black', label='Park Limit')
        ax.add_patch(circle)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(planta.paths), 10)))
        for i, path in enumerate(planta.paths):
            if len(path) > 1:
                valid_path = [k for k in path if 0 <= k < len(coords_with_sub)]
                if len(valid_path) > 1:
                    x_path = [coords_with_sub[k, 0] for k in valid_path]
                    y_path = [coords_with_sub[k, 1] for k in valid_path]
                    ax.plot(x_path, y_path, '-', linewidth=CABLE_LINEWIDTH, 
                           color=colors[i % len(colors)], alpha=0.7, zorder=4)
        
        ax.scatter(coords_knee[:, 0], coords_knee[:, 1], 
                  s=TURBINE_MARKER_SIZE, c='red', edgecolors='black', linewidths=TURBINE_EDGE_WIDTH, zorder=5)
        
        if substation_pos_knee is not None:
            ax.scatter(substation_pos_knee[0], substation_pos_knee[1],
                      marker='*', s=SUBSTATION_MARKER_SIZE, c='gold', edgecolors='black',
                      linewidths=TURBINE_EDGE_WIDTH, zorder=6)
        
        ax.set_xlim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
        ax.set_ylim(-1.2 * CIRCLE_RADIUS, 1.2 * CIRCLE_RADIUS)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_title(f'Phase 2 - Knee Point\n'
                    f'AEP: {knee_aep_gwh:.2f} GWh | Cost: ${cost_recalculated_kusd:.0f}k USD\n'
                    f'Cable: {secao_cabo_mm2:.0f} mm² | Length: {comprimento_total_km:.2f} km', 
                    fontsize=12, fontweight='bold')
    except Exception as e:
        axes[2].text(0.5, 0.5, f'Error loading\nknee point', 
                    ha='center', va='center', transform=axes[2].transAxes)
        print(f"  ERRO ao carregar knee point: {e}")
    
    plt.tight_layout()
    output_path = os.path.join(project_root, "phase1_vs_phase2_comparison.png")
    output_path_pdf = os.path.join(project_root, "phase1_vs_phase2_comparison.pdf")
    output_path_eps = os.path.join(project_root, "phase1_vs_phase2_comparison.eps")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor='white')
    plt.savefig(output_path_eps, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"Comparação Fase 1 vs Fase 2 salva em:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {output_path_pdf}")
    print(f"  EPS: {output_path_eps}")
    plt.close(fig)

plot_phase1_vs_phase2_comparison(project_root, df, idx_knee, results_dir_name)

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
