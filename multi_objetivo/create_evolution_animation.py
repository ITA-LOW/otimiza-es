"""
Script para criar animação da evolução do algoritmo genético.
Mostra como o melhor layout evolui ao longo das gerações.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import glob
import re

# Importa módulo de cabeamento
try:
    import multi_objetivo.cabling_v3 as cabling_v3
    CABLING_AVAILABLE = True
except ImportError:
    CABLING_AVAILABLE = False
    print("AVISO: Módulo de cabeamento não disponível. Cabeamento não será visualizado.")

# Verifica se pillow está disponível para salvar GIF
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("AVISO: Pillow não está instalado. Instale com: pip install pillow")

# Configurações
CIRCLE_RADIUS = 1300
IND_SIZE = 16
EVOLUTION_DIR_PHASE1 = "pareto_front_results/evolution_phase1"
EVOLUTION_DIR_PHASE2 = "pareto_front_results/evolution"
OUTPUT_ANIMATION = "pareto_front_results/evolution_animation.gif"

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
    for line in lines:
        if line.startswith('aep:'):
            aep = float(line.split(':')[1].strip())
        elif line.startswith('cost:'):
            cost = float(line.split(':')[1].strip())
        elif line.startswith('phase:'):
            phase = int(line.split(':')[1].strip())
        elif line.startswith('n_grupos:'):
            n_grupos = int(line.split(':')[1].strip())
    
    return coords, aep, cost, phase, n_grupos

def get_generation_files():
    """Retorna lista de arquivos de geração ordenados (Fase 1 + Fase 2)."""
    files = []
    
    # Adiciona arquivos da Fase 1
    if os.path.exists(EVOLUTION_DIR_PHASE1):
        pattern_phase1 = os.path.join(EVOLUTION_DIR_PHASE1, "gen_*.txt")
        files.extend(glob.glob(pattern_phase1))
    
    # Adiciona arquivos da Fase 2
    if os.path.exists(EVOLUTION_DIR_PHASE2):
        pattern_phase2 = os.path.join(EVOLUTION_DIR_PHASE2, "gen_*.txt")
        files.extend(glob.glob(pattern_phase2))
    
    # Ordena por número da geração
    def get_gen_number(filename):
        match = re.search(r'gen_(\d+)_', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_gen_number)
    return files

def create_animation():
    """Cria animação da evolução."""
    files = get_generation_files()
    
    if len(files) == 0:
        print(f"Erro: Nenhum arquivo encontrado em {EVOLUTION_DIR_PHASE1} ou {EVOLUTION_DIR_PHASE2}")
        return
    
    print(f"Encontrados {len(files)} arquivos de geração")
    print("Carregando dados e calculando cabeamento para Fase 2...")
    print(f"AVISO: Arquivos antigos podem não ter 'n_grupos' salvo. Usando valor padrão.")
    
    # Carrega todos os dados
    all_coords = []
    all_aep = []
    all_cost = []
    all_gen_numbers = []
    all_phases = []
    all_n_grupos = []
    all_cabling_paths = []  # Armazena caminhos de cabeamento para Fase 2
    
    SUBSTATION_CONTINENT = np.array([[-1.0, -1350.0]])
    
    for filename in files:
        coords, aep, cost, phase, n_grupos = load_generation_file(filename)
        all_coords.append(coords)
        all_aep.append(aep)
        all_cost.append(cost)
        all_phases.append(phase)
        all_n_grupos.append(n_grupos)
        
        # Extrai número da geração do nome do arquivo
        match = re.search(r'gen_(\d+)_', filename)
        gen_num = int(match.group(1)) if match else 0
        all_gen_numbers.append(gen_num)
        
        # Calcula cabeamento para Fase 2
        cabling_paths = None
        if phase == 2 and CABLING_AVAILABLE:
            try:
                # Encontra ponto de coleta mais próximo do continente
                distancias_ao_continente = np.linalg.norm(coords - SUBSTATION_CONTINENT, axis=1)
                ponto_de_coleta_idx = np.argmin(distancias_ao_continente)
                
                # Usa n_grupos do arquivo ou tenta inferir do custo
                # Se não tem n_grupos salvo, usa valor padrão baseado no número de turbinas
                if n_grupos is not None:
                    n_grupos_to_use = n_grupos
                else:
                    # Valor padrão: sqrt do número de turbinas (4 para 16 turbinas)
                    n_grupos_to_use = int(np.sqrt(IND_SIZE))
                
                # Calcula cabeamento
                planta, _ = cabling_v3.analisar_layout_completo(
                    coords, sub=ponto_de_coleta_idx, n_grupos=n_grupos_to_use)
                cabling_paths = planta.paths
                
                # Debug: mostra quantos paths foram calculados
                if len(all_coords) % 50 == 0:  # Print a cada 50 frames
                    print(f"  Frame {len(all_coords)}: {len(cabling_paths)} paths de cabeamento, n_grupos={n_grupos_to_use}")
                    if len(cabling_paths) > 0:
                        print(f"    Exemplo path[0]: {cabling_paths[0]}")
            except Exception as e:
                # Mostra erro apenas para debug (pode ser removido depois)
                if len(all_coords) % 50 == 0:
                    print(f"  Erro ao calcular cabeamento no frame {len(all_coords)}: {e}")
                cabling_paths = None
        
        all_cabling_paths.append(cabling_paths)
        
        # Mostra progresso a cada 100 arquivos
        if len(all_coords) % 100 == 0:
            print(f"  Processados {len(all_coords)}/{len(files)} arquivos...")
    
    # Configura figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Evolução do Algoritmo Genético - Fase 2', fontsize=16, fontweight='bold')
    
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
    
    # Lista para armazenar linhas de cabeamento (será atualizada a cada frame)
    # Usa lista mutável para poder modificar dentro de animate
    cabling_lines_container = []
    
    # Linhas de evolução
    line_aep, = ax2.plot([], [], 'b-', linewidth=2, label='AEP (GWh)', marker='o', markersize=4)
    
    # Cria eixos secundários para custo ANTES de criar a linha
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
        # Adiciona margem de 2% em cada lado
        cost_ymin = cost_min_real - cost_range * 0.02
        cost_ymax = cost_max_real + cost_range * 0.02
        print(f"Escala de custo calculada: ${cost_min_real:,.0f} - ${cost_max_real:,.0f}")
    else:
        cost_ymin = None
        cost_ymax = None
        print("AVISO: Nenhum dado de custo encontrado para Fase 2!")
    
    def animate(frame):
        """Função de animação para cada frame."""
        if frame >= len(all_coords):
            return scatter, title_text, line_aep, line_cost
        
        coords = all_coords[frame]
        aep = all_aep[frame]
        cost = all_cost[frame]
        gen_num = all_gen_numbers[frame]
        phase = all_phases[frame]
        cabling_paths = all_cabling_paths[frame]
        
        # Remove linhas de cabeamento anteriores
        for line in cabling_lines_container:
            try:
                line.remove()
            except:
                pass
        cabling_lines_container.clear()
        
        # Desenha linhas de cabeamento (apenas Fase 2)
        if phase == 2 and cabling_paths is not None and len(cabling_paths) > 0:
            # Debug: verifica se paths estão sendo processados
            if frame < 5 or frame % 50 == 0:
                print(f"  Desenhando frame {frame}: {len(cabling_paths)} paths, coords shape: {coords.shape}")
            
            # Usa cores distintas para cada string/grupo
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(cabling_paths), 10)))
            paths_drawn = 0
            for i, path in enumerate(cabling_paths):
                if len(path) > 1:
                    # Filtra índices válidos (dentro do range de coordenadas)
                    valid_path = [k for k in path if 0 <= k < len(coords)]
                    if len(valid_path) > 1:
                        x_path = [coords[k, 0] for k in valid_path]
                        y_path = [coords[k, 1] for k in valid_path]
                        line, = ax1.plot(x_path, y_path, '-', linewidth=2.5, 
                                        color=colors[i % len(colors)], alpha=0.8, zorder=4,
                                        label=f'String {i+1}' if i < 5 else '')
                        cabling_lines_container.append(line)
                        paths_drawn += 1
            
            # Debug: mostra quantos paths foram desenhados
            if frame < 5 or frame % 50 == 0:
                print(f"    Paths desenhados: {paths_drawn}/{len(cabling_paths)}")
        
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
        aeps_so_far = [a/1000 for a in all_aep[:frame+1]]  # Converte para GWh
        costs_so_far = [c for c in all_cost[:frame+1] if c is not None]
        
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
                # Reaplica limites fixos se foram definidos
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
            
            # Usa escala real do custo (calculada anteriormente)
            if len(phase2_indices) > 0 and cost_ymin is not None and cost_ymax is not None:
                # Atualiza limites do eixo de custo
                ax2_twin.set_ylim(cost_ymin, cost_ymax)
                ax2_twin.set_visible(True)
                # Força atualização
                ax2_twin.relim()
                ax2_twin.autoscale_view()
            elif len(phase2_indices) > 0:
                # Se há dados mas não calculou escala, usa autoscale
                phase2_costs_current = [all_cost[i] for i in phase2_indices if all_cost[i] is not None]
                if len(phase2_costs_current) > 0:
                    cost_min_curr = min(phase2_costs_current)
                    cost_max_curr = max(phase2_costs_current)
                    cost_range_curr = cost_max_curr - cost_min_curr
                    ax2_twin.set_ylim(cost_min_curr - cost_range_curr * 0.05, 
                                     cost_max_curr + cost_range_curr * 0.05)
                    ax2_twin.set_visible(True)
                else:
                    ax2_twin.set_visible(False)
            else:
                ax2_twin.set_visible(False)
        
        return scatter, title_text, line_aep, line_cost
    
    # Cria animação
    print("Criando animação...")
    anim = animation.FuncAnimation(fig, animate, frames=len(all_coords), 
                                   interval=100, blit=False, repeat=True)
    
    # Salva animação
    if not PILLOW_AVAILABLE:
        print("Erro: Pillow não está instalado. Não é possível salvar GIF.")
        print("Instale com: pip install pillow")
        print("Mostrando animação interativa...")
        plt.show()
    else:
        print(f"Salvando animação em {OUTPUT_ANIMATION}...")
        try:
            anim.save(OUTPUT_ANIMATION, writer='pillow', fps=10)
            print(f"Animação salva com sucesso em {OUTPUT_ANIMATION}!")
        except Exception as e:
            print(f"Erro ao salvar animação: {e}")
            print("Mostrando animação interativa...")
            plt.show()
    
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(EVOLUTION_DIR_PHASE1) and not os.path.exists(EVOLUTION_DIR_PHASE2):
        print(f"Erro: Diretórios de evolução não encontrados.")
        print(f"Procurei em: {EVOLUTION_DIR_PHASE1} e {EVOLUTION_DIR_PHASE2}")
        print("Execute primeiro o algoritmo de otimização.")
    else:
        create_animation()

