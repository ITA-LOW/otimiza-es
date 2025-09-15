# cabling.py (Refatorado)

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math
import json # Adicionado para imprimir dicionários de forma legível

# ================================================
# FUNÇÕES AUXILIARES E CLASSES
# <<< NENHUMA ALTERAÇÃO NESTA SEÇÃO >>>
# Sua lógica original de classes e cálculo de distância está 100% preservada.
# ================================================

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Cable:
    """Cabo entre turbinas, seleciona seção discreta de acordo com tabela industrial."""
    SECTION_TABLE = {
        50: 0.49, 
        70: 0.34, 
        95: 0.25, 
        120: 0.20, 
        150: 0.16, 
        185: 0.13, 
        240: 0.10,
    }

    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.dI = 2.3
        self.I = self.Pn / (math.sqrt(3) * self.Vn)
        self.A_continuous = self.I / self.dI
        self.A = None
        self.R_km = None
        self.R = None
        self.Pj = None
        self.C = 0
        self.Ctot = 0

    def assign_section(self, section):
        self.A = section
        self.R_km = self.SECTION_TABLE[section]
        self.R = self.R_km * (self.lc / 1000)
        self.Pj = 3 * (self.I ** 2) * self.R

    def summary(self):
        return {
            'length_m': self.lc, 'current_A': round(self.I, 2),
            'A_calc_mm2': round(self.A_continuous, 2), 'A_sel_mm2': self.A,
            'R_km': self.R_km, 'R_tot_Ohm': round(self.R, 6), 'Pj_W': round(self.Pj, 2),
            'cost_R$_per_m': self.C, 'cost_total_R$': self.Ctot
        }

class Turbine:
    """Turbina com coordenadas e potência."""
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y

class Plant:
    INDUSTRIAL_CABLE_COSTS = {
        50: 69.52, 
        70: 97.33, 
        95: 132.09, 
        120: 166.85, 
        150: 208.56, 
        185: 257.22, 
        240: 333.70
    }

    def __init__(self, Vn, Tr, paths):
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.Cb = []
        self.cables_flat = []
        self.Pjtot = 0
        self.Ctot = 0
        self.lay_cables()
        self.uniform_cable_section()
        self.calculate_losses()
        self.calculate_cost()
        
    def lay_cables(self):
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Ptransmitted = 0
            for k in range(len(path)-1):
                curr = path[k]
                nxt = path[k+1]
                Ptransmitted += self.Tr[curr].P
                length = calculate_distance(self.Tr[curr].x, self.Tr[curr].y, self.Tr[nxt].x, self.Tr[nxt].y)
                cable_path.append(Cable(length, self.Vn, Ptransmitted))
            self.Cb.append(cable_path)
        self.cables_flat = [c for sub in self.Cb for c in sub]

    def uniform_cable_section(self):
        max_cont = max(c.A_continuous for c in self.cables_flat) if self.cables_flat else 0
        for sec in sorted(Cable.SECTION_TABLE):
            if sec >= max_cont:
                chosen = sec
                break
        else:
            chosen = max(Cable.SECTION_TABLE)
        # print(f"Aplicando seção única a todos os trechos: {chosen} mm²") # Comentado para não poluir a saída quando usado como lib
        for cable in self.cables_flat:
            cable.assign_section(chosen)

    def calculate_losses(self):
        self.Pjtot = sum(c.Pj for c in self.cables_flat)

    def calculate_cost(self):
        sec = self.cables_flat[0].A if self.cables_flat else 0
        custo_m = self.INDUSTRIAL_CABLE_COSTS.get(sec, 0)
        total = 0
        for cable in self.cables_flat:
            cable.C = custo_m
            cable.Ctot = cable.lc * custo_m
            total += cable.Ctot
        self.Ctot = total

    def get_max_calculated_section(self):
        if not self.cables_flat: return 0
        return self.cables_flat[0].A

    def print_turbines(self):
        print(f"Number of turbines: {len(self.Tr)}")
        print(f"{'Details':<15}{'P (MW)':>15}{'x(m)':>15}{'y(m)':>15}")
        for k, T in enumerate(self.Tr):
            print(f"{k:<15}{T.P/1e6:>15.2f}{T.x:>15.2f}{T.y:>15.2f}")
        print()
        
    def print_cables(self):
        total_length = sum(c.lc for c in self.cables_flat)
        print("Cable routing report")
        print(f"{'Connection':<15}{'Length(m)':>15}{'Section(mm²)':>15}{'P (MW)':>15}{'P_J (kW)':>15}{'Cost milions (R$)':>22}{'Resistance':>22}")
        for j, path in enumerate(self.Cb):
            nodes = self.paths[j]
            for k, cable in enumerate(path):
                conn = f"{nodes[k]} --> {nodes[k+1]}"
                print(f"{conn:<15}{cable.lc:15.1f}{cable.A:15.1f}{cable.Pn/1e6:15.2f}{cable.Pj/1e3:15.2f}{cable.Ctot/1e6:21.2f}{cable.R:21.2f}")
        print(f"Total length: {total_length:,.2f} m")

# ================================================
# LÓGICA DE AGRUPAMENTO E PATHS
# <<< MODIFICAÇÃO >>>
# As funções agora aceitam 'coordenadas' e 'indice_subestacao' como argumentos
# em vez de depender de variáveis globais. A lógica interna é a mesma.
# ================================================

def agrupar_torres_por_cosseno(coordenadas, indice_subestacao, n_grupos=4):
    """Agrupa turbinas por similaridade angular"""
    vetores = coordenadas - coordenadas[indice_subestacao]
    mascara = np.arange(len(coordenadas)) != indice_subestacao
    
    kmeans = KMeans(n_clusters=n_grupos, random_state=33)
    grupos = kmeans.fit_predict(cosine_similarity(vetores[mascara]))
    
    indices_originais = np.where(mascara)[0].tolist()
    return [sorted([indices_originais[i] for i in np.where(grupos == id)[0]]) for id in range(n_grupos)]

def balancear_grupos(grupos, vectors, alvo=4):
    """Balanceia grupos mantendo similaridade"""
    # <<< LÓGICA 100% MANTIDA >>>
    def similaridade_grupo(turbina, grupo):
        return np.mean([cosine_similarity([vectors[turbina]], [vectors[t]])[0][0] for t in grupo]) if grupo else 0
    balanced = [g.copy() for g in grupos]
    for _ in range(10):
        tamanhos = [len(g) for g in balanced]
        if all(t == alvo for t in tamanhos): break
        fontes = sorted([(len(g), i) for i, g in enumerate(balanced) if len(g) > alvo], reverse=True)
        for _, fonte in fontes:
            while len(balanced[fonte]) > alvo:
                turbinas = balanced[fonte]
                sims = [similaridade_grupo(t, turbinas) for t in turbinas]
                turbina = turbinas.pop(np.argmin(sims))
                melhor_dest = np.argmax([similaridade_grupo(turbina, g) if i != fonte and len(g) < alvo else -1 for i, g in enumerate(balanced)])
                balanced[melhor_dest].append(turbina)
    melhorias = True
    while melhorias:
        melhorias = False
        for i in range(len(balanced)):
            for j in range(i+1, len(balanced)):
                copia_i, copia_j = list(balanced[i]), list(balanced[j])
                for t1 in copia_i:
                    for t2 in copia_j:
                        if t1 not in balanced[i] or t2 not in balanced[j]: continue
                        sim_original = (similaridade_grupo(t1, balanced[i]) + similaridade_grupo(t2, balanced[j]))
                        sim_novo = (similaridade_grupo(t2, balanced[i]) + similaridade_grupo(t1, balanced[j]))
                        if sim_novo > sim_original:
                            try:
                                balanced[i].remove(t1); balanced[j].remove(t2)
                                balanced[i].append(t2); balanced[j].append(t1)
                                melhorias = True
                            except ValueError: continue
    return balanced

# ================================================
# <<< NOVAS FUNÇÕES ORQUESTRADORAS >>>
# Estas funções encapsulam a lógica que estava no `if __name__ == "__main__"`
# tornando o script reutilizável.
# ================================================

def analisar_layout_completo(coordenadas, substation_idx, n_grupos=4, Vn=33e3, P_turbina=3.35e6):
    """
    Função principal que encapsula toda a análise de cabeamento para um dado layout.
    Recebe as coordenadas e retorna o objeto Plant e um dicionário de resultados.
    """
    # 1. Roteamento (lógica original do if __name__ == '__main__')
    grupos = agrupar_torres_por_cosseno(coordenadas, substation_idx, n_grupos)
    vectors = coordenadas - coordenadas[substation_idx]
    grupos_balanceados = balancear_grupos(grupos, vectors)
    paths = []
    for grupo in grupos_balanceados:
        distancias = [np.linalg.norm(coordenadas[t] - coordenadas[substation_idx]) for t in grupo]
        ordenado = [x for _, x in sorted(zip(distancias, grupo), reverse=True)]
        paths.append(ordenado + [substation_idx])
        
    # 2. Análise da Planta
    turbinas = [Turbine(P_turbina, x, y) for x, y in coordenadas]
    planta = Plant(Vn=Vn, Tr=turbinas, paths=paths)
    
    # 3. Coleta de Resultados
    total_length = sum(c.lc for c in planta.cables_flat) if planta.cables_flat else 0
    perda_anual_MWh = planta.Pjtot * 8760 / 1e6
    cotDolar = 0.1722 # Conversão do real para dólar 21/04/25
    
    resultados = {
        'custo_total_usd': planta.Ctot * cotDolar,
        'comprimento_total_m': total_length,
        'perda_total_kw': planta.Pjtot / 1e3,
        'perda_anual_mwh': perda_anual_MWh,
        'secao_cabo_mm2': planta.get_max_calculated_section()
    }
    return planta, resultados

def plotar_cabeamento(planta, coordenadas, substation_idx, titulo="Layout do Cabeamento", output_filename=None):
    """
    Gera e, opcionalmente, salva um gráfico visual do layout do cabeamento.
    """
    font_index = 25
    font_config = {
        'title': font_index * 1.2, 'axis_label': font_index * 1.1, 'tick_label': font_index * 0.8,
        'legend': font_index * 0.8, 'node_label': font_index * 0.7,
    }

    plt.figure(figsize=(14, 12))
    num_paths = len(planta.paths)
    colormap = plt.colormaps.get_cmap('tab20')
    cores = [colormap(i) for i in np.linspace(0, 1, num_paths)]
    
    for i, path in enumerate(planta.paths):
        x = [coordenadas[node][0] for node in path]
        y = [coordenadas[node][1] for node in path]
        plt.plot(x, y, 'o-', linewidth=2, color=cores[i], markersize=8, markeredgecolor='black', label=f'String {i+1}')
        for node in path:
            if node != substation_idx:
                plt.text(coordenadas[node][0] + 20, coordenadas[node][1] + 20, str(node), fontsize=font_config['node_label'], color='black', weight='bold')
    
    plt.scatter(coordenadas[substation_idx][0], coordenadas[substation_idx][1], c='yellow', s=300, marker='*', edgecolors='black', label='Substation')
    
    plt.title(titulo, fontsize=font_config['title'], weight='bold')
    plt.xlabel("X (m)", fontsize=font_config['axis_label'])
    plt.ylabel("Y (m)", fontsize=font_config['axis_label'])
    plt.tick_params(axis='both', which='major', labelsize=font_config['tick_label'])
    plt.legend(loc='best', fontsize=font_config['legend'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_filename}")
        plt.close()
    else:
        plt.show()

# ================================================
# DADOS E EXECUÇÃO (MODO DE TESTE)
# <<< MODIFICAÇÃO >>>
# O bloco principal agora serve como um teste para as novas funções,
# usando os dados hardcoded originais. Isso garante que a refatoração
# não quebrou a lógica e permite testar este arquivo de forma isolada.
# ================================================

if __name__ == "__main__":
    print("--- Executando cabling.py em modo de teste ---")
    
    # Os dados hardcoded agora vivem apenas dentro deste bloco de teste.
    x_opt = [-1350, -134.92021516964914, 475.55640005887045, 47.341265591348645, -291.87618347006656, -467.2781430286087, 349.0231420338762, 1203.3903276016408, 1080.9073355869466, 247.26455478961583, -725.2478601140702, -1100.162845125174, -1294.5429885693968, -949.8004833332534, -609.7234640375709, 181.263092795678, 1148.844649045738]
    y_opt = [0.0, -801.113357244488, 275.0418014163371, 914.8073394319176, 440.4982633422105, -288.84091129919307, -451.7884456528136, 94.70985786260522, 722.1799528444428, 1276.1492031053472, 1078.8503732558997, 588.5083327203014, -117.19792126546781, -633.071166755402, -1130.428394544164, -1287.2676446864946, -608.0829014957618]
    
    substation_idx_teste = 0
    coordenadas_teste = np.array(list(zip(x_opt, y_opt)))
    
    # 1. Chamar a nova função orquestradora
    planta_teste, resultados_teste = analisar_layout_completo(coordenadas_teste, substation_idx_teste)
    
    # 2. Imprimir os relatórios (usando os métodos da classe, como antes)
    planta_teste.print_turbines()
    planta_teste.print_cables()
    
    # 3. Imprimir o resumo (usando o dicionário retornado, que será usado pelo script externo)
    print("\n=== Resumo dos Resultados (dicionário retornado) ===")
    print(json.dumps(resultados_teste, indent=2))

    # 4. Gerar o gráfico interativo (como antes)
    titulo_teste = f"Visualização de Teste do Módulo\nCusto: $ {resultados_teste['custo_total_usd']/1e6:.2f} mi | Perda Anual: {resultados_teste['perda_anual_mwh']:.2f} MWh"
    plotar_cabeamento(planta_teste, coordenadas_teste, substation_idx_teste, titulo=titulo_teste)