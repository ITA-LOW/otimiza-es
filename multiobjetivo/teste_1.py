import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
import math
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =============================================
# NOVAS CLASSES PARA MODELAGEM DE CABOS E USINA
# =============================================

def calculate_distance(x1, y1, x2, y2):
    """Distância Euclidiana entre dois pontos."""
    return math.hypot(x2 - x1, y2 - y1)

class Cable:
    """Cabo entre turbinas com cálculo de perda e seção mínima."""
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc          # comprimento do cabo (m)
        self.Vn = Vn          # tensão nominal (V)
        self.Pn = Pn          # potência transmitida (W)
        self.Qi = Qi
        self.dI = 2.3         # densidade de corrente mínima (A/mm²)
        # Corrente trifásica (A); se Pn = 0, então I = 0
        self.I = self.Pn / (math.sqrt(3) * self.Vn) if self.Pn > 0 else 0.0
        # Seção mínima em mm² (garante pelo menos 50 mm² mesmo que I seja baixo)
        self.A = max(self.I / self.dI, 50)
        # Resistividade do cobre e coeficiente de temperatura
        self.p = 0.0173e-6    # Ω·m a 20 °C
        self.alpha = 0.00393  # coeficiente de temperatura do cobre
        # Resistividade a 90 °C
        self.p90 = self.p * (1 + self.alpha * (90 - 20))
        # Resistência total do cabo: R = p90 · lc / (A·1e-6)
        # (A em mm² → A·1e-6 em m²)
        self.R = (self.p90 * self.lc) / (self.A * 1e-6)
        # Perda Joule trifásica (W): Pj = 3 · I² · R
        self.Pj = self.__Pj__()
        # Custo por metro inicial (será ajustado em Plant.calculate_cost)
        self.C = 2e3
        # Custo total inicial do cabo (será recalculado em Plant.calculate_cost)
        self.Ctot = self.__Ctot__()

    def __Pj__(self):
        return 3 * (self.I ** 2) * self.R

    def __Ctot__(self):
        return self.lc * self.C

class Turbine:
    """Turbina com potência (W) e coordenadas (m)."""
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y

class Plant:
    """Parque eólico que monta cabeamento, calcula perdas e custo total."""
    # Tabela de custos industriais por seção transversal (mm² → R$/m)
    INDUSTRIAL_CABLE_COSTS = {
        50: 2e3,
        70: 3e3,
        95: 4e3,
        120: 5e3,
        150: 6e3,
        185: 7e3,
        240: 9e3,
        300: 11e3,
        400: 14e3
    }

    def __init__(self, Vn, Tr, paths):
        """
        Vn: tensão nominal (V)
        Tr: lista de objetos Turbine
        paths: lista de listas, onde cada lista é uma sequência de índices de turbinas (nós)
        """
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.Cb = []           # lista de caminhos, cada caminho é lista de objetos Cable
        self.cables_flat = []  # lista achatada de todos os objetos Cable
        self.Pjtot = 0         # perda Joule total (W)
        self.Ctot = 0          # custo total (R$)
        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        """
        Para cada caminho em self.paths, percorre pares de nós
        e instancia um Cable(length, Vn, Ptransmitida).
        Ptransmitida é soma das potências de todas as turbinas a jusante.
        """
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Ptransmitted = 0
            for k in range(len(path) - 1):
                current = path[k]
                next_node = path[k + 1]
                Ptransmitted += self.Tr[current].P
                length = calculate_distance(
                    self.Tr[current].x, self.Tr[current].y,
                    self.Tr[next_node].x, self.Tr[next_node].y
                )
                cabo = Cable(length, self.Vn, Ptransmitted)
                cable_path.append(cabo)
            self.Cb.append(cable_path)

        # Achata a lista de listas em uma só lista de objetos Cable
        self.cables_flat = [item for sublist in self.Cb for item in sublist]

    def calculate_losses(self):
        """Soma as perdas Joule de todos os cabos da planta."""
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)

    def get_max_calculated_section(self):
        """
        Retorna a maior seção transversal calculada (cable.A)
        entre todos os cabos da planta.
        """
        if not self.cables_flat:
            return 0.0
        return max(cable.A for cable in self.cables_flat)

    def calculate_cost(self):
        """
        Recalcula o custo total dos cabos usando a seção mínima necessária para a planta.
        1) Encontra maior seção A entre todos os cabos.
        2) Busca na tabela INDUSTRIAL_CABLE_COSTS a primeira seção padrão ≥ A.
        3) Usa o custo por metro correspondente para TODOS os trechos de cabo.
        4) Atualiza cada cable.C e cable.Ctot e soma para self.Ctot.
        """
        maior_secao_necessaria = self.get_max_calculated_section()
        secoes_padrao = sorted(self.INDUSTRIAL_CABLE_COSTS.keys())
        custo_por_metro = None
        secao_usada = None
        for secao in secoes_padrao:
            if secao >= maior_secao_necessaria:
                secao_usada = secao
                custo_por_metro = self.INDUSTRIAL_CABLE_COSTS[secao]
                break
        if custo_por_metro is None:
            secao_usada = secoes_padrao[-1]
            custo_por_metro = self.INDUSTRIAL_CABLE_COSTS[secao_usada]

        total_cost = 0
        for cable in self.cables_flat:
            cable.C = custo_por_metro
            cable.Ctot = cable.lc * cable.C
            total_cost += cable.Ctot
        self.Ctot = total_cost

    def print_turbines(self):
        """Imprime relatório das turbinas (índice, potência e coordenadas)."""
        print(f"Number of turbines: {len(self.Tr)}")
        print(f"{'Index':<6}{'P (MW)':>10}{'x (m)':>12}{'y (m)':>12}")
        for k, T in enumerate(self.Tr):
            print(f"{k:<6}{T.P/1e6:>10.2f}{T.x:>12.2f}{T.y:>12.2f}")
        print()

    def print_cables(self):
        """
        Imprime relatório detalhado de cada trecho de cabo:
        conexão, comprimento, seção, P transmissível, perda Joule, custo e resistência.
        """
        print("Cable routing report")
        print(f"{'Connection':<15}{'Length (m)':>12}{'Section (mm²)':>15}"
              f"{'P (MW)':>12}{'Pj (kW)':>12}{'Cost (M R$)':>14}{'Resistance (Ω)':>16}")
        for path_idx, path in enumerate(self.Cb):
            nodes = self.paths[path_idx]
            for k, cable in enumerate(path):
                conn = f"{nodes[k]} -> {nodes[k+1]}"
                print(f"{conn:<15}"
                      f"{cable.lc:12.1f}"
                      f"{cable.A:15.1f}"
                      f"{cable.Pn/1e6:12.2f}"
                      f"{cable.Pj/1e3:12.2f}"
                      f"{cable.Ctot/1e6:14.2f}"
                      f"{cable.R:16.4f}")
        print()

# =============================================
# FUNÇÕES DE AGRUPAMENTO POR SIMILARIDADE DE COSSENO
# =============================================

def agrupar_torres_por_cosseno(coordenadas, indice_subestacao, n_grupos=4):
    """Agrupa turbinas por similaridade angular (cosine similarity)."""
    vetores = coordenadas - coordenadas[indice_subestacao]
    mascara = np.arange(len(coordenadas)) != indice_subestacao

    kmeans = KMeans(n_clusters=n_grupos, random_state=42)
    grupos = kmeans.fit_predict(cosine_similarity(vetores[mascara]))

    indices_originais = np.where(mascara)[0].tolist()
    return [
        sorted([indices_originais[i] for i in np.where(grupos == gid)[0]])
        for gid in range(n_grupos)
    ]

def balancear_grupos(grupos, vectors, alvo=4):
    """
    Balanceia grupos mantendo similaridade:
    1) Move turbinas de grupos maiores que 'alvo' para os que estão menores.
    2) Tenta swaps que aumentem similaridade.
    """
    def similaridade_grupo(turbina, grupo):
        return np.mean([
            cosine_similarity([vectors[turbina]], [vectors[turbina_ref]])[0][0]
            for turbina_ref in grupo
        ]) if grupo else 0

    balanced = [g.copy() for g in grupos]

    # Primeira fase: redistribuir até cada grupo ter tamanho 'alvo'
    for _ in range(10):
        tamanhos = [len(g) for g in balanced]
        if all(t == alvo for t in tamanhos):
            break
        fontes = sorted([(len(g), idx) for idx, g in enumerate(balanced) if len(g) > alvo], reverse=True)
        for _, fonte in fontes:
            while len(balanced[fonte]) > alvo:
                turbinas = balanced[fonte]
                sims = [similaridade_grupo(t, turbinas) for t in turbinas]
                turbina = turbinas.pop(np.argmin(sims))
                dest_sim = [
                    similaridade_grupo(turbina, balanced[d]) if d != fonte and len(balanced[d]) < alvo else -1
                    for d in range(len(balanced))
                ]
                melhor_dest = int(np.argmax(dest_sim))
                balanced[melhor_dest].append(turbina)

    # Segunda fase: swaps entre pares de grupos para aumentar similaridade
    melhorias = True
    while melhorias:
        melhorias = False
        for i in range(len(balanced)):
            for j in range(i + 1, len(balanced)):
                copia_i = balanced[i].copy()
                copia_j = balanced[j].copy()
                for t1 in copia_i:
                    for t2 in copia_j:
                        if t1 not in balanced[i] or t2 not in balanced[j]:
                            continue
                        sim_orig = similaridade_grupo(t1, balanced[i]) + similaridade_grupo(t2, balanced[j])
                        sim_novo = similaridade_grupo(t2, balanced[i]) + similaridade_grupo(t1, balanced[j])
                        if sim_novo > sim_orig:
                            balanced[i].remove(t1)
                            balanced[j].remove(t2)
                            balanced[i].append(t2)
                            balanced[j].append(t1)
                            melhorias = True
    return balanced

# =============================================
# CONFIGURAÇÃO DO ALGORITMO GENÉTICO (DEAP)
# =============================================

IND_SIZE = 16         # número de nós (1 subestação + 15 turbinas)
CIRCLE_RADIUS = 1300  # raio máximo permitido (m)
N_DIAMETERS = 260     # distância mínima entre turbinas (m)
TURB_POWER = 3.35e6   # potência nominal de cada turbina (W)
CABLE_VOLTAGE = 33e3  # tensão nominal do cabo (V)

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def create_individual_from_coordinates(coords):
    """
    Gera indivíduo: [x0, y0, x1, y1, ..., x15, y15, central_index]
    Inicialmente, central_index = 0.
    """
    flat = np.array(coords).flatten().tolist()
    return creator.Individual(flat + [0])

initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2

def enforce_circle(individual):
    """
    Garante que todos os 16 pontos (2 genes por ponto) fiquem dentro do círculo de raio CIRCLE_RADIUS.
    Se estiver fora, projeta para a circunferência.
    """
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            individual[2*i] = CIRCLE_RADIUS * math.cos(angle)
            individual[2*i + 1] = CIRCLE_RADIUS * math.sin(angle)

def mutate(individual, mu, sigma, indpb):
    """
    Mutação:
    - Coordenadas (gaussiana) com probabilidade indpb.
    - Ajusta de novo para dentro do círculo.
    - Gene discreto (central_index) troca com prob. indpb.
    """
    coords = np.array(individual[:2*IND_SIZE])
    if random.random() < indpb:
        for idx in range(len(coords)):
            coords[idx] += random.gauss(mu, sigma)
    for i in range(IND_SIZE):
        x, y = coords[2*i], coords[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            coords[2*i] = CIRCLE_RADIUS * math.cos(angle)
            coords[2*i + 1] = CIRCLE_RADIUS * math.sin(angle)
    central = individual[-1]
    if random.random() < indpb:
        central = random.randint(0, IND_SIZE-1)
    new_ind = list(coords) + [central]
    return creator.Individual(new_ind),

def mate(ind1, ind2, alpha=0.5):
    """
    Cruzamento blend para coordenadas:
    ind1[i] = (1-γ)*g1 + γ*g2 ; ind2[i] = γ*g1 + (1-γ)*g2, γ ∈ [-α, 1+α]
    Para o gene discreto, troca com prob. 0.5.
    """
    for i in range(2*IND_SIZE):
        g1, g2 = ind1[i], ind2[i]
        gamma = random.uniform(-alpha, 1+alpha)
        ind1[i] = (1-gamma)*g1 + gamma*g2
        ind2[i] = gamma*g1 + (1-gamma)*g2
    if random.random() < 0.5:
        ind1[-1], ind2[-1] = ind2[-1], ind1[-1]
    return ind1, ind2

toolbox.register("mate", mate, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=5)

def evaluate_multiobjetivo(individual,
                           turb_atrbt_data=getTurbAtrbtYAML("iea37-335mw.yaml"),
                           wind_rose_data=getWindRoseYAML("iea37-windrose.yaml")):
    """
    1) Extrai coordenadas e índice do nó central.
    2) Penalidades: fora do círculo e pares muito próximos (< N_DIAMETERS).
    3) Calcula AEP via calcAEP; AEP líquido = sum(AEP) - penalidades.
    4) Agrupa turbinas por similaridade de cosseno em n_clusters = 8.
    5) Balanceia grupos (alvo default = 8).
    6) Para cada grupo balanceado, ordena por distância ao central (decrescente) e adiciona o central no fim → path.
    7) Monta lista Tr de Turbine (P=0 para o central).
    8) Instancia Plant com o novo construtor e recupera Pjtot.
    Retorna (AEP_liquido, -cable_loss).
    """
    coords_flat = individual[:2*IND_SIZE]
    turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    central_index = int(individual[-1])
    if central_index < 0 or central_index >= IND_SIZE:
        central_index = 0

    mask_inside = (turb_coords[:,0]**2 + turb_coords[:,1]**2) <= CIRCLE_RADIUS**2
    penalty_out = np.sum(~mask_inside) * 1e6

    diff = turb_coords.reshape(IND_SIZE, 1, 2) - turb_coords.reshape(1, IND_SIZE, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i_upper, j_upper = np.triu_indices(IND_SIZE, k=1)
    penalty_close = np.sum(dist_matrix[i_upper, j_upper] < N_DIAMETERS) * 1e6

    penalty = penalty_out + penalty_close

    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    aep_liquido = np.sum(aep) - penalty

    try:
        grupos = agrupar_torres_por_cosseno(turb_coords, central_index, n_grupos=8)
        vectors = turb_coords - turb_coords[central_index]
        grupos_bal = balancear_grupos(grupos, vectors)

        valid_paths = []
        for grupo in grupos_bal:
            distancias = [np.linalg.norm(turb_coords[t] - turb_coords[central_index]) for t in grupo]
            ordenado = [node for _, node in sorted(zip(distancias, grupo), reverse=True)]
            valid_paths.append(ordenado + [central_index])

        Tr = []
        for i, (x, y) in enumerate(turb_coords):
            if i == central_index:
                Tr.append(Turbine(0, x, y))
            else:
                Tr.append(Turbine(TURB_POWER, x, y))

        plant = Plant(CABLE_VOLTAGE, Tr, valid_paths)
        cable_loss = plant.Pjtot

    except Exception as e:
        print("Erro no cálculo de cabos:", e)
        cable_loss = 1e8

    return aep_liquido, -cable_loss

toolbox.register("evaluate", evaluate_multiobjetivo)

def plotar_grafo_com_subtree_sum(coords, caminhos, turbine_powers, central):
    """
    Plota o grafo resultante, marcando em cada aresta a soma de potências a jusante.
    """
    vizinhos = {i: set() for i in range(len(coords))}
    for caminho in caminhos:
        if len(caminho) < 2:
            continue
        for i in range(len(caminho) - 1):
            a, b = caminho[i], caminho[i + 1]
            vizinhos[a].add(b)
            vizinhos[b].add(a)

    tree_filhos = {i: [] for i in range(len(coords))}
    visitados = set()

    def dfs(no, pai):
        visitados.add(no)
        for nb in vizinhos[no]:
            if nb == pai:
                continue
            if nb not in visitados:
                tree_filhos[no].append(nb)
                dfs(nb, no)

    dfs(central, None)

    subtree_power = {}
    def compute_subtree(no):
        total = turbine_powers[no]
        for filho in tree_filhos[no]:
            total += compute_subtree(filho)
        subtree_power[no] = total
        return total

    compute_subtree(central)

    plt.figure(figsize=(10, 10))
    for i, (x, y) in enumerate(coords):
        plt.plot(x, y, 'bo')
        plt.text(x + 10, y + 10, str(i), fontsize=9)

    def plot_edge(pai, filho):
        x_vals = [coords[pai][0], coords[filho][0]]
        y_vals = [coords[pai][1], coords[filho][1]]
        plt.plot(x_vals, y_vals, 'k-', linewidth=1)
        xm = (coords[pai][0] + coords[filho][0]) / 2
        ym = (coords[pai][1] + coords[filho][1]) / 2
        plt.text(xm, ym, f"{(subtree_power[filho]/1e6):.2f}", color='red', fontsize=8)
        for f2 in tree_filhos[filho]:
            plot_edge(filho, f2)

    for child in tree_filhos[central]:
        plot_edge(central, child)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Grafo com potência agregada (subtree sum) nas arestas")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("grafo.png")
    plt.show()

def main():
    random.seed(42)
    start_time = time.time()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.95, mutpb=0.7,
        ngen=20,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    pool.close()
    pool.join()

    best_individual = hof[0]
    coords_flat = best_individual[:2*IND_SIZE]
    best_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    best_central_index = int(best_individual[-1])
    if best_central_index < 0 or best_central_index >= IND_SIZE:
        best_central_index = 0

    grupos = agrupar_torres_por_cosseno(best_coords, best_central_index, n_grupos=8)
    vectors = best_coords - best_coords[best_central_index]
    grupos_bal = balancear_grupos(grupos, vectors)

    valid_paths = []
    for grupo in grupos_bal:
        distancias = [np.linalg.norm(best_coords[t] - best_coords[best_central_index]) for t in grupo]
        ordenado = [node for _, node in sorted(zip(distancias, grupo), reverse=True)]
        valid_paths.append(ordenado + [best_central_index])

    turbines = []
    for i, (x, y) in enumerate(best_coords):
        if i == best_central_index:
            turbines.append(Turbine(0, x, y))
        else:
            turbines.append(Turbine(TURB_POWER, x, y))
    plant = Plant(CABLE_VOLTAGE, turbines, valid_paths)

    cable_loss_kw = plant.Pjtot / 1e3
    cable_cost_total = plant.Ctot

    print("\nMelhor solução encontrada:")
    print("Coordenadas X:", best_coords[:,0].tolist())
    print("Coordenadas Y:", best_coords[:,1].tolist())
    print("Índice do nodo central:", best_central_index)

    print("\nGrafo (caminhos por similaridade de cosseno) do melhor indivíduo:")
    for idx, path in enumerate(valid_paths):
        print(f" Caminho {idx+1}: {[int(n) for n in path]}")

    print("\nResultados dos cabos:")
    print(f" Joule losses on cabling (kW): {cable_loss_kw:.6f}")
    print(f" Custo total dos cabos: {cable_cost_total:.6f} (R$)")

    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}min {total_time%60:.2f}s")

    turbine_powers = [0 if i == best_central_index else TURB_POWER for i in range(IND_SIZE)]
    plotar_grafo_com_subtree_sum(
        best_coords.tolist(),
        valid_paths,
        turbine_powers,
        best_central_index
    )

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
