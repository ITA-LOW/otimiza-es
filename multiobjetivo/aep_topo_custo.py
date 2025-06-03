import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
from scipy.sparse.csgraph import minimum_spanning_tree
import math
from collections import defaultdict
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

# =============================================
# Classes para modelagem de cabos e usina
# =============================================
class Cable:
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.I = self.Pn / (math.sqrt(3) * self.Vn) if self.Pn > 0 else 0.0  
        self.dI = 2.3  
        self.A = self.I / self.dI if self.I > 0 else 0.0  
        self.p = 0.0173e-6  
        self.alpha = 0.00393  
        self.p90 = self.p * (1 + self.alpha * (90 - 20))  
        self.C = 2e3  
        self.Pj = self._calc_pj()
        self.Ctot = self._calc_ctot()

    def _calc_pj(self):
        if self.I == 0 or self.A == 0:
            return 0.0
        return 3 * (self.I ** 2) * self.p90 * self.lc / (self.A * 1e-6)

    def _calc_ctot(self):
        return self.lc * self.C

class Turbine:
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y

class Plant:
    def __init__(self, Vn, turbines, paths):
        self.Vn = Vn
        self.turbines = turbines
        self.paths = paths
        self.Cb = []
        self.cables_flat = []
        self.cable_map = {}
        self.Pjtot = 0
        self.Ctot = 0
        
        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        """
        Cria as conexões dos cabos. Cada 'path' é uma lista ordenada de nós.
        """
        self.Cb = []
        for path in self.paths:
            cable_path = []
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i+1]
                x1, y1 = self.turbines[current_node].x, self.turbines[current_node].y
                x2, y2 = self.turbines[next_node].x, self.turbines[next_node].y
                length = math.hypot(x2 - x1, y2 - y1)
                
                # Soma a potência de todas as turbinas a jusante deste cabo
                Ptransmitted = 0
                for j in range(i+1, len(path)):
                    node = path[j]
                    # Se a subestação (nodo central) não gera, esse if evita somá-la
                    if self.turbines[node].P > 0:
                        Ptransmitted += self.turbines[node].P
                
                cable = Cable(lc=length, Vn=self.Vn, Pn=Ptransmitted)
                cable_path.append(cable)
                
                # Salva o cabo no dicionário, associando-o ao par (current_node, next_node)
                self.cable_map[(current_node, next_node)] = cable
                # Se você quiser acesso reverso (next_node, current_node), pode salvar também:
                self.cable_map[(next_node, current_node)] = cable
            
            self.Cb.append(cable_path)
        self.cables_flat = [cable for path in self.Cb for cable in path]

    def calculate_losses(self):
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)

    def calculate_cost(self):
        self.Ctot = sum(cable.Ctot for cable in self.cables_flat)

# =============================================
# Funções auxiliares para otimização
# =============================================
def generate_clustered_paths_with_central(coords, central_index, n_clusters=7, debug=False):
    """
    Gera caminhos usando clusterização (KMeans) dos nós não centrais e conecta cada cluster
    ao nodo central escolhido.

    Parâmetros:
      coords: lista de coordenadas dos nós.
      central_index: índice do nodo central.
      n_clusters: número de clusters desejados.
      debug: se True, plota as etapas intermediárias para visualização.
      
    Retorna:
      clustered_paths: lista de caminhos gerados.
    """
    
    # Define a torre central de acordo com o gene do indivíduo
    substation = coords[central_index]
    # Seleciona as turbinas (todos os nós exceto o central)
    remaining_indices = [i for i in range(len(coords)) if i != central_index]
    turbines = np.array([coords[i] for i in remaining_indices])
    
    # Aplica KMeans para agrupar as turbinas
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(turbines)
    
    if debug:
        # Plota a clusterização
        plt.figure(figsize=(8, 6))
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for cluster in range(n_clusters):
            cluster_points = turbines[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[cluster % len(colors)], label=f'Cluster {cluster}')
        # Marca a subestação
        plt.scatter([substation[0]], [substation[1]], color='black', marker='*', s=200, label='Subestação')
        plt.title("Clusterização dos nós (exceto o central)")
        plt.legend()
        plt.show()
    
    clustered_paths = []
    
    for cluster in range(n_clusters):
        # Índices locais (na lista "turbines") dos nós do cluster
        indices_cluster = np.where(labels == cluster)[0]
        if len(indices_cluster) == 0:
            continue
        # Mapeia para índices originais
        cluster_nodes = [remaining_indices[i] for i in indices_cluster]
        cluster_coords = np.array([coords[i] for i in cluster_nodes])
        
        # Ponto de conexão: o nó do cluster mais próximo da subestação
        distances = np.linalg.norm(cluster_coords - np.array(substation), axis=1)
        min_idx = np.argmin(distances)
        connection_node = cluster_nodes[min_idx]
        
        if debug:
            # Mostra o ponto de conexão para este cluster
            plt.figure(figsize=(8, 6))
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], color='gray', label='Turbinas do cluster')
            plt.scatter([substation[0]], [substation[1]], color='black', marker='*', s=200, label='Subestação')
            plt.scatter([cluster_coords[min_idx][0]], [cluster_coords[min_idx][1]], 
                        color='red', marker='o', s=150, label='Ponto de conexão')
            plt.title(f"Cluster {cluster} - Ponto de conexão")
            plt.legend()
            plt.show()
        
        # Gera a MST local para o cluster
        n = len(cluster_nodes)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = math.hypot(cluster_coords[i][0] - cluster_coords[j][0],
                                  cluster_coords[i][1] - cluster_coords[j][1])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        local_mst = minimum_spanning_tree(dist_matrix)
        local_paths = []
        rows, cols = local_mst.nonzero()
        for i, j in zip(rows, cols):
            node_i = cluster_nodes[i]
            node_j = cluster_nodes[j]
            local_paths.append([node_i, node_j])
        
        if debug:
            # Visualiza a MST para o cluster
            plt.figure(figsize=(8, 6))
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], color='gray', label='Turbinas do cluster')
            # Desenha as arestas da MST
            for i, j in zip(rows, cols):
                x_vals = [cluster_coords[i][0], cluster_coords[j][0]]
                y_vals = [cluster_coords[i][1], cluster_coords[j][1]]
                plt.plot(x_vals, y_vals, 'k--', linewidth=1)
            plt.scatter([cluster_coords[min_idx][0]], [cluster_coords[min_idx][1]], 
                        color='red', marker='o', s=150, label='Ponto de conexão')
            plt.title(f"Cluster {cluster} - MST local")
            plt.legend()
            plt.show()
        
        # Conecta a subestação (nodo central) ao nodo de conexão do cluster
        connection_path = [central_index, connection_node]
        
        clustered_paths.append(connection_path)
        clustered_paths.extend(local_paths)
    
    return clustered_paths

# =============================================
# Configuração do Algoritmo Genético e definição do problema multiobjetivo
# =============================================
# IND_SIZE: 16 nós (1 nodo central + 15 turbinas)
IND_SIZE = 16
CIRCLE_RADIUS = 1300            
N_DIAMETERS = 260               
TURB_POWER = 3.35e6             
CABLE_VOLTAGE = 33e3            

# Criando um fitness multiobjetivo:
# Objetivo 1: Maximizar AEP líquido (com penalizações)
# Objetivo 2: Minimizar as perdas dos cabos (maximizando -cable_loss)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
# O indivíduo agora tem 2*IND_SIZE genes (para coordenadas) + 1 gene inteiro para o índice do nodo central
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def create_individual_from_coordinates(coords):
    # Cria o indivíduo a partir das coordenadas do YAML e adiciona o gene do nodo central (inicialmente 0)
    flat_coords = np.array(coords).flatten().tolist()
    return creator.Individual(flat_coords + [0])

# Carrega as coordenadas iniciais a partir do arquivo YAML.
initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    # Aplica a restrição apenas aos 2*IND_SIZE genes (parte contínua)
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            individual[2*i] = CIRCLE_RADIUS * math.cos(angle)
            individual[2*i + 1] = CIRCLE_RADIUS * math.sin(angle)

def mutate(individual, mu, sigma, indpb):
    # Mutação na parte contínua (coordenadas)
    coords = np.array(individual[:2*IND_SIZE])
    if random.random() < indpb:
        for i in range(len(coords)):
            coords[i] += random.gauss(mu, sigma)
    # Garante que as coordenadas estejam dentro do círculo
    for i in range(IND_SIZE):
        x, y = coords[2*i], coords[2*i+1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            coords[2*i] = CIRCLE_RADIUS * math.cos(angle)
            coords[2*i+1] = CIRCLE_RADIUS * math.sin(angle)
    # Mutação no gene discreto (índice do nodo central)
    central = individual[-1]
    if random.random() < indpb:
        central = random.randint(0, IND_SIZE-1)
    new_ind = list(coords) + [central]
    return creator.Individual(new_ind),

def mate(ind1, ind2, alpha=0.5):
    # Cruzamento na parte contínua (blend crossover)
    for i in range(2*IND_SIZE):
        gene1 = ind1[i]
        gene2 = ind2[i]
        gamma = random.uniform(-alpha, 1+alpha)
        ind1[i] = (1-gamma)*gene1 + gamma*gene2
        ind2[i] = gamma*gene1 + (1-gamma)*gene2
    # Para o gene discreto, troca com probabilidade 0.5
    if random.random() < 0.5:
        ind1[-1], ind2[-1] = ind2[-1], ind1[-1]
    return ind1, ind2

toolbox.register("mate", mate, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=5)

def evaluate_multiobjetivo(individual,
                           turb_atrbt_data=getTurbAtrbtYAML("iea37-335mw.yaml"),
                           wind_rose_data=getWindRoseYAML("iea37-windrose.yaml")):
    # Desempacota os dados do YAML
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    
    # Extrai as coordenadas (primeiros 2*IND_SIZE genes) e o índice do nodo central (último gene)
    coords_flat = individual[:2*IND_SIZE]
    turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    central_index = int(individual[-1])
    if central_index < 0 or central_index >= IND_SIZE:
        central_index = 0  # caso o gene esteja fora do intervalo
    
    # Penalizações: nós fora do círculo
    mask_inside = (turb_coords[:,0]**2 + turb_coords[:,1]**2) <= CIRCLE_RADIUS**2
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6
    # Penalização para nós muito próximos
    diff = turb_coords.reshape(IND_SIZE, 1, 2) - turb_coords.reshape(1, IND_SIZE, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(IND_SIZE, k=1)
    penalty_close = np.sum(dist_matrix[i, j] < N_DIAMETERS) * 1e6
    penalty = penalty_out_of_circle + penalty_close

    # Cálculo do AEP: para o nodo central (subestação) a produção é 0.
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    aep_liquido = np.sum(aep) - penalty

    try:
        # Usa a clusterização via KMeans para gerar os caminhos, considerando o nodo central evolutivo
        paths = generate_clustered_paths_with_central(turb_coords.tolist(), central_index)
        # (Opcional: você pode aplicar outras penalizações, por exemplo, para redundância nos caminhos)
        valid_paths = [path for path in paths if all(0 <= node < IND_SIZE for node in path)]
        
        # Define as turbinas: o nodo central (em central_index) tem produção 0; os demais têm TURB_POWER
        turbines = []
        for i, (x, y) in enumerate(turb_coords):
            if i == central_index:
                turbines.append(Turbine(0, x, y))
            else:
                turbines.append(Turbine(TURB_POWER, x, y))
        # Variável global para uso na classe Plant (para identificar o nodo central)
        global central_index_global
        central_index_global = central_index
        
        plant = Plant(CABLE_VOLTAGE, turbines, valid_paths)
        cable_loss = plant.Pjtot  # perdas Joule (W)
    except Exception as e:
        print("Erro no cálculo de cabos:", e)
        cable_loss = 1e8

    return aep_liquido, -cable_loss

toolbox.register("evaluate", evaluate_multiobjetivo)

def plotar_grafo_com_subtree_sum(coords, caminhos, turbine_powers, central):
    # Constrói um dicionário de vizinhos (grafo não direcionado)
    vizinhos = {i: set() for i in range(len(coords))}
    for caminho in caminhos:
        if len(caminho) < 2:
            continue
        # Mesmo que os caminhos venham como pares, adicionamos as conexões
        for i in range(len(caminho) - 1):
            a, b = caminho[i], caminho[i + 1]
            vizinhos[a].add(b)
            vizinhos[b].add(a)
    
    # Constrói a árvore direcionada a partir do nodo central usando DFS
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

    # Calcula recursivamente a potência acumulada (subtree sum) para cada nó
    subtree_power = {}
    def compute_subtree(no):
        total = turbine_powers[no]  # potência do próprio nó
        for filho in tree_filhos[no]:
            total += compute_subtree(filho)
        subtree_power[no] = total
        return total

    compute_subtree(central)

    # Plotagem
    plt.figure(figsize=(10, 10))
    # Plota os nós (torres)
    for i, (x, y) in enumerate(coords):
        plt.plot(x, y, 'bo')
        plt.text(x + 10, y + 10, str(i), fontsize=9)

    # Função para plotar recursivamente cada aresta e o rótulo com a potência acumulada
    def plot_edge(pai, filho):
        x_values = [coords[pai][0], coords[filho][0]]
        y_values = [coords[pai][1], coords[filho][1]]
        plt.plot(x_values, y_values, 'k-', linewidth=1)
        xm = (coords[pai][0] + coords[filho][0]) / 2
        ym = (coords[pai][1] + coords[filho][1]) / 2
        # O rótulo mostra a potência acumulada no nó filho (ou seja, a potência que passa pelo cabo)
        plt.text(xm, ym, f"{(subtree_power[filho]/1e6):.2f}", color='red', fontsize=8)
        # Chama recursivamente para os filhos do nó atual
        for filho2 in tree_filhos[filho]:
            plot_edge(filho, filho2)

    # Inicia a plotagem das arestas a partir do nodo central
    for filho in tree_filhos[central]:
        plot_edge(central, filho)

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
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.95, mutpb=0.7, 
                                         ngen=2000, stats=stats, halloffame=hof, verbose=True)
    
    pool.close()
    pool.join()
    
    best_individual = hof[0]
    coords_flat = best_individual[:2*IND_SIZE]
    best_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    best_central_index = int(best_individual[-1])
    if best_central_index < 0 or best_central_index >= IND_SIZE:
        best_central_index = 0
    
    best_paths = generate_clustered_paths_with_central(best_coords.tolist(), best_central_index)
    valid_paths = [path for path in best_paths if all(0 <= node < IND_SIZE for node in path)]
    
    turbines = []
    for i, (x, y) in enumerate(best_coords):
        if i == best_central_index:
            turbines.append(Turbine(0, x, y))
        else:
            turbines.append(Turbine(TURB_POWER, x, y))
    global central_index_global
    central_index_global = best_central_index
    plant = Plant(CABLE_VOLTAGE, turbines, valid_paths)
    
    cable_loss_kw = plant.Pjtot / 1e3  
    cable_cost_total = plant.Ctot         
    
    print("\nMelhor solução encontrada:")
    print("Coordenadas X:", best_coords[:,0].tolist())
    print("Coordenadas Y:", best_coords[:,1].tolist())
    print("Índice do nodo central:", best_central_index)
    
    print("\nGrafo (caminhos clusterizados) do melhor indivíduo:")
    for idx, path in enumerate(valid_paths):
        path_int = [int(node) for node in path]
        print(f" Caminho {idx+1}: {path_int}")
    
    print("\nResultados dos cabos:")
    print(f" Joule losses on cabling (kW): {cable_loss_kw:.6f}")
    print(f" Custo total dos cabos: {cable_cost_total:.6f} (R$)")
    
    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}min {total_time%60:.2f}s")
    
    turbine_powers = [0 if i == best_central_index else TURB_POWER for i in range(IND_SIZE)]
    plotar_grafo_com_subtree_sum(
    best_coords.tolist(),  # lista de coordenadas
    valid_paths,           # caminhos gerados (lista de listas com pares de nós)
    turbine_powers,        # potência de cada turbina (0 para o central)
    best_central_index     # índice do nodo central
    )
    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
