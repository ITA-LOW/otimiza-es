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

# =============================================
# Classes para modelagem de cabos e usina
# =============================================
class Cable:
    """
    Representa um cabo que conecta dois nós (subestação ou turbina)
    Inputs:
        lc: comprimento do cabo (m)
        Vn: tensão nominal (V)
        Pn: potência ativa transmitida (W)
        Qi: potência reativa na entrada do cabo (VAr)
    """
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        # Calcula corrente: se Pn é zero, corrente é zero.
        self.I = self.Pn / (math.sqrt(3) * self.Vn) if self.Pn > 0 else 0.0  
        self.dI = 2.3  # Densidade de corrente (A/mm²)
        self.A = self.I / self.dI if self.I > 0 else 0.0  # Seção transversal (mm²)
        
        # Propriedades do material
        self.p = 0.0173e-6  # Resistividade do cobre a 20°C (Ω·m)
        self.alpha = 0.393  # Coeficiente de temperatura
        self.p90 = self.p * (1 + self.alpha * (90 - 20))  # Resistividade a 90°C
        
        # Parâmetro de custo
        self.C = 2e3  # Custo em R$/m
        
        # Propriedades calculadas
        self.Pj = self._calc_pj()
        self.Ctot = self._calc_ctot()

    def _calc_pj(self):
        """Calcula as perdas Joule.
           Se não há corrente, retorna 0.
           Caso contrário:
           P_loss = 3 * I² * (p90 * lc) / (A * 1e-6)
        """
        if self.I == 0 or self.A == 0:
            return 0.0
        return 3 * (self.I ** 2) * self.p90 * self.lc / (self.A * 1e-6)

    def _calc_ctot(self):
        """Calcula o custo total do cabo"""
        return self.lc * self.C


class Turbine:
    """
    Representa uma turbina ou subestação da usina.
    Inputs:
        Pt: potência nominal (W) - para a subestação, use 0.
        x: coordenada x (m)
        y: coordenada y (m)
    """
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y


class Plant:
    """
    Representa a usina offshore com turbinas e cabos.
    MODIFICADO: A subestação é definida como a primeira torre (nó 0) e os demais nós são as turbinas.
    """
    def __init__(self, Vn, turbines, paths):
        self.Vn = Vn
        self.turbines = turbines  # Lista de nós: índice 0 é a subestação; 1..n-1 são turbinas
        self.paths = paths        # Lista de caminhos (cada caminho é uma lista de nós)
        self.Cb = []
        self.cables_flat = []
        self.Pjtot = 0
        self.Ctot = 0
        
        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        """
        Cria as conexões dos cabos com base nos caminhos gerados pela MST.
        MODIFICADO: Cada caminho é uma lista ordenada de nós, onde o nó 0 é a subestação.
        Para cada segmento, a potência transmitida é a soma das potências de todas as turbinas a jusante.
        """
        self.Cb = []
        for path in self.paths:
            cable_path = []
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i+1]
                
                # Usa as coordenadas dos nós diretamente (subestação é o nó 0)
                x1, y1 = self.turbines[current_node].x, self.turbines[current_node].y
                x2, y2 = self.turbines[next_node].x, self.turbines[next_node].y
                length = math.hypot(x2 - x1, y2 - y1)
                
                # A potência transmitida é a soma das potências de todas as turbinas a partir do próximo nó.
                # Como a subestação (nó 0) não gera energia, ela não é considerada.
                Ptransmitted = 0
                for j in range(i+1, len(path)):
                    node = path[j]
                    if node != 0:
                        Ptransmitted += self.turbines[node].P
                cable = Cable(lc=length, Vn=self.Vn, Pn=Ptransmitted)
                cable_path.append(cable)
            self.Cb.append(cable_path)
        
        # "Achata" a lista de cabos
        self.cables_flat = [cable for path in self.Cb for cable in path]

    def calculate_losses(self):
        """Calcula as perdas totais dos cabos"""
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)

    def calculate_cost(self):
        """Calcula o custo total dos cabos"""
        self.Ctot = sum(cable.Ctot for cable in self.cables_flat)


# =============================================
# Funções auxiliares para otimização
# =============================================
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def generate_mst_paths(coords):
    """
    Gera caminhos de cabos usando MST.
    MODIFICADO: Considera que as coordenadas fornecidas já incluem a subestação como o primeiro nó.
    """
    try:
        n = len(coords)  # n = número total de nós (subestação + turbinas)
        # Cria matriz de distâncias (n x n)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        # Gera a MST a partir da matriz
        mst = minimum_spanning_tree(dist_matrix)
        # Converte a MST para um grafo (dicionário)
        graph = defaultdict(list)
        rows, cols = mst.nonzero()
        for i, j in zip(rows, cols):
            graph[i].append(j)
            graph[j].append(i)
        # Extrai caminhos do nó 0 (subestação) para cada nó folha usando BFS
        return get_mst_paths_from_graph(graph, root=0)
    except Exception as e:
        print(f"Erro na geração da MST: {e}")
        return [[0]]  # Retorna um caminho seguro

def get_mst_paths_from_graph(graph, root=0):
    """
    Extrai caminhos da MST a partir dos vizinhos diretos do nó 'root'.
    Para cada vizinho, executa uma busca para encontrar todos os caminhos
    da subárvore correspondente, retornando-os como grafos independentes.
    """
    paths = []
    # Para cada vizinho direto da subestação, extraia os caminhos até as folhas
    for neighbor in graph[root]:
        # Realiza uma DFS para encontrar todos os caminhos a partir de 'neighbor'
        stack = [(neighbor, [root, neighbor])]
        while stack:
            current, path = stack.pop()
            # Se o nó atual é uma folha (exceto a subestação)
            if current != root and len(graph[current]) == 1:
                paths.append(path)
            else:
                # Para cada vizinho do nó atual que ainda não foi visitado no caminho
                for next_node in graph[current]:
                    if next_node not in path:
                        stack.append((next_node, path + [next_node]))
    return paths

# =============================================
# Configuração do Algoritmo Genético
# =============================================
# MODIFICADO: IND_SIZE agora representa o número total de nós (1 subestação + 15 turbinas)
IND_SIZE = 16                   # Número total de nós: 1 subestação + 15 turbinas
CIRCLE_RADIUS = 1300            # Raio do círculo (m)
N_DIAMETERS = 260               # Distância mínima entre nós (m)
TURB_POWER = 3.35e6             # Potência nominal por turbina (W)
CABLE_VOLTAGE = 33e3            # Tensão nominal dos cabos (V)

# Para o problema multiobjetivo:
# Objetivo 1: Maximizar AEP líquido (AEP - penalizações)
# Objetivo 2: Minimizar perdas no cabo -> maximizando (- cable_loss)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carrega as coordenadas iniciais a partir do arquivo YAML.
# Assume-se que o arquivo YAML fornece as coordenadas para 16 torres,
# onde a primeira delas será utilizada como subestação.
initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    # Garante que todas as coordenadas estejam dentro do círculo.
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            individual[2*i] = CIRCLE_RADIUS * math.cos(angle)
            individual[2*i + 1] = CIRCLE_RADIUS * math.sin(angle)

from sklearn.cluster import KMeans

def generate_clustered_paths(coords, n_clusters=3):
    """
    Gera caminhos para os cabos com base em clusterização dos nós (excluindo a subestação).
    coords: Lista de coordenadas (nó 0 é a subestação).
    n_clusters: Número de clusters desejados para agrupar as turbinas.
    """
    # Separa a subestação e as turbinas
    substation = coords[0]
    turbines = np.array(coords[1:])  # turbinas sem a subestação
    
    # Aplica o KMeans para agrupar as turbinas
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(turbines)
    
    # Lista para armazenar os caminhos gerados
    clustered_paths = []
    
    # Para cada cluster, gera a MST local e adiciona a conexão com a subestação
    for cluster in range(n_clusters):
        # Indices das turbinas que pertencem ao cluster (lembre que índice 0 agora é a turbina 1)
        cluster_indices = np.where(labels == cluster)[0] + 1  # ajuste para índices originais
        
        # Ponto de conexão do cluster com a subestação (pode ser a turbina mais próxima da subestação)
        cluster_coords = turbines[labels == cluster]
        distances = np.linalg.norm(cluster_coords - np.array(substation), axis=1)
        min_idx = np.argmin(distances)
        connection_node = cluster_indices[min_idx]
        
        # Gera a MST local para o cluster (usando as coordenadas do cluster)
        # Cria uma matriz de distância para as turbinas do cluster
        n = len(cluster_indices)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = math.hypot(cluster_coords[i][0] - cluster_coords[j][0],
                                  cluster_coords[i][1] - cluster_coords[j][1])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        # Calcula a MST local (podendo utilizar o minimum_spanning_tree do SciPy)
        local_mst = minimum_spanning_tree(dist_matrix)
        # Converte a MST em caminhos: para simplificar, consideramos os segmentos como arestas
        local_paths = []
        rows, cols = local_mst.nonzero()
        for i, j in zip(rows, cols):
            # Cada aresta do cluster é um caminho entre dois nós
            node_i = cluster_indices[i]
            node_j = cluster_indices[j]
            local_paths.append([node_i, node_j])
        
        # Conecta a subestação ao ponto de conexão do cluster
        connection_path = [0, connection_node]
        
        # Combina o caminho de conexão com as arestas locais
        # Aqui, mantemos separadamente: um caminho para conectar a subestação e os segmentos internos do cluster
        clustered_paths.append(connection_path)
        clustered_paths.extend(local_paths)
    
    return clustered_paths


# Dados carregados para avaliação
TURB_LOC_DATA = getTurbLocYAML("iea37-ex16.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("iea37-windrose.yaml")

def evaluate_multiobjetivo(individual, turb_loc_data=TURB_LOC_DATA,
                           turb_atrbt_data=TURB_ATRBT_DATA,
                           wind_rose_data=WIND_ROSE_DATA):
    # Desempacota dados
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    
    # Converte o indivíduo em coordenadas (IND_SIZE x 2)
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Penalidades: fora do círculo e proximidade entre nós
    mask_inside = (turb_coords[:,0]**2 + turb_coords[:,1]**2) <= CIRCLE_RADIUS**2
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6
    diff = turb_coords.reshape(IND_SIZE, 1, 2) - turb_coords.reshape(1, IND_SIZE, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(IND_SIZE, k=1)
    penalty_close = np.sum(dist_matrix[i, j] < N_DIAMETERS) * 1e6
    
    # Cálculo do AEP (Annual Energy Production)
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Objetivo 1: AEP líquido (maximizar)
    aep_liquido = np.sum(aep) - penalty_out_of_circle - penalty_close
    
    try:
        
        paths = generate_clustered_paths(turb_coords.tolist()) # algoritmos de topologia
        valid_paths = [path for path in paths if all(0 <= node < IND_SIZE for node in path)]
        
        # NOVA SEÇÃO: Penalidade por redundância nos caminhos
        # Conta quantas vezes cada torre (exceto a subestação) aparece
        node_usage = defaultdict(int)
        for path in valid_paths:
            for node in path[1:]:  # ignora o nó 0 (subestação)
                node_usage[node] += 1
        penalty_redundancy = sum((count - 1) * 1e6 for count in node_usage.values() if count > 1)
        # Incorpora a penalidade ao AEP líquido
        aep_liquido -= penalty_redundancy
        
        turbines = [Turbine(0, x, y) if i == 0 else Turbine(TURB_POWER, x, y) 
                    for i, (x, y) in enumerate(turb_coords)]
        plant = Plant(CABLE_VOLTAGE, turbines, valid_paths)
        cable_loss = plant.Pjtot  # perdas por efeito Joule (W)
    except Exception as e:
        import traceback
        print("Erro no cálculo de cabos:", e)
        traceback.print_exc()
        cable_loss = 1e8  # Penalização elevada em caso de erro
    
    # Objetivo 2: Minimizar perdas de cabo (retorna -cable_loss para maximizar)
    return aep_liquido, -cable_loss

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_multiobjetivo)

import matplotlib.pyplot as plt

def plotar_grafo_com_coordenadas(coords, caminhos):
    plt.figure(figsize=(10, 10))
    
    # Plota as torres
    for i, (x, y) in enumerate(coords):
        plt.plot(x, y, 'bo')  # 'bo' = blue circle
        plt.text(x + 10, y + 10, str(i), fontsize=9)

    # Plota os caminhos
    for caminho in caminhos:
        for i in range(len(caminho) - 1):
            a, b = caminho[i], caminho[i + 1]
            x_values = [coords[a][0], coords[b][0]]
            y_values = [coords[a][1], coords[b][1]]
            plt.plot(x_values, y_values, 'k-', linewidth=1)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Grafo com coordenadas reais das torres")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("grafo.png")

def main():
    random.seed(42)
    start_time = time.time()

    # Configuração do paralelismo
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # População inicial
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Execução do Algoritmo Genético (200 gerações)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.95, mutpb=0.7, 
                                         ngen=500, stats=stats, halloffame=hof, verbose=True)
    
    pool.close()
    pool.join()
    
    # Melhor indivíduo e suas coordenadas
    best_individual = hof[0]
    best_coords = np.array(best_individual).reshape((IND_SIZE, 2))
    
    # Recalcula os caminhos e monta a planta para o melhor indivíduo
    best_paths = generate_mst_paths(best_coords.tolist())
    valid_paths = [path for path in best_paths if all(0 <= node < IND_SIZE for node in path)]
    turbines = [Turbine(0, x, y) if i == 0 else Turbine(TURB_POWER, x, y)
                for i, (x, y) in enumerate(best_coords)]
    plant = Plant(CABLE_VOLTAGE, turbines, valid_paths)
    
    # Cálculo dos parâmetros desejados para análise
    cable_loss_kw = plant.Pjtot / 1e3   # perdas em kW (se Pjtot estiver em watts)
    cable_cost_total = plant.Ctot         # custo total (R$)
    
    print("\nMelhor solução encontrada:")
    print("Coordenadas X:", best_coords[:,0].tolist())
    print("Coordenadas Y:", best_coords[:,1].tolist())
    
    # MODIFICADO: Converter os nós do MST para inteiros puros na exibição
    print("\nGrafo (caminhos MST) do melhor indivíduo:")
    for idx, path in enumerate(valid_paths):
        path_int = [int(node) for node in path]
        print(f" Caminho {idx+1}: {path_int}")
    
    print("\nResultados dos cabos:")
    print(f" Joule losses on cabling (kW): {cable_loss_kw:.6f}")
    print(f" Custo total dos cabos: {cable_cost_total:.6f} (R$)")
    
    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}min {total_time%60:.2f}s")

    # Plotar o grafo com as coordenadas reais
    plotar_grafo_com_coordenadas(best_coords.tolist(), valid_paths)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()