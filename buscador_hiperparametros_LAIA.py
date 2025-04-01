import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multiprocessing
import csv
import time
from shapely.geometry import Point, Polygon

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros do problema
IND_SIZE = 100  # Número de turbinas
N_DIAMETERS = 480  # 2 diâmetros de distância no mínimo, disco de rotor turbina de 15mwh é 240m

# Definir polígonos
POLYGONS = [
    Polygon([(0, 0), (14500, 0), (22740, 16000), (8240, 16000)]),
    #Polygon([(315, -10), (965, -20), (1200, 1000), (315, 1200)]),
]

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carregando coordenadas iniciais
initial_coordinates, _, _ = getTurbLocYAML('Testes_artigo_2/caso_100_turbinas/iea37-teste_LAIA_100_n_otimizado.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Função para verificar se um ponto está dentro de qualquer polígono
def is_within_polygons(x, y, polygons):
    point = Point(x, y)
    return any(polygon.contains(point) for polygon in polygons)


def is_within_polygon_vectorized(x_array, y_array, polygon):
    points = [Point(x, y) for x, y in zip(x_array, y_array)]
    return np.array([polygon.contains(point) for point in points], dtype=bool)
##########################################################################################################3

# Ajustar coordenadas para ficar dentro do polígono
def enforce_polygons(individual):
    for i in range(IND_SIZE):
        x, y = individual[2 * i], individual[2 * i + 1]
        point = Point(x, y)
        if not any(polygon.contains(point) for polygon in POLYGONS):
            # Encontrar o polígono mais próximo e projetar o ponto
            nearest_polygon = min(POLYGONS, key=lambda p: point.distance(p))
            projected_point = nearest_polygon.exterior.interpolate(nearest_polygon.exterior.project(point))
            individual[2 * i], individual[2 * i + 1] = projected_point.x, projected_point.y
###########################################################################################################

# Função de mutação modificada
def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_polygons(individual)
    return creator.Individual(individual.tolist()), 
#######################################################################################################

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("Testes_artigo_2/caso_100_turbinas/iea37-teste_LAIA_100_n_otimizado.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("iea37-15mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("iea37-windrose_LAIA.yaml")

def evaluate_otimizado(individual, turb_loc_data=TURB_LOC_DATA,
             turb_atrbt_data=TURB_ATRBT_DATA,
             wind_rose_data=WIND_ROSE_DATA):
    # Desempacota os dados previamente carregados
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data

    # Converte o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_polygon = 0
    penalty_close_turbines = 0

    # Penaliza turbinas fora dos polígonos
    # Se is_within_polygons aceita arrays, pode ser vetorizada
    mask_inside = is_within_polygon_vectorized(turb_coords[:, 0], turb_coords[:, 1], POLYGONS[0])
    penalty_out_of_polygon = np.sum(~mask_inside) * 1e6

    # Penaliza turbinas muito próximas: vetorize o cálculo das distâncias
    # Utiliza a técnica de matriz de distância (apenas a parte superior, sem repetição)
    num_turb = len(turb_coords)
    if num_turb > 1:
        # Calcula todas as distâncias de uma vez
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        # Pega a parte superior da matriz (não considera a diagonal)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6

    # Calcula o AEP com os dados já carregados
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Penaliza a solução se houver turbinas fora do polígono ou muito próximas
    fitness = np.sum(aep) - penalty_out_of_polygon - penalty_close_turbines
    
    return fitness,

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=150, indpb=0.55) 
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_otimizado)

# Parâmetros para testar
cxpb_values = [i / 100.0 for i in range(5, 101, 5)]     # 5% a 100%
indpb_values = [i / 100.0 for i in range(5, 101, 5)]    # 5% a 100%
mutpb_values = [i / 100.0 for i in range(5, 101, 5)]    # 5% a 100%

# Função principal do algoritmo genético
def main(indpb, mutpb, cxpb):

    random.seed(42)

    pop = 300
    torneio = 5
    alpha = 0.5
    gen = 1000
    sigma = 100

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register("mutate", mutate, mu=0, sigma=sigma, indpb=indpb) 
    toolbox.register("select", tools.selTournament, tournsize=torneio)
    toolbox.register("evaluate", evaluate_otimizado)

    pop = toolbox.population(n=pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gen, 
                                        stats=stats, halloffame=hof, verbose=False)
    
    pool.close()
    pool.join()

    best_individual = hof[0]
    #best_coords = np.array(best_individual).reshape((IND_SIZE, 2))
    aep = evaluate_otimizado(best_individual)[0]
    
    return aep

# Testando combinações de parâmetros
results = []
start_time = time.time()
for indpb in indpb_values:
    for mutpb in mutpb_values:
        for cxpb in cxpb_values:
            aep = main(indpb, mutpb, cxpb)
            results.append((indpb, mutpb, cxpb, aep))
            print(f"INDPB: {indpb:.2f}, MUTPB: {mutpb:.2f}, CXPB: {cxpb:.2f} AEP: {aep:.2f} MWh")
            # Salvando os resultados em um arquivo CSV
            with open('results.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['INDPB', 'MUTPB', 'CXPB', 'AEP'])
                writer.writerows(results)
end_time = time.time()
total_min = int((end_time - start_time)//60)
total_sec = int((end_time - start_time)%60)
print(f"\nTempo de computação: {total_min}:{total_sec}")

# Exibindo os melhores resultados
best_result = max(results, key=lambda x: x[3])
print("Melhores parâmetros sugeridos:")
print(f"indpb = {best_result[0]:.2f},")
print(f"mutpb = {best_result[1]:.2f}")
print(f"cxpb = {best_result[2]:.2f}")
print(f"AEP = {best_result[3]:.6f} MWh")
