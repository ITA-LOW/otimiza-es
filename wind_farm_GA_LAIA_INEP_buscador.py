import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution_polygons, plot_fitness, save_logbook_to_csv
import multiprocessing
import time
import csv
from shapely.geometry import Point, Polygon

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 100  # Número de turbinas
N_DIAMETERS = 2*240  # 2 diâmetros de distância no mínimo

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

# Função de avaliação
def evaluate(individual):
    # Carregando os dados dos arquivos YAML
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("Testes_artigo_2/caso_100_turbinas/iea37-teste_LAIA_100_n_otimizado.yaml")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-15mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose_LAIA.yaml")

    # Convertendo o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
   
    penalty_out_of_polygon = 0
    penalty_close_turbines = 0
    
    for x, y in turb_coords:
        if not is_within_polygons(x, y, POLYGONS):
            penalty_out_of_polygon += 1e6  

    # Penalizar turbinas muito próximas
    min_distance = N_DIAMETERS
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < min_distance:
                penalty_close_turbines += 1e6  
    
    # Calculando o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Penalizando a solução se tiver turbinas fora do círculo ou muito próximas
    fitness = np.sum(aep) - penalty_out_of_polygon - penalty_close_turbines
    
    return fitness,

# Função de mutação modificada
def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_polygons(individual)
    return creator.Individual(individual.tolist()), 

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=150, indpb=0.55) 
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

# Parâmetros para testar
cxpb_values = [i / 100.0 for i in range(50, 81, 5)]    # 0.50 a 0,80
indpb_values = [i / 100.0 for i in range(50, 101, 5)]    # 0.55 a 0.85
mutpb_values = [i / 100.0 for i in range(65, 101, 5)]    # 0.40 a 0.55

# Configuração da otimização
def main(indpb, mutpb, cxpb):
    random.seed(42)

    pop = 300
    torneio = 5
    alpha = 0.5
    gen = 300
    sigma = 100

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register("mutate", mutate, mu=0, sigma=sigma, indpb=indpb) 
    toolbox.register("select", tools.selTournament, tournsize=torneio)
    toolbox.register("evaluate", evaluate)

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
    best_coords = np.array(best_individual).reshape((IND_SIZE, 2))
    aep = evaluate(best_individual)[0]
    

    return aep

# Testando combinações de parâmetros
results = []
for indpb in indpb_values:
    for mutpb in mutpb_values:
        for cxpb in cxpb_values:
            aep = main(indpb, mutpb, cxpb)
            results.append((indpb, mutpb, cxpb, aep))
            print(f"INDPB: {indpb:.2f}, MUTPB: {mutpb:.2f}, CXPB: {cxpb:.2f}, AEP: {aep:.2f} MWh")
            # Salvando os resultados em um arquivo CSV
            with open('results.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['INDPB', 'MUTPB', 'CXPB' 'AEP'])
                writer.writerows(results)

# Exibindo os melhores resultados
best_result = max(results, key=lambda x: x[2])
print("Melhores parâmetros sugeridos:")
print(f"cxpb = {best_result[0]:.2f},")
print(f"indpb = {best_result[0]:.2f}")
print(f"mutpb = {best_result[1]:.2f}")
print(f"AEP = {best_result[2]:.6f} MWh")


