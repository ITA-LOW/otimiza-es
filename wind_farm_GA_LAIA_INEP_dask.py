"""
1 - rode dask scheduler -> vai aparecer um ip, substitua no ip abaixo
2 - nas duas máquinas rode dask worker tcp://192.168.1.X:8786 --nworkers 24 --nthreads 1
3 - execute o código 
"""


import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution_polygons, plot_fitness, save_logbook_to_csv
import multiprocessing
import time
from shapely.geometry import Point, Polygon
from dask.distributed import Client
import dask

client = Client("tcp://192.168.1.8:8786")
client.upload_file('iea37_aepcalc.py')
print(client)

# Função de mapeamento usando Dask
def dask_map(func, iterable):
    futures = [client.submit(func, item) for item in iterable]
    return client.gather(futures)

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 60  # Número de turbinas
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
initial_coordinates, _, _ = getTurbLocYAML('Testes_artigo_2/caso_60_turbinas/iea37-teste_LAIA_60_n_otimizado.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Função para verificar se um ponto está dentro de qualquer polígono
def is_within_polygons(x, y, polygons):
    point = Point(x, y)
    return any(polygon.contains(point) for polygon in polygons)


def is_within_polygon_vectorized(x_array, y_array, polygon):
    points = [Point(x, y) for x, y in zip(x_array, y_array)]
    return np.array([polygon.contains(point) for point in points], dtype=bool)


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
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("Testes_artigo_2/caso_60_turbinas/iea37-teste_LAIA_60_n_otimizado.yaml")
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

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("Testes_artigo_2/caso_60_turbinas/iea37-teste_LAIA_60_n_otimizado.yaml")
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
toolbox.register("evaluate", evaluate_otimizado)


# Configuração da otimização
def main():
    random.seed(42)

    
    start_time = time.time()
    
    # Criação do pool de processos
    #pool = multiprocessing.Pool()
    
    # Configura o ambiente DEAP
    toolbox.register("map", lambda func, *args: client.gather(client.map(func, *args)))  
    pop = toolbox.population(n=300)  # Tamanho da população
    hof = tools.HallOfFame(1)  # Manter o melhor indivíduo
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    generation_data = []
    max_fitness_data = []

    # Loop principal de otimização
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.95, mutpb=0.55, ngen=50, 
                                        stats=stats, halloffame=hof, verbose=True)
    
    # Fechando o pool para liberar os recursos
    #pool.close()
    #pool.join()

    # Salvando a aptidão máxima por geração, todas as informaçoes do verbose estao aqui
    #for record in logbook:
    #    generation_data.append(record['gen'])
    #    max_fitness_data.append(record['max'])

    #best_individual = hof[0]
    #best_coords = np.array(best_individual).reshape((IND_SIZE, 2))
    
    #x_coords = best_coords[:, 0].tolist()
    #y_coords = best_coords[:, 1].tolist()

    #print("Melhor solução:")
    #print("xc:", x_coords)
    #print("yc:", y_coords)

    # Plotar a solução e a evolução da aptidão
    #plot_solution_circle(x_coords, y_coords, radius=CIRCLE_RADIUS)
    #plot_solution_polygons(x_coords, y_coords, POLYGONS)
    #plot_fitness(generation_data[3:], max_fitness_data[3:]) # começo a partir do 3 pois os valores de fit iniciais são tão baixos que estragam o grafico
    #save_logbook_to_csv(logbook, "set_19") essa linha é util para plotar multiplos fitness no mesmo grafico

    end_time = time.time()
    total_min = int((end_time - start_time)//60)
    total_sec = int((end_time - start_time)%60)
    print(f"Tempo de computação: {total_min}:{total_sec}")


    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()

