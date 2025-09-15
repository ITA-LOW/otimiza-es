import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
#from plot import plot_solution, plot_fitness, save_logbook_to_csv
import multiprocessing
import time

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 36  # Número de turbinas
CIRCLE_RADIUS = 2000  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carregando coordenadas iniciais
initial_coordinates, _, _ = getTurbLocYAML('artigo_1/iea37-ex36.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    x = np.asarray(x)
    y = np.asarray(y)
    return x**2 + y**2 <= radius**2


def is_within_circle_otimizado(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle_otimizado(x, y, CIRCLE_RADIUS):
            # Ajusta a turbina para ficar dentro do círculo
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

# Função de avaliação
def evaluate(individual):
    # Carregando os dados dos arquivos YAML
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("artigo_1/iea37-ex36.yaml")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")

    # Convertendo o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
   
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    for x, y in turb_coords:
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            penalty_out_of_circle += 1e6  # Penalização ajustada se a turbina estiver fora do círculo

    # Penaliza se as turbinas estão muito próximas
    min_distance = N_DIAMETERS  # Distância mínima entre turbinas
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < min_distance:
                penalty_close_turbines += 1e6  # Penalização ajustada se a turbina estiver muito próxima de outra
    
    # Calculando o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Penalizando a solução se tiver turbinas fora do círculo ou muito próximas
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("artigo_1/iea37-ex36.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("iea37-windrose.yaml")

def evaluate_otimizado(individual, turb_loc_data=TURB_LOC_DATA,
             turb_atrbt_data=TURB_ATRBT_DATA,
             wind_rose_data=WIND_ROSE_DATA):
    # Desempacota os dados previamente carregados
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data

    # Converte o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_circle = 0
    penalty_close_turbines = 0

   
    # Penaliza turbinas fora do círculo
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6

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
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,

# Função de mutação modificada
def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        # Garantir que a turbina permaneça dentro do círculo após mutação
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.10) 
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_otimizado)

# Configuração da otimização
def main():
    random.seed(42)

    start_time = time.time()

    # Criação do pool de processos
    pool = multiprocessing.Pool()
    
    # Configura o ambiente DEAP
    toolbox.register("map", pool.map)  
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
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=1.00, mutpb=0.35, ngen=1500, 
                                        stats=stats, halloffame=hof, verbose=True)
    
    # Fechando o pool para liberar os recursos
    pool.close()
    pool.join()

    # Salvando a aptidão máxima por geração, todas as informaçoes do verbose estao aqui
    for record in logbook:
        generation_data.append(record['gen'])
        max_fitness_data.append(record['max'])

    best_individual = hof[0]
    best_coords = np.array(best_individual).reshape((IND_SIZE, 2))
    
    x_coords = best_coords[:, 0].tolist()
    y_coords = best_coords[:, 1].tolist()

    print("Melhor solução:")
    print("Coordenadas X:", x_coords)
    print("Coordenadas Y:", y_coords)

    # Plotar a solução e a evolução da aptidão
    #plot_solution(x_coords, y_coords, radius=CIRCLE_RADIUS)
    #plot_fitness(generation_data[3:], max_fitness_data[3:]) # começo a partir do 3 pois os valores de fit iniciais são tão baixos que estragam o grafico
    #save_logbook_to_csv(logbook, "set_19") essa linha é util para plotar multiplos fitness no mesmo grafico

    end_time = time.time()
    total_min = int((end_time - start_time)//60)
    total_sec = int((end_time - start_time)%60)
    print(f"Tempo de computação: {total_min}:{total_sec}")


    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
