import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import multiprocessing
import csv
import time

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

initial_coordinates, _, _ = getTurbLocYAML('iea37-ex64.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Parâmetros do problema
IND_SIZE = 64  # Número de turbinas
CIRCLE_RADIUS = 3000  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo


def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            distance = CIRCLE_RADIUS
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

# Função de avaliação
def evaluate(individual):
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("iea37-ex64.yaml")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")

    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    for x, y in turb_coords:
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            penalty_out_of_circle += 1e6

    min_distance = N_DIAMETERS
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < min_distance:
                penalty_close_turbines += 1e6

    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,

# Parâmetros para testar
cxpb_values = [i / 100.0 for i in range(50, 81, 5)]    # 0.50 a 0,80
indpb_values = [i / 100.0 for i in range(50, 101, 5)]    # 0.55 a 0.85
mutpb_values = [i / 100.0 for i in range(65, 101, 5)]    # 0.40 a 0.55

# Função principal do algoritmo genético
def main(indpb, mutpb, cxpb):
    random.seed(42)

    pop = 300
    torneio = 5
    alpha = 0.5
    gen = 300
    sigma = 100
    #cxpb = 0.8

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
        aep = main(indpb, mutpb)
        results.append((indpb, mutpb, aep))
        print(f"INDPB: {indpb:.2f}, MUTPB: {mutpb:.2f} AEP: {aep:.2f} MWh")
        # Salvando os resultados em um arquivo CSV
        with open('results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['INDPB', 'MUTPB', 'AEP'])
            writer.writerows(results)

# Exibindo os melhores resultados
best_result = max(results, key=lambda x: x[2])
print("Melhores parâmetros sugeridos:")
#print(f"cxpb = {best_result[0]:.2f},")
print(f"indpb = {best_result[0]:.2f}")
print(f"mutpb = {best_result[1]:.2f}")
print(f"AEP = {best_result[2]:.6f} MWh")


#1513311G
#1506388G
#1480850G

#1479753GF 4° lugar

#1476689G
#1455075GF
#1445967G
#1422268GF
#1364943GF
#1336164GG
#1332883GF