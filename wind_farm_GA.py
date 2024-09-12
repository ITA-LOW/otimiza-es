import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution, create_animation, plot_fitness
import multiprocessing

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carregando coordenadas iniciais
initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual):
    for i in range(IND_SIZE):
        x, y = individual[2*i], individual[2*i + 1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            # Ajusta a turbina para ficar dentro do círculo
            angle = np.arctan2(y, x)
            distance = min(np.sqrt(x**2 + y**2), CIRCLE_RADIUS)
            individual[2*i] = distance * np.cos(angle)
            individual[2*i + 1] = distance * np.sin(angle)

# Função de avaliação
def evaluate(individual):
    # Carregando os dados dos arquivos YAML
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("iea37-ex16.yaml")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")

    # Convertendo o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Verifica se todas as turbinas estão dentro do círculo de raio 1300m
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
toolbox.register("mutate", mutate, mu=0, sigma=50, indpb=0.2) 
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("evaluate", evaluate)

# Configuração da otimização
def main():
    random.seed(42)

    # Criação do pool de processos
    pool = multiprocessing.Pool()
    
    # Configura o ambiente DEAP
    toolbox.register("map", pool.map)  
    pop = toolbox.population(n=500)  # Tamanho da população
    hof = tools.HallOfFame(1)  # Manter o melhor indivíduo
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    generation_data = []
    max_fitness_data = []

    # Loop principal de otimização
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, ngen=3500, 
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
    plot_solution(x_coords, y_coords)
    plot_fitness(generation_data[3:], max_fitness_data[3:])

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()


# cxpb=0.7,     mutpb=0.3,  pop=100,    torneio=2,  alpha=0.5,  gen=1000,    indpb=0.2,     simga=50,   -> ~411815MWh
# cxpb=0.8,     mutpb=0.4,  pop=100,    torneio=2,  alpha=0.5,  gen=1000,    indpb=0.2,     simga=50,   -> ~411936MWh
# cxpb=0.85,    mutpb=0.45, pop=100,    torneio=2,  alpha=0.5,  gen=1000,    indpb=0.2,     simga=50,   -> ~410619MWh
# cxpb=0.8,     mutpb=0.4,  pop=250,    torneio=2,  alpha=0.5,  gen=1500,    indpb=0.2,     simga=50,   -> ~411012MWh
# cxpb=0.8,     mutpb=0.4,  pop=250,    torneio=4,  alpha=0.5,  gen=1500,    indpb=0.2,     simga=50,   -> ~412294MWh - winner
# cxpb=0.8,     mutpb=0.4,  pop=250,    torneio=5,  alpha=0.6,  gen=2500,    indpb=0.2,     simga=50,   -> ~410555MWh
# cxpb=0.5,     mutpb=0.45, pop=250,    torneio=6,  alpha=0.6,  gen=1500,    indpb=0.35,    simga=50,   -> ~400872MWh
# cxpb=0.8,     mutpb=0.4,  pop=250,    torneio=3,  alpha=0.5,  gen=2000,    indpb=0.15,    sigma=10,   -> ~407065MWh
# cxpb=0.8,     mutpb=0.4,  pop=250,    torneio=4,  alpha=0.75, gen=1500,    indpb=0.3,     sigma=150,  -> ~409235MWh
# cxpb=0.7,     mutpb=0.3,  pop=250,    torneio=4,  alpha=0.4,  gen=1500,    indpb=0.4,     sigma=150,  -> ~404785MWh
# cxpb=0.8,     mutpb=0.4,  pop=500,    torneio=4,  alpha=0.5,  gen=3500,    indpb=0.2,     simga=50,   -> ~410847MWh

#NÃO DESLIGUE O COMPUTADOR, VOLTAREI PRA DESLIGÁ-LO -> ITALO

