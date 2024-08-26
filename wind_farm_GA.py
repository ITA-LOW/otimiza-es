import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution, create_animation

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Funções para criar indivíduos e população
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo

def create_individual():
    return creator.Individual([random.uniform(-CIRCLE_RADIUS, CIRCLE_RADIUS) for _ in range(IND_SIZE * 2)])

toolbox.register("individual", create_individual)
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
            penalty_out_of_circle += 1e6  # Penalização alta se a turbina estiver fora do círculo

    # Penaliza se as turbinas estão muito próximas
    min_distance = 130  # Distância mínima entre turbinas
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < min_distance:
                penalty_close_turbines += 1e6  # Penalização alta se a turbina estiver muito próxima de outra
    
    # Calculando o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Penalizando a solução se tiver turbinas fora do círculo ou muito próximas
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,

toolbox.register("evaluate", evaluate)

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.2)  # Ajuste o sigma conforme necessário
toolbox.register("select", tools.selTournament, tournsize=3)

# Função de mutação modificada
def mutate(individual, mu, sigma, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            # Garantir que a turbina permaneça dentro do círculo
            enforce_circle(individual)
    return individual,

toolbox.register("mutate", mutate, mu=0, sigma=50, indpb=0.2)  # Ajuste o sigma conforme necessário

# Configuração da otimização
def main():
    random.seed(42)
    
    # Configura o ambiente DEAP
    num_turbines = 16  # Defina o número de turbinas
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1300, 1300)  # Ajuste o intervalo se necessário
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_turbines * 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Registre os operadores genéticos e a função de avaliação
    toolbox.register("evaluate", evaluate)  # Certifique-se de definir a função evaluate corretamente
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutate, mu=0, sigma=50, indpb=0.2)  # Ajuste o sigma conforme necessário
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(50):
        # Executa uma geração do algoritmo
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, 
                            stats=stats, halloffame=hof, verbose=True)

    # Imprime as coordenadas da melhor solução
    best_individual = hof[0]
    best_coords = np.array(best_individual).reshape((num_turbines, 2))
    x_coords = best_coords[:, 0]
    y_coords = best_coords[:, 1]
    
    print("Melhor solução:")
    print("Coordenadas X:", x_coords)
    print("Coordenadas Y:", y_coords)

    # Salva a solução da melhor resposta
    plot_solution(hof[0], "final", num_turbines)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
