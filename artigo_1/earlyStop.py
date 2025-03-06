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

initial_coordinates, _, _ = getTurbLocYAML('iea37-ex16.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Parâmetros do problema
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
N_DIAMETERS = 260  # Distância mínima entre turbinas

def is_within_circle(x, y, radius):
    x = np.asarray(x)
    y = np.asarray(y)
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

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("iea37-ex16.yaml")
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
    
    # Penalizações
    penalty_out_of_circle = 0
    penalty_close_turbines = 0

    # Penaliza turbinas fora do círculo
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6

    # Penaliza turbinas muito próximas
    num_turb = len(turb_coords)
    if num_turb > 1:
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6

    # Calcula o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    
    return fitness,


# Parâmetros para testar combinações
cxpb_values = [i / 100.0 for i in range(80, 101, 5)]    # 0.05 a 1.00 (ajuste conforme necessário)
indpb_values = [i / 100.0 for i in range(80, 101, 5)]
mutpb_values = [i / 100.0 for i in range(80, 101, 5)]

# Função principal do algoritmo genético
def main(indpb, mutpb, cxpb):
    random.seed(42)
    pop_size = 300
    torneio = 5
    alpha = 0.5
    maxgen = 300     # Número máximo de gerações
    sigma = 100
    patience = 50    # Gerações consecutivas sem melhora para early stopping
    tol = 1.0     # Limiar mínimo de melhora na fitness

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register("mutate", mutate, mu=0, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=torneio)
    toolbox.register("evaluate", evaluate_otimizado)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Avalia a população inicial
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    best_fitness = None
    no_improve = 0
    gen_early = 0

    # Executa o algoritmo geração a geração
    for gen in range(1, maxgen + 1):
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1,
                                           stats=stats, halloffame=hof, verbose=False)
        # Recupera a melhor fitness da geração atual
        current_best = max(pop, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        if best_fitness is None:
            best_fitness = current_best
        elif (current_best - best_fitness) < tol:
            no_improve += 1
        else:
            best_fitness = current_best
            no_improve = 0

        if gen >= 200 and no_improve >= patience:
            gen_early = gen
            break
    else:
        gen_early = maxgen

    pool.close()
    pool.join()

    best_ind = hof[0]
    aep = evaluate_otimizado(best_ind)[0]
    return aep, gen_early


# Testando combinações de parâmetros
results = []
start_time = time.time()
for indpb in indpb_values:
    for mutpb in mutpb_values:
        for cxpb in cxpb_values:
            aep = main(indpb, mutpb, cxpb)
            results.append((indpb, mutpb, cxpb, aep))
            # Salvando os resultados em um arquivo CSV
            with open('results.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['INDPB', 'MUTPB', 'CXPB', 'AEP', 'Final Gen'])  # Adicionando cabeçalho correto
                for result in results:
                    indpb, mutpb, cxpb, (aep, final_gen) = result  # Desempacota a tupla corretamente
                    writer.writerow([indpb, mutpb, cxpb, float(aep), final_gen])  # Converte np.float64 para float normal

end_time = time.time()
total_min = int((end_time - start_time) // 60)
total_sec = int((end_time - start_time) % 60)
print(f"\nTempo de computação: {total_min}:{total_sec:02d}")

best_result = max(results, key=lambda x: x[3][0])  # Ordena pelo primeiro valor da tupla (AEP)
aep, final_gen = best_result[3]  # Desempacotando corretamente

print("Melhores parâmetros sugeridos:")
print(f"indpb = {best_result[0]:.2f},")
print(f"mutpb = {best_result[1]:.2f}")
print(f"cxpb = {best_result[2]:.2f}")
print(f"AEP = {aep:.6f} MWh")  # Agora AEP está separado
print(f"Geração Final: {final_gen}")  # Exibe em qual geração ocorreu o early stop

