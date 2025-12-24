import sys
import os
import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
import matplotlib.pyplot as plt

# Ensure the config directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from config.iea37_aepcalc import calcAEP, getWindRoseYAML, getTurbAtrbtYAML

# --- Your Custom Parameters ---
IND_SIZE = 20  # Number of turbines
RECTANGLE_WIDTH = 4000  # meters
RECTANGLE_HEIGHT = 6000  # meters
TURBINE_DIAMETER = 130 # meters
MIN_DISTANCE = 2 * TURBINE_DIAMETER # 260 meters

# --- Genetic Algorithm Parameters (Enhanced) ---
POP_SIZE = 300
NGEN = 500
CXPB = 0.9
MUTPB = 0.4
TOURNEY_SIZE = 5

# --- Adaptive Mutation Parameters ---
PATIENCE = 50 # Generations to wait for improvement before acting
MIN_DELTA = 1e-6 # Minimum change in fitness to be considered an improvement
SIGMA_NORMAL = 150
SIGMA_AGGRESSIVE = 300
AGGRESSIVE_DURATION = 20 # Generations for aggressive mutation to last

# --- Local Search Parameters ---
LOCAL_SEARCH_ITERATIONS = 5
LOCAL_SEARCH_STEP_SIZE = 10 # meters to nudge each coordinate

# --- DEAP Toolbox Setup ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def create_random_individual():
    """Creates a single individual with random coordinates within the rectangle."""
    coords = []
    for _ in range(IND_SIZE):
        x = random.uniform(0, RECTANGLE_WIDTH)
        y = random.uniform(0, RECTANGLE_HEIGHT)
        coords.extend([x, y])
    return creator.Individual(coords)

toolbox.register("individual", create_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Constraint and Evaluation Functions ---

def is_within_rectangle(x, y, width, height):
    return 0 <= x <= width and 0 <= y <= height

def enforce_rectangle(individual, width, height):
    for i in range(IND_SIZE):
        individual[2*i] = max(0, min(individual[2*i], width))
        individual[2*i+1] = max(0, min(individual[2*i+1], height))

# Pre-load data to speed up evaluation
try:
    TURB_ATRBT_DATA = getTurbAtrbtYAML("config/iea37-335mw.yaml")
    WIND_ROSE_DATA = getWindRoseYAML("config/iea37-windrose.yaml")
except FileNotFoundError as e:
    print(f"Error: Could not find configuration files. Make sure they are in the 'config' directory.")
    print(e)
    sys.exit(1)

def evaluate_aep(individual):
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    turb_ci, turb_co, rated_ws, rated_pwr, _ = TURB_ATRBT_DATA
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    
    penalty_outside = 0
    for i in range(IND_SIZE):
        if not is_within_rectangle(turb_coords[i, 0], turb_coords[i, 1], RECTANGLE_WIDTH, RECTANGLE_HEIGHT):
            penalty_outside += 1
    
    penalty_distance = 0
    if IND_SIZE > 1:
        dist_matrix = np.linalg.norm(turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        close_pairs = np.sum(dist_matrix < MIN_DISTANCE) / 2
        penalty_distance = close_pairs

    total_penalty = (penalty_outside + penalty_distance) * 1e7
    if total_penalty > 0:
        return -total_penalty,

    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  TURBINE_DIAMETER, turb_ci, turb_co, rated_ws, rated_pwr)
    return np.sum(aep),

def mutate_layout(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
    enforce_rectangle(individual, RECTANGLE_WIDTH, RECTANGLE_HEIGHT)
    return creator.Individual(individual.tolist()),

# --- Local Search Function ---
def local_search_hill_climb(individual):
    print("\n--- Starting Local Search (Hill Climbing) Phase ---")
    current_best = toolbox.clone(individual)
    if not current_best.fitness.valid:
        current_best.fitness.values = toolbox.evaluate(current_best)
    
    for i in range(LOCAL_SEARCH_ITERATIONS):
        print(f"Local Search Iteration {i + 1}/{LOCAL_SEARCH_ITERATIONS}...")
        improved_in_pass = False
        for j in range(len(current_best)):
            for step in [-LOCAL_SEARCH_STEP_SIZE, LOCAL_SEARCH_STEP_SIZE]:
                neighbor = toolbox.clone(current_best)
                neighbor[j] += step
                enforce_rectangle(neighbor, RECTANGLE_WIDTH, RECTANGLE_HEIGHT)
                
                # Check if the neighbor is fundamentally different
                if np.array_equal(neighbor, current_best):
                    continue

                neighbor.fitness.values = toolbox.evaluate(neighbor)

                if neighbor.fitness.values[0] > current_best.fitness.values[0]:
                    current_best = neighbor
                    improved_in_pass = True
                    # Found a better spot, no need to check other direction for this coordinate
                    break 
            if improved_in_pass:
                # Move to the next coordinate once an improvement is found
                continue

        if not improved_in_pass:
            print("Local search converged. No further improvement found.")
            break
    print("--- Local Search Finished ---")
    return current_best

# Register genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate_layout, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=TOURNEY_SIZE)
toolbox.register("evaluate", evaluate_aep)

# --- Main Execution ---
def main():
    random.seed(42)
    start_time = time.time()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Hybrid GA + Local Search Optimization', fontsize=16)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    print("--- Starting Genetic Algorithm Phase ---")
    stagnation_counter = 0
    aggressive_phase_countdown = 0
    last_max_fitness = hof[0].fitness.values[0] if len(hof) > 0 else -np.inf

    for gen in range(1, NGEN + 1):
        current_max_fitness = hof[0].fitness.values[0]
        if (current_max_fitness - last_max_fitness) < MIN_DELTA:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_max_fitness = current_max_fitness

        if stagnation_counter >= PATIENCE:
            print(f"\n--- Stagnation detected. Triggering aggressive mutation. ---")
            toolbox.register("mutate", mutate_layout, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)
            aggressive_phase_countdown = AGGRESSIVE_DURATION
            stagnation_counter = 0

        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                print(f"\n--- End of aggressive phase. Reverting mutation. ---")
                toolbox.register("mutate", mutate_layout, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)

        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTPB:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        ax1.clear(); gens = logbook.select("gen"); max_fitness = logbook.select("max")
        ax1.plot(gens, max_fitness, 'b-', label='Max Fitness (AEP)')
        ax1.set_xlabel('Generation'); ax1.set_ylabel('Max Fitness (AEP)'); ax1.set_title('Fitness Evolution')
        ax1.grid(True); ax1.legend(); ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        ax2.clear()
        best_ind_coords = np.array(hof[0]).reshape((IND_SIZE, 2))
        ax2.scatter(best_ind_coords[:, 0], best_ind_coords[:, 1], c='blue', s=50, edgecolor='black')
        rect = plt.Rectangle((0, 0), RECTANGLE_WIDTH, RECTANGLE_HEIGHT, color='red', fill=False, linestyle='--')
        ax2.add_artist(rect)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim(-0.05 * RECTANGLE_WIDTH, 1.05 * RECTANGLE_WIDTH)
        ax2.set_ylim(-0.05 * RECTANGLE_HEIGHT, 1.05 * RECTANGLE_HEIGHT)
        ax2.set_title(f'Best Layout in Generation {gen}'); ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.grid(True, linestyle='--')

        fig.canvas.draw(); plt.pause(0.1)

    print("--- Genetic Algorithm Phase Finished ---")
    pool.close(); pool.join()

    # --- Hybrid Step: Local Search Fine-Tuning ---
    ga_best_individual = hof[0]
    final_best_individual = local_search_hill_climb(ga_best_individual)

    # --- Final Results ---
    best_fitness = final_best_individual.fitness.values[0]
    best_coords = np.array(final_best_individual).reshape((IND_SIZE, 2))

    print(f"\nFinal solution found with AEP = {best_fitness / 1e9:.4f} GWh")
    
    layouts_dir = os.path.join(output_dir, "best_layouts")
    os.makedirs(layouts_dir, exist_ok=True)
    filename = os.path.join(layouts_dir, "best_layout_coords_hybrid.txt")
    np.savetxt(filename, best_coords, fmt="%.4f", header="X_COORDS, Y_COORDS")
    print(f"Saved best layout coordinates to '{filename}'")

    final_plot_path = os.path.join(output_dir, "final_optimized_layout_hybrid.png")
    fig.savefig(final_plot_path, dpi=300)
    print(f"Saved final plot to '{final_plot_path}'")

    end_time = time.time()
    total_min = int((end_time - start_time) // 60)
    total_sec = int((end_time - start_time) % 60)
    print(f"Total computation time: {total_min}m {total_sec}s")

    plt.ioff()
    print("Optimization complete. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()