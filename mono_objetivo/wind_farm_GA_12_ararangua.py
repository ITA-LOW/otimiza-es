import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution_circle, plot_fitness, save_logbook_to_csv
import multiprocessing
import time
from wfwe import WindFarm
import matplotlib.pyplot as plt

# Definindo o tipo de problema (Maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Criando a toolbox
toolbox = base.Toolbox()

# Parâmetros
IND_SIZE = 12  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
N_DIAMETERS = 260  # 2 diâmetros de distância no mínimo

def create_individual_from_coordinates(coords):
    individual = creator.Individual(np.array(coords).flatten().tolist())
    return individual

# Carregando coordenadas iniciais
initial_coordinates, _, _ = getTurbLocYAML('config/iea37-ex12.yaml')
toolbox.register("individual", create_individual_from_coordinates, coords=initial_coordinates.tolist())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

# Pré-carrega os dados fora da função evaluate:
TURB_LOC_DATA = getTurbLocYAML("config/iea37-ex12.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML("config/iea37-335mw.yaml")
WIND_ROSE_DATA = getWindRoseYAML("config/ararangua-windrose.yaml")

def evaluate_otimizado(individual, turb_loc_data=TURB_LOC_DATA,
             turb_atrbt_data=TURB_ATRBT_DATA,
             wind_rose_data=WIND_ROSE_DATA):
    turb_coords_yaml, fname_turb, fname_wr = turb_loc_data
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    mask_inside = is_within_circle(turb_coords[:, 0], turb_coords[:, 1], CIRCLE_RADIUS)
    penalty_out_of_circle = np.sum(~mask_inside) * 1e6

    num_turb = len(turb_coords)
    if num_turb > 1:
        diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(num_turb, k=1)
        close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
        penalty_close_turbines = np.sum(close_mask) * 1e6

    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    fitness = np.sum(aep) - penalty_out_of_circle - penalty_close_turbines
    return fitness,

def calculate_pure_aep(individual):
    """Calculates the AEP of an individual without applying penalties."""
    turb_coords_yaml, fname_turb, fname_wr = TURB_LOC_DATA
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = TURB_ATRBT_DATA
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    return np.sum(aep)

def get_weighted_average_velocities(layout_coords):
    """
    Calculates the frequency-weighted average wind speed for each turbine in a layout.
    """
    n_turbines = len(layout_coords)
    avg_velocities = np.zeros(n_turbines)
    turbine_coords_list = list(map(tuple, layout_coords))

    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = TURB_ATRBT_DATA
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA

    for i in range(len(wind_dir)):
        direction = wind_dir[i]
        speed = wind_speed[i]
        frequency = wind_freq[i]
        
        if frequency == 0:
            continue

        farm = WindFarm(layout_coords, wind_direction=direction,
                        turbine_diameter=turb_diam, wind_speed_free_stream=speed)
        farm.calculate_wake_effects()

        for j in range(n_turbines):
            turbine_pos = turbine_coords_list[j]
            velocity_at_turbine = farm.turbine_velocities[turbine_pos]
            avg_velocities[j] += velocity_at_turbine * frequency
            
    return avg_velocities

def mutate(individual, mu, sigma, indpb):
    individual = np.array(individual)
    if random.random() < indpb:
        for i in range(len(individual)):
            individual[i] += random.gauss(mu, sigma)
        enforce_circle(individual)
    return creator.Individual(individual.tolist()), 

def local_search(individual, step_size, iterations):
    """
    Performs a simple hill-climbing local search on an individual.
    Nudges each coordinate and keeps the change if it improves fitness.
    """
    current_best = toolbox.clone(individual)
    if not current_best.fitness.valid:
        current_best.fitness.values = toolbox.evaluate(current_best)
    current_fitness = current_best.fitness.values[0]

    for _ in range(iterations):
        improved_in_pass = False
        for i in range(len(current_best)):
            # Tenta um nudge positivo
            neighbor = toolbox.clone(current_best)
            neighbor[i] += step_size
            enforce_circle(neighbor)
            neighbor.fitness.values = toolbox.evaluate(neighbor)
            neighbor_fitness = neighbor.fitness.values[0]

            if neighbor_fitness > current_fitness:
                current_best = neighbor
                current_fitness = neighbor_fitness
                improved_in_pass = True
                continue

            # Tenta um nudge negativo
            neighbor = toolbox.clone(current_best)
            neighbor[i] -= step_size
            enforce_circle(neighbor)
            neighbor.fitness.values = toolbox.evaluate(neighbor)
            neighbor_fitness = neighbor.fitness.values[0]

            if neighbor_fitness > current_fitness:
                current_best = neighbor
                current_fitness = neighbor_fitness
                improved_in_pass = True
                continue
        
        if not improved_in_pass:
            break
            
    return current_best

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=100, indpb=0.4) 
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate_otimizado)

def main():
    random.seed(42)
    start_time = time.time()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # --- Parameters for Adaptive GA and Early Stopping ---
    NGEN = 500
    PATIENCE = 50
    MIN_DELTA = 10.0
    CXPB = 0.95
    MUTPB = 0.7
    SIGMA_NORMAL = 100
    SIGMA_AGGRESSIVE = 250
    AGGRESSIVE_DURATION = 15

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    stagnation_counter = 0
    aggressive_phase_triggered = False
    aggressive_phase_countdown = 0
    last_max_fitness = 0.0

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)
    last_max_fitness = hof[0].fitness.values[0]

    for gen in range(1, NGEN + 1):
        
        current_max_fitness = hof[0].fitness.values[0]

        if (current_max_fitness - last_max_fitness) < MIN_DELTA:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        last_max_fitness = current_max_fitness

        if stagnation_counter >= PATIENCE:
            if not aggressive_phase_triggered:
                print(f"--- Stagnation detected at gen {gen}. Increasing sigma to {SIGMA_AGGRESSIVE} for {AGGRESSIVE_DURATION} generations. ---")
                toolbox.register("mutate", mutate, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)
                aggressive_phase_triggered = True
                aggressive_phase_countdown = AGGRESSIVE_DURATION
                stagnation_counter = 0
            else:
                print(f"--- Stagnation persists after aggressive mutation. Stopping early at generation {gen}. ---")
                break

        if aggressive_phase_countdown > 0:
            aggressive_phase_countdown -= 1
            if aggressive_phase_countdown == 0:
                print(f"--- End of aggressive mutation phase at generation {gen}. Reverting sigma to {SIGMA_NORMAL}. ---")
                toolbox.register("mutate", mutate, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)

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

    pool.close()
    pool.join()

    generation_data = logbook.select("gen")
    max_fitness_data = logbook.select("max")
    
    layouts_dir = os.path.join(output_dir, "best_layouts")
    os.makedirs(layouts_dir, exist_ok=True)
    
    print(f"\n--- Salvando os {len(hof)} melhores layouts em '{layouts_dir}/' ---")
    for i, individual in enumerate(hof):
        rank = i + 1
        aep_fitness = individual.fitness.values[0]
        coords = np.array(individual).reshape((IND_SIZE, 2))
        filename = os.path.join(layouts_dir, f"layout_rank_{rank}_coords.txt")
        with open(filename, 'w') as f:
            x_coords_list = coords[:, 0]
            y_coords_list = coords[:, 1]
            x_str = ", ".join([f"{val:.4f}" for val in x_coords_list])
            y_str = ", ".join([f"{val:.4f}" for val in y_coords_list])
            f.write(f"xc: [{x_str}]\n")
            f.write(f"yc: [{y_str}]\n")
        print(f"Layout Rank {rank} (AEP: {aep_fitness:,.2f}) salvo em '{filename}' no formato xc/yc.")

    print("\nPlotando a melhor solução (Rank 1)...")
    best_coords_for_plot = np.array(hof[0]).reshape((IND_SIZE, 2))
    x_coords = best_coords_for_plot[:, 0]
    y_coords = best_coords_for_plot[:, 1]
    
    #plot_solution_circle(x_coords, y_coords, radius=CIRCLE_RADIUS, output_dir=output_dir)
    plot_fitness(generation_data[3:], max_fitness_data[3:], output_dir=output_dir)
    #save_logbook_to_csv(logbook, "set_19.csv", output_dir=output_dir)

    # --- Section for generating comparative report ---
    print("\n--- Generating Comparative Report ---")

    optimized_coords = np.array(hof[0]).reshape((IND_SIZE, 2))
    initial_individual = toolbox.individual()
    initial_aep = calculate_pure_aep(initial_individual)
    optimized_aep = hof[0].fitness.values[0]

    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = TURB_ATRBT_DATA

    print("Generating AEP comparison plot...")
    labels = ['Initial Layout', 'Optimized Layout']
    aep_values_gwh = [initial_aep / 1e6, optimized_aep / 1e6]
    percentage_gain = ((optimized_aep - initial_aep) / initial_aep) * 100 if initial_aep != 0 else 0

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.bar(labels, aep_values_gwh, color=['#4682B4', '#50C878'])
    ax.set_ylabel('Annual Energy Production (GWh)', fontsize=14)
    
    title_text = f'AEP Comparison: Initial vs. Optimized\nGain: {percentage_gain:.2f}%'
    ax.set_title(title_text, fontsize=16, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_aep_comparison.png'), dpi=300)
    plt.close()
    print("Saved 'report_aep_comparison.png'")

    print("Generating combined layout plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(initial_coordinates[:, 0], initial_coordinates[:, 1], c='#4682B4', s=100, edgecolor='black', label='Initial Position')
    ax.scatter(optimized_coords[:, 0], optimized_coords[:, 1], c='#FF6347', s=100, edgecolor='black', marker='^', label='Optimized Position')

    circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='black', fill=False, linestyle='--', linewidth=1.5)
    ax.add_artist(circle)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate (m)', fontsize=14)
    ax.set_ylabel('Y Coordinate (m)', fontsize=14)
    ax.set_title('Turbine Layout Comparison', fontsize=16, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_combined_layout.png'), dpi=300)
    plt.close()
    print("Saved 'report_combined_layout.png'")

    print("Calculating weighted average velocities for both layouts (this may take a moment)...")
    initial_avg_velocities = get_weighted_average_velocities(initial_coordinates)
    optimized_avg_velocities = get_weighted_average_velocities(optimized_coords)
    print("...calculation complete.")

    print("Generating weighted velocity distribution plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.hist(initial_avg_velocities, bins=8, alpha=0.6, color='#0072B2', label='Initial Layout')
    ax.hist(optimized_avg_velocities, bins=8, alpha=0.6, color='#D55E00', label='Optimized Layout')
    
    ax.set_xlabel('Frequency-Weighted Average Wind Speed (m/s)', fontsize=14)
    ax.set_ylabel('Number of Turbines', fontsize=14)
    ax.set_title('Distribution of Turbine-Specific Average Wind Speeds', fontsize=16, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_velocity_distribution.png'), dpi=300)
    plt.close()
    print("Saved 'report_velocity_distribution.png'")

    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    most_frequent_idx = np.argmax(wind_freq)
    vis_wind_direction = wind_dir[most_frequent_idx]
    vis_wind_speed = wind_speed[most_frequent_idx]
    
    print(f"Generating wake field plot for initial layout (using most frequent direction: {vis_wind_direction}°)...")
    initial_farm = WindFarm(initial_coordinates, wind_direction=vis_wind_direction,
                            turbine_diameter=turb_diam, wind_speed_free_stream=vis_wind_speed)
    initial_farm.plot_layout_with_wake_field(
        title=f'Initial Layout: Wake Field (Direction: {vis_wind_direction}°)',
        save_path=os.path.join(output_dir, 'report_initial_layout_wake.png')
    )
    print("Saved 'report_initial_layout_wake.png'")

    print(f"Generating wake field plot for optimized layout (using most frequent direction: {vis_wind_direction}°)...")
    optimized_farm = WindFarm(optimized_coords, wind_direction=vis_wind_direction,
                              turbine_diameter=turb_diam, wind_speed_free_stream=vis_wind_speed)
    optimized_farm.plot_layout_with_wake_field(
        title=f'Optimized Layout: Wake Field (Direction: {vis_wind_direction}°)',
        save_path=os.path.join(output_dir, 'report_optimized_layout_wake.png')
    )
    print("Saved 'report_optimized_layout_wake.png'")

    end_time = time.time()
    total_min = int((end_time - start_time)//60)
    total_sec = int((end_time - start_time)%60)
    print(f"Tempo de computação: {total_min}:{total_sec}")

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
