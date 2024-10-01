import numpy as np
from scipy.optimize import minimize
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution

# Definindo constantes
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
MIN_DISTANCE = 130  # Distância mínima entre turbinas

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2

def calculate_penalty(individual):
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Penalização para turbinas fora do círculo
    for x, y in turb_coords:
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            penalty_out_of_circle += 1e3 * (np.sqrt(x**2 + y**2) - CIRCLE_RADIUS)  # Penalização linear
    
    # Penalização para turbinas muito próximas
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < MIN_DISTANCE:
                penalty_close_turbines += 1e3 * (MIN_DISTANCE - dist)  # Penalização linear
    
    return penalty_out_of_circle + penalty_close_turbines

def objective_function(params):
    turb_coords = np.array(params).reshape((IND_SIZE, 2))
    penalty = calculate_penalty(params)
    
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")
    
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    fitness = np.sum(aep) - penalty
    
    return -fitness  # Minimização

def print_verbose(xk, *args):
    current_fitness = objective_function(xk)
    print(f"Current fitness: {-current_fitness}")

def optimize_wind_farm():
    initial_params, _, _ = getTurbLocYAML("iea37-ex16.yaml")
    initial_params = initial_params.flatten()
    
    bounds = [(-CIRCLE_RADIUS, CIRCLE_RADIUS)] * (IND_SIZE * 2)
    
    # Usando Nelder-Mead como método alternativo
    result = minimize(
        objective_function,
        initial_params,
        method='Nelder-Mead',
        bounds=bounds,
        callback=print_verbose,
        options={
            'maxiter': 5000,  # Número máximo de iterações
            'disp': True
        }
    )

    optimized_params = result.x
    optimized_coords = np.array(optimized_params).reshape((IND_SIZE, 2))
    
    # Calcular o AEP otimizado
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")
    aep = calcAEP(optimized_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)

    return optimized_coords, np.sum(aep)

def main():
    optimized_coords, total_aep = optimize_wind_farm()
    
    print("Melhor solução:")
    print("Coordenadas X:", optimized_coords[:, 0].tolist())
    print("Coordenadas Y:", optimized_coords[:, 1].tolist())
    print(f"AEP Total: {total_aep:.2f} MWh")
    
    # Visualizando a solução
    plot_solution(optimized_coords)

if __name__ == "__main__":
    main()
