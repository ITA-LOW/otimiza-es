import numpy as np
from scipy.optimize import minimize
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from plot import plot_solution, create_animation

# Definindo constantes
IND_SIZE = 16  # Número de turbinas
CIRCLE_RADIUS = 1300  # Raio do círculo
MIN_DISTANCE = 130  # Distância mínima entre turbinas

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2

def calculate_penalty(individual):
    penalty_out_of_circle = 0
    penalty_close_turbines = 0
    
    # Convertendo o indivíduo para coordenadas de turbinas
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Penalização para turbinas fora do círculo
    for x, y in turb_coords:
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            distance_from_center = np.sqrt(x**2 + y**2) - CIRCLE_RADIUS
            penalty_out_of_circle += 1e6 * distance_from_center**2  # Penalização quadrática
    
    # Penalização para turbinas muito próximas
    for i in range(len(turb_coords)):
        for j in range(i + 1, len(turb_coords)):
            dist = np.linalg.norm(turb_coords[i] - turb_coords[j])
            if dist < MIN_DISTANCE:
                penalty_close_turbines += 1e6 * (MIN_DISTANCE - dist)**2  # Penalização quadrática
    
    return penalty_out_of_circle + penalty_close_turbines

def objective_function(params):
    # Carregando os dados dos arquivos YAML
    turb_coords, fname_turb, fname_wr = getTurbLocYAML("iea37-ex16.yaml")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML("iea37-335mw.yaml")
    wind_dir, wind_freq, wind_speed = getWindRoseYAML("iea37-windrose.yaml")

    # Convertendo o vetor de parâmetros para coordenadas de turbinas
    turb_coords = np.array(params).reshape((IND_SIZE, 2))
    
    # Calculando penalidades
    penalty = calculate_penalty(params)
    
    # Calculando o AEP
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # Fitness é o AEP menos as penalizações
    fitness = np.sum(aep) - penalty
    
    return -fitness  # O scipy.optimize.minimize busca minimizar a função, então retornamos o negativo do fitness

def print_verbose(xk, *args):
    # Função de callback para imprimir o fitness
    current_fitness = objective_function(xk)
    print(f"Current fitness: {-current_fitness}")

def optimize_wind_farm():
    # Definindo os parâmetros iniciais e limites
    initial_params = np.random.uniform(-CIRCLE_RADIUS, CIRCLE_RADIUS, IND_SIZE * 2)
    bounds = [(-CIRCLE_RADIUS, CIRCLE_RADIUS)] * (IND_SIZE * 2)
    
    # Otimizando usando SciPy
    result = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        callback=print_verbose  # Adicionando a função de callback
        # options={
        #    'maxiter': 10000,  # Número máximo de iterações elevado
        #    'ftol': 1e-9,      # Tolerância muito baixa para mudanças na função objetivo
        #    'gtol': 1e-9       # Tolerância muito baixa para o gradiente
        #}
    )

    # Resultados da otimização
    optimized_params = result.x
    optimized_coords = np.array(optimized_params).reshape((IND_SIZE, 2))
    
    return optimized_coords

def main():
    optimized_coords = optimize_wind_farm()
    
    print("Melhor solução:")
    print("Coordenadas X:", optimized_coords[:, 0])
    print("Coordenadas Y:", optimized_coords[:, 1])
    
    # Salva a solução da melhor resposta
    plot_solution(optimized_coords.flatten(), "final", IND_SIZE)
    

if __name__ == "__main__":
    main()
