import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


"""
Esse código cria um indivíduo que serve de base pra população. Ele a princípio está programado para otimizar o espaço mas não a geração
de energia. Altere os parâmetros para gerar o indivíduo de cada caso (60, 80, 100). 

Parâmetros
total_turbines -> máximo de turbinas que serão posicionadas
boundary_percentage -> percentual das turbinas que serão posicionadas na borda da região de otimização
dx, dy -> espaçamento da grade interna
b -> deslocamento (offset) entre 2 linhas consecutivas
theta -> rotação da grade, ou seja, de todo o parque eólico

Indivíduo com 60 turbinas
total_turbines = 60 
boundary_percentage = 0.35  
dx, dy = 2200, 2300  
b = 100  
theta = np.radians(0) 

Indivíduo com 80 turbinas
total_turbines = 80  
boundary_percentage = 0.35  
dx, dy = 2250, 1900  
b = 100  
theta = np.radians(15)

Indivíduo com 100 turbinas
total_turbines = 100  
boundary_percentage = 0.35  
dx, dy = 2250, 1500 
b = 100  
theta = np.radians(72)

""" # Parâmetros 
total_turbines = 100  # Total de turbinas
boundary_percentage = 0.35  # Percentual de turbinas no perímetro
dx, dy = 2250, 1500  # Espaçamento da grade
b = 100  # Deslocamento entre 2 linhas consecutivas
theta = np.radians(72)  # Ângulo de rotação
num_boundary_points = int(total_turbines * boundary_percentage)
num_inner_turbines = total_turbines - num_boundary_points
turbine_diameter = 240
""" """

""" # Parâmetros 
total_turbines = 150  # Total de turbinas
boundary_percentage = 0.45  # Percentual de turbinas no perímetro
dx, dy = 1700, 1500  # Espaçamento da grade
b = 150  # Deslocamento entre 2 linhas consecutivas
theta = np.radians(75)  # Ângulo de rotação
num_boundary_points = int(total_turbines * boundary_percentage)
num_inner_turbines = total_turbines - num_boundary_points
turbine_diameter = 240 """

# Função para distribuir pontos na borda de um polígono
def distribute_boundary_points(polygon, num_points):
    coords = np.array(polygon.exterior.coords)
    perimeter = np.cumsum([0] + [
        np.linalg.norm(coords[i] - coords[i - 1]) for i in range(1, len(coords))
    ])
    total_perimeter = perimeter[-1]

    step = total_perimeter / num_points
    points = []

    for i in range(num_points):
        target_distance = i * step
        for j in range(1, len(perimeter)):
            if target_distance <= perimeter[j]:
                ratio = (target_distance - perimeter[j - 1]) / (perimeter[j] - perimeter[j - 1])
                point = coords[j - 1] + ratio * (coords[j] - coords[j - 1])
                points.append(point)
                break

    return np.array(points)

# Função para garantir distância mínima entre pontos
def enforce_min_distance(points, min_distance):
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in filtered_points):
            filtered_points.append(point)
    return np.array(filtered_points)

# Função para gerar uma grade rotacionada dentro do polígono
def generate_inner_grid(polygon, dx, dy, b, theta, num_turbines):
    # Determinar limites do polígono
    xmin, ymin, xmax, ymax = polygon.bounds
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Gerar pontos da grade
    x_coords = np.arange(xmin, xmax, dx)
    y_coords = np.arange(ymin, ymax, dy)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Aplicar deslocamento e rotação
    grid[:, 0] += (np.arange(len(grid)) % 2) * b
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_grid = np.dot(grid - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]

    # Filtrar pontos dentro do polígono
    inside_points = np.array([p for p in rotated_grid if polygon.contains(Point(p))])

    # Limitar o número de pontos ao valor definido pelo usuário
    return inside_points[:num_turbines]

# Definir exemplo de polígono
POLYGONS = [
    Polygon([(0, 0), (14500, 0), (22740, 16000), (8240, 16000)])  
]

# Gerar turbinas dentro de cada polígono
boundary_points_list = []
inner_points_list = []

for polygon in POLYGONS:
    # Distribuir pontos na borda
    boundary_points = distribute_boundary_points(polygon, num_boundary_points)
    boundary_points = enforce_min_distance(boundary_points, 2 * turbine_diameter)
    boundary_points_list.append(boundary_points)

    # Distribuir turbinas dentro do polígono
    inner_points = generate_inner_grid(polygon, dx, dy, b, theta, num_inner_turbines)
    inner_points = enforce_min_distance(inner_points, 2 * turbine_diameter)
    inner_points_list.append(inner_points)

# Consolidar os pontos
xc = []
yc = []

for boundary_points, inner_points in zip(boundary_points_list, inner_points_list):
    xc.extend(boundary_points[:, 0])
    yc.extend(boundary_points[:, 1])
    xc.extend(inner_points[:, 0])
    yc.extend(inner_points[:, 1])

# Plotar os resultados
plt.figure(figsize=(10, 10))

for polygon, boundary_points, inner_points in zip(POLYGONS, boundary_points_list, inner_points_list):
    x, y = polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Borda do Polígono')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='red', label='Turbinas na Borda')
    plt.scatter(inner_points[:, 0], inner_points[:, 1], color='blue', label='Turbinas Internas')

plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("poligono", dpi=300)
plt.legend()
plt.show()

# Exibir coordenadas no formato correto
print("xc =", [float(coord) for coord in xc])
print("yc =", [float(coord) for coord in yc])

min_distance = 2 * turbine_diameter  # Distância mínima permitida

def check_turbine_distances(xc, yc, min_distance):
    num_turbines = len(xc)
    for i in range(num_turbines):
        for j in range(i + 1, num_turbines):
            # Calcula a distância euclidiana entre turbinas i e j
            dist = np.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2)
            if dist < min_distance:
                #print(f"Turbinas {i} e {j} estão a {dist:.2f}, menor que {min_distance:.2f}")
                return False
    return True

# Executa a função e imprime o resultado
result = check_turbine_distances(xc, yc, min_distance)
print("As turbinas estão corretamente espaçadas?", result)

print(len(xc))
print(len(yc))