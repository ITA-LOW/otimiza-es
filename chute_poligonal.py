import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# Função para distribuir pontos uniformemente ao longo da borda de um polígono
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

# Função para gerar uma grade de pontos dentro do polígono
def generate_grid(xmin, xmax, ymin, ymax, dx, dy, theta):
    x_coords = np.arange(xmin, xmax, dx)
    y_coords = np.arange(ymin, ymax, dy)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Aplicar rotação à grade
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_grid = np.dot(grid, rotation_matrix.T)

    return rotated_grid

# Função para filtrar pontos que estão dentro do polígono
def filter_points_inside_polygon(points, polygon):
    return np.array([p for p in points if polygon.contains(Point(p))])

# Função para garantir distância mínima entre pontos
def enforce_min_distance(points, min_distance):
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in filtered_points):
            filtered_points.append(point)
    return np.array(filtered_points)

# Definir os polígonos usando Shapely
POLYGONS = [
    Polygon([(-1500, -1500), (-20, -315), (-10, 315), (-965, 315)]),
    Polygon([(315, -10), (965, -20), (1200, 1000), (315, 1200)])
]

# Parâmetros
total_turbines = 8  # Quantidade total de turbinas
boundary_percentage = 0.45  # Percentual de turbinas no perímetro
num_boundary_points = int(total_turbines * boundary_percentage)
num_internal_points = total_turbines - num_boundary_points

turbine_radius = 50  # Raio da turbina
dx = dy = 2 * turbine_radius  # Espaçamento da grade
theta = np.radians(15)  # Rotação de 15 graus

plt.figure(figsize=(12, 6))

xc, yc = [], []  # Listas para coordenadas finais

for idx, polygon in enumerate(POLYGONS):
    # Distribuir pontos na borda
    boundary_points = distribute_boundary_points(polygon, num_boundary_points)
    boundary_points = enforce_min_distance(boundary_points, 2 * turbine_radius)

    # Gerar grade interna
    xmin, ymin, xmax, ymax = polygon.bounds
    grid_points = generate_grid(xmin, xmax, ymin, ymax, dx, dy, theta)

    # Filtrar apenas os pontos da grade que estão dentro do polígono
    grid_points_inside = filter_points_inside_polygon(grid_points, polygon)

    # Garantir distância mínima para os pontos internos
    grid_points_inside = enforce_min_distance(grid_points_inside, 2 * turbine_radius)

    # Selecionar pontos internos aleatórios
    if len(grid_points_inside) > num_internal_points:
        np.random.shuffle(grid_points_inside)
        grid_points_inside = grid_points_inside[:num_internal_points]

    # Combinar pontos no perímetro e internos
    all_turbine_points = np.vstack([boundary_points, grid_points_inside])

    # Adicionar coordenadas ao resultado final
    xc.extend(all_turbine_points[:, 0])
    yc.extend(all_turbine_points[:, 1])

    # Plotar o resultado
    plt.subplot(1, len(POLYGONS), idx + 1)
    x, y = polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Boundary')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='red', label='Boundary Points')
    plt.scatter(grid_points_inside[:, 0], grid_points_inside[:, 1], color='blue', label='Internal Points')

    #plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Polygon {idx + 1}")

plt.show()

# Imprimir as coordenadas finais sem np.float64
print("xc =", [float(coord) for coord in xc])
print("yc =", [float(coord) for coord in yc])
