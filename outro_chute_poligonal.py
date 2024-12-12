""" import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString

# Function to distribute turbines along the boundary of a polygon
def distribute_boundary_turbines(polygon, num_turbines, s=0):
    boundary = LineString(polygon.exterior.coords)
    perimeter = boundary.length
    distances = np.linspace(s, s + perimeter, num_turbines, endpoint=False) % perimeter
    points = [boundary.interpolate(distance) for distance in distances]
    return np.array([[p.x, p.y] for p in points])

# Function to generate a rotated grid of points within a polygon
def generate_inner_grid(polygon, dx, dy, b, theta, num_turbines):
    # Determine bounds of the polygon
    xmin, ymin, xmax, ymax = polygon.bounds
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Generate grid points
    x_coords = np.arange(xmin, xmax, dx)
    y_coords = np.arange(ymin, ymax, dy)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Apply offset and rotation
    grid[:, 0] += (np.arange(len(grid)) % 2) * b
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_grid = np.dot(grid - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]

    # Filter points inside the polygon
    inside_points = np.array([p for p in rotated_grid if polygon.contains(Point(p))])

    # Limit the number of turbines to the user-defined value
    return inside_points[:num_turbines]

# Define example polygons
POLYGONS = [
    Polygon([(-1500, -1500), (-20, -315), (-10, 315), (-965, 315)]),
    Polygon([(315, -10), (965, -20), (1200, 1000), (315, 1200)])
]

# User-defined parameters
num_boundary_turbines_user_input = [1, 1]  # Number of boundary turbines for each polygon
num_grid_turbines_user_input = [50, 50]  # Number of inner grid turbines for each polygon
dx, dy = 200, 200  # Grid spacing
b = 100  # Offset distance
theta = np.radians(90)  # Rotation angle
turbine_diameter = 100  # Turbine diameter
s = 0  # Starting position for boundary turbines

# Initialize results lists
boundary_points_list = []
grid_points_inside_list = []

for polygon, num_boundary_turbines, num_grid_turbines in zip(POLYGONS, num_boundary_turbines_user_input, num_grid_turbines_user_input):
    # Distribute boundary turbines
    boundary_points = distribute_boundary_turbines(polygon, num_boundary_turbines, s)
    boundary_points_list.append(boundary_points)

    # Generate inner grid turbines
    grid_points_inside = generate_inner_grid(polygon, dx, dy, b, theta, num_grid_turbines)
    grid_points_inside_list.append(grid_points_inside)

# Plot the result
plt.figure(figsize=(10, 10))
for polygon, boundary_points, grid_points_inside in zip(POLYGONS, boundary_points_list, grid_points_inside_list):
    x, y = polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Boundary')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='red', label='Boundary Turbines')
    plt.scatter(grid_points_inside[:, 0], grid_points_inside[:, 1], color='blue', label='Grid Turbines')

plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show() """


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# Function to generate a rotated grid of points within a polygon
def generate_inner_grid(polygon, dx, dy, b, theta, num_turbines):
    # Determine bounds of the polygon
    xmin, ymin, xmax, ymax = polygon.bounds
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Generate grid points
    x_coords = np.arange(xmin, xmax, dx)
    y_coords = np.arange(ymin, ymax, dy)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Apply offset and rotation
    grid[:, 0] += (np.arange(len(grid)) % 2) * b
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]);
    rotated_grid = np.dot(grid - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]

    # Filter points inside the polygon
    inside_points = np.array([p for p in rotated_grid if polygon.contains(Point(p))])

    # Limit the number of turbines to the user-defined value
    return inside_points[:num_turbines]

# Define example polygons
POLYGONS = [
    Polygon([(-2000, -2000), (-20, -1500), (1500, -500), (1250, -20), (1200, 1000), (-1500, 700)]),
    #Polygon([(315, -10), (965, -20), (1200, 1000), (315, 1200)])(-10, 315)
]

# User-defined parameters
total_turbines = 100  # Total number of turbines
dx, dy = 200, 130  # Grid spacing
b = 10  # Offset distance
theta = np.radians(75)  # Rotation angle

# Calculate areas of polygons and determine the number of turbines for each
areas = [polygon.area for polygon in POLYGONS]
total_area = sum(areas)
turbines_per_polygon = [int(total_turbines * (area / total_area)) for area in areas]

# Adjust for rounding differences to ensure the total number of turbines matches
turbines_per_polygon[-1] += total_turbines - sum(turbines_per_polygon)

# Generate turbines within each polygon
grid_points_inside_list = []
for polygon, num_turbines in zip(POLYGONS, turbines_per_polygon):
    grid_points_inside = generate_inner_grid(polygon, dx, dy, b, theta, num_turbines)
    grid_points_inside_list.append(grid_points_inside)

# Plot the result
plt.figure(figsize=(10, 10))
for polygon, grid_points_inside in zip(POLYGONS, grid_points_inside_list):
    x, y = polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Boundary')
    plt.scatter(grid_points_inside[:, 0], grid_points_inside[:, 1], color='blue', label='Grid Turbines')

plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

xc = []
yc = []

for grid in grid_points_inside_list:
    xc.extend([float(val) for val in grid[:, 0]])  # Converte cada elemento para float
    yc.extend([float(val) for val in grid[:, 1]])  # Converte cada elemento para float

# Agora os valores serão exibidos no formato correto
print("xc:", xc)
print("yc:", yc)
