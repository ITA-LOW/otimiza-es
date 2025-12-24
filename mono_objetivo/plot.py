import matplotlib.animation as animation
from matplotlib.patches import Polygon as mplPolygon
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

estilo='default'

def plot_solution_circle(x, y, radius, output_dir='.'):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'bo', markersize=6)
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim(-radius-100, radius+100)
    ax.set_ylim(-radius-100, radius+100)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate (m)', fontsize=28)
    ax.set_ylabel('Y Coordinate (m)', fontsize=28)
    ax.set_title('Optimized turbines', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'wind_farm_solution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_logbook_to_csv(logbook, filename, output_dir='.'):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'MaxFitness'])
        for entry in logbook:
            writer.writerow([entry['gen'], entry['max']])


def plot_fitness(x, y, output_dir='.'):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label='Max Fitness', color='b',linewidth=1)
    ax.set_title('Max Fitness x Generations', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'max_fitness_vs_generations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple_tests(file_list):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(8, 6))
    for filename in file_list:
        generations = []
        max_fitness = []
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                generations.append(int(row['Generation']))
                max_fitness.append(float(row['MaxFitness']))
        ax.plot(generations, max_fitness, label=f'Test: {filename}', linewidth=1)
    ax.set_title('Max Fitness x Generations for Multiple Tests', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.savefig('max_fitness_multiple_tests.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution_polygons(x, y, polygons):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, 'bo', markersize=6, label='Turbines')
    for poly in polygons:
        mpl_poly = mplPolygon(list(poly.exterior.coords), edgecolor='r', facecolor='none', 
                               linestyle='--', linewidth=2, label='Boundary')
        ax.add_patch(mpl_poly)
    all_coords = [coord for poly in polygons for coord in poly.exterior.coords]
    x_coords, y_coords = zip(*all_coords)
    ax.set_xlim(min(x_coords) - 1000, max(x_coords) + 1000)
    ax.set_ylim(min(y_coords) - 1000, max(y_coords) + 1000)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('wind_farm_solution_polygons.png', dpi=300, bbox_inches='tight')
    plt.close()