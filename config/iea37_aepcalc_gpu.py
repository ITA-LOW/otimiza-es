"""IEA Task 37 Combined Case Study AEP Calculation Code - GPU Accelerated

Modified to use CuPy for GPU acceleration. It will fall back to NumPy if CuPy is not available.
"""

from __future__ import print_function
import sys
import yaml
from math import radians as DegToRad

try:
    import cupy as np
    print("✓ CuPy importado com sucesso. Usando GPU.")
except (ImportError, ModuleNotFoundError):
    print("⚠️  CuPy não encontrado. Usando NumPy em CPU.")
    import numpy as np


# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def WindFrame(turb_coords, wind_dir_deg):
    """Convert map coordinates to downwind/crosswind coordinates."""
    
    # Convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=W)
    wind_dir_deg = 270. - wind_dir_deg
    # Convert inflow wind direction from degrees to radians
    wind_dir_rad = np.radians(wind_dir_deg)
    
    # Constants to use below
    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)
    
    # Ensure turb_coords is a cupy/numpy array
    turb_coords = np.asarray(turb_coords)
    
    # Create an empty array with the same shape as turb_coords but with the dtype for coordinate
    frame_coords = np.empty(turb_coords.shape)
    
    # Convert to downwind(x) & crosswind(y) coordinates
    frame_coords[:, 0] = (turb_coords[:, 0] * cos_dir) - (turb_coords[:, 1] * sin_dir)
    frame_coords[:, 1] = (turb_coords[:, 0] * sin_dir) + (turb_coords[:, 1] * cos_dir)
    
    return frame_coords


def GaussianWake_vetorizado_optimizado(frame_coords, turb_diam):
    """
    Retorna a perda total de cada turbina devido ao wake das turbinas a montante,
    utilizando otimizações para maior precisão e desempenho.
    """
    # Garantir precisão dupla
    frame_coords = frame_coords.astype(np.float64)
    num_turb = len(frame_coords)

    # Constantes
    CT = 4.0 * (1. / 3.) * (1.0 - 1. / 3.)
    k = 0.0324555

    # Extrai as coordenadas e reformata para matriz coluna
    x_coords = frame_coords[:, 0].reshape(-1, 1)
    y_coords = frame_coords[:, 1].reshape(-1, 1)

    # Calcula as diferenças entre todas as turbinas
    x_diff = x_coords.T - x_coords   # Matriz de diferenças de X
    y_diff = y_coords.T - y_coords   # Matriz de diferenças de Y

    # Máscara para turbinas em que a turbina primária está a jusante (x_diff > 0)
    mask = x_diff > 0

    # Inicializa sigma com zeros e calcula somente para os casos válidos
    sigma = np.zeros_like(x_diff)
    sigma[mask] = k * x_diff[mask] + turb_diam / np.sqrt(8.)

    # Calcula o expoente somente onde a máscara é True
    exponent = np.zeros_like(sigma)
    # Evitar overflow/underflow com np.exp
    exponent_val = np.zeros_like(sigma)
    exponent_val[mask] = -0.5 * (y_diff[mask] / sigma[mask])**2
    
    # Calcula o radical (fator do modelo Bastankhah) somente para os valores válidos
    radical = np.ones_like(sigma)
    radical[mask] = 1. - CT / (8. * sigma[mask]**2 / turb_diam**2)

    # Para evitar valores negativos na raiz, aplicamos np.maximum e calculamos a raiz
    radical_val = np.ones_like(sigma)
    radical_val[mask] = np.sqrt(np.maximum(radical[mask], 0))

    # Calcula a matriz de perda utilizando a equação do modelo
    loss_matrix = np.zeros_like(sigma)
    loss_matrix[mask] = (1. - radical_val[mask]) * np.exp(exponent_val[mask])

    # Agrega as perdas para cada turbina usando a raiz da soma dos quadrados
    loss = np.sqrt(np.sum(loss_matrix**2, axis=1))

    return loss


def calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
            turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    """Calculate the wind farm AEP."""
    num_bins = len(wind_freq)

    pwr_produced = np.zeros(num_bins)
    for i in range(num_bins):
        pwr_produced[i] = DirPower(turb_coords, wind_dir[i], wind_speed[i],
                                   turb_diam, turb_ci, turb_co,
                                   rated_ws, rated_pwr)

    hrs_per_year = 365.*24.
    AEP = hrs_per_year * (wind_freq * pwr_produced)
    AEP /= 1.E6

    return AEP


def DirPower(turb_coords, wind_dir_deg, wind_speed,
             turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    #Return the power produced by each turbine.
    num_turb = len(turb_coords)

    frame_coords = WindFrame(turb_coords, wind_dir_deg)
    
    loss = GaussianWake_vetorizado_optimizado(frame_coords, turb_diam)
    
    wind_speed_eff = wind_speed*(1.-loss)
    turb_pwr = np.zeros(num_turb)

    # Vetorização da lógica de cálculo de potência
    # Condição 1: entre cut-in e rated
    mask1 = (turb_ci <= wind_speed_eff) & (wind_speed_eff < rated_ws)
    turb_pwr[mask1] = rated_pwr * ((wind_speed_eff[mask1] - turb_ci) / (rated_ws - turb_ci))**3
    
    # Condição 2: entre rated e cut-out
    mask2 = (rated_ws <= wind_speed_eff) & (wind_speed_eff < turb_co)
    turb_pwr[mask2] = rated_pwr

    pwrDir = np.sum(turb_pwr)

    return pwrDir

# Funções de leitura de YAML permanecem usando NumPy para processamento inicial de CPU
import numpy as numpy_cpu

def getTurbLocYAML(file_name):
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']
    
    turb_xc = numpy_cpu.asarray(defs['position']['items']['xc'])
    turb_yc = numpy_cpu.asarray(defs['position']['items']['yc'])
    
    turb_coords = numpy_cpu.column_stack((turb_xc, turb_yc))
    
    ref_list_turbs = defs['wind_plant']['properties']['layout']['items']
    ref_list_wr = (defs['plant_energy']['properties']
                       ['wind_resource_selection']['properties']['items'])
    
    fname_turb = next(ref['$ref']
                      for ref in ref_list_turbs if ref['$ref'][0] != '#')
    fname_wr = next(ref['$ref']
                    for ref in ref_list_wr if ref['$ref'][0] != '#')
    
    return turb_coords, fname_turb, fname_wr


def getWindRoseYAML(file_name):
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)['definitions']['wind_inflow']['properties']

    wind_dir = numpy_cpu.asarray(props['direction']['bins'])
    wind_freq = numpy_cpu.asarray(props['probability']['default'])
    wind_speed = numpy_cpu.asarray(props['speed']['default'])

    return wind_dir, wind_freq, wind_speed


def getTurbAtrbtYAML(file_name):
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']
        op_props = defs['operating_mode']['properties']
        turb_props = defs['wind_turbine_lookup']['properties']
        rotor_props = defs['rotor']['properties']

    turb_ci = float(op_props['cut_in_wind_speed']['default'])
    turb_co = float(op_props['cut_out_wind_speed']['default'])
    rated_ws = float(op_props['rated_wind_speed']['default'])
    rated_pwr = float(turb_props['power']['maximum'])
    turb_diam = float(rotor_props['radius']['default']) * 2.

    return turb_ci, turb_co, rated_ws, rated_pwr, turb_diam


if __name__ == "__main__":
    turb_coords, fname_turb, fname_wr = getTurbLocYAML(sys.argv[1])
    wind_dir, wind_freq, wind_speed = getWindRoseYAML(fname_wr)
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
        fname_turb)
        
    # Move data to GPU if using CuPy
    if 'cupy' in sys.modules:
        turb_coords = np.asarray(turb_coords)
        wind_freq = np.asarray(wind_freq)
        wind_speed = np.asarray(wind_speed)
        wind_dir = np.asarray(wind_dir)

    AEP = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    
    # If result is on GPU, move it to CPU for printing
    if 'cupy' in sys.modules:
        AEP = np.asnumpy(AEP)

    print(np.array2string(AEP, precision=5, floatmode='fixed',
                          separator=', ', max_line_width=62))
    print(np.around(np.sum(AEP), decimals=5))