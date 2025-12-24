"""
IEA Task 37 AEP Calculation Code - BATCHED GPU ACCELERATED

Versão modificada para aceitar um lote (batch) de layouts de fazendas eólicas
e processá-los de uma só vez na GPU para máxima eficiência.
"""

from __future__ import print_function
import sys
import yaml
from math import radians as DegToRad

try:
    import cupy as np
    IS_GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    import numpy as np
    IS_GPU_AVAILABLE = False


def WindFrame_batch(turb_coords_batch, wind_dir_deg):
    """Converte um lote de coordenadas para o referencial do vento."""
    # turb_coords_batch tem shape (pop_size, num_turb, 2)
    
    wind_dir_rad = np.radians(270. - wind_dir_deg)
    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)
    
    # Rotação aplicada a todo o lote de uma vez
    frame_coords_batch = np.empty_like(turb_coords_batch)
    frame_coords_batch[..., 0] = (turb_coords_batch[..., 0] * cos_dir) - (turb_coords_batch[..., 1] * sin_dir)
    frame_coords_batch[..., 1] = (turb_coords_batch[..., 0] * sin_dir) + (turb_coords_batch[..., 1] * cos_dir)
    
    return frame_coords_batch


def GaussianWake_batch(frame_coords_batch, turb_diam):
    """Calcula o efeito de esteira para um lote de fazendas eólicas."""
    # frame_coords_batch tem shape (pop_size, num_turb, 2)
    pop_size, num_turb, _ = frame_coords_batch.shape

    CT = 4.0 * (1. / 3.) * (1.0 - 1. / 3.)
    k = 0.0324555

    # Coordenadas X e Y para todo o lote
    x_coords = frame_coords_batch[..., 0] # Shape: (pop_size, num_turb)
    y_coords = frame_coords_batch[..., 1] # Shape: (pop_size, num_turb)

    # Reformata para permitir broadcasting e cálculo de diferenças par a par
    x_coords_reshaped = x_coords.reshape(pop_size, num_turb, 1)
    y_coords_reshaped = y_coords.reshape(pop_size, num_turb, 1)

    # Diferenças para todo o lote. Shape: (pop_size, num_turb, num_turb)
    x_diff = x_coords_reshaped - x_coords.reshape(pop_size, 1, num_turb)
    y_diff = y_coords_reshaped - y_coords.reshape(pop_size, 1, num_turb)
    
    # Máscara para turbinas a jusante para todo o lote
    mask = x_diff > 0

    # Inicializa matrizes 3D com zeros
    sigma = np.zeros_like(x_diff)
    # k * x_diff[mask] -> shape (pop_size, n_turb, n_turb)
    sigma[mask] = k * x_diff[mask] + turb_diam / np.sqrt(8.)
    
    # Evita divisão por zero onde sigma é zero
    sigma_no_zero = np.where(sigma == 0, 1e-9, sigma)

    exponent_val = np.zeros_like(x_diff)
    exponent_val[mask] = -0.5 * (y_diff[mask] / sigma_no_zero[mask])**2
    
    radical = np.ones_like(x_diff)
    radical[mask] = 1. - CT / (8. * sigma_no_zero[mask]**2 / turb_diam**2)
    
    radical_val = np.ones_like(x_diff)
    radical_val[mask] = np.sqrt(np.maximum(radical[mask], 0))

    loss_matrix = np.zeros_like(x_diff)
    loss_matrix[mask] = (1. - radical_val[mask]) * np.exp(exponent_val[mask])

    # Agrega perdas para cada turbina em cada fazenda do lote
    # Soma ao longo do eixo 2 (as turbinas que causam o wake)
    loss = np.sqrt(np.sum(loss_matrix**2, axis=2)) # Shape: (pop_size, num_turb)
    
    return loss


def DirPower_batch(turb_coords_batch, wind_dir_deg, wind_speed,
                   turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    """Calcula a potência para um lote de fazendas eólicas em uma direção de vento."""
    pop_size, num_turb, _ = turb_coords_batch.shape

    frame_coords_batch = WindFrame_batch(turb_coords_batch, wind_dir_deg)
    loss = GaussianWake_batch(frame_coords_batch, turb_diam) # Shape: (pop_size, num_turb)
    
    wind_speed_eff = wind_speed * (1. - loss) # wind_speed é escalar, broadcasting funciona
    
    turb_pwr = np.zeros((pop_size, num_turb))

    # Vetorização em todo o lote
    mask1 = (turb_ci <= wind_speed_eff) & (wind_speed_eff < rated_ws)
    turb_pwr[mask1] = rated_pwr * ((wind_speed_eff[mask1] - turb_ci) / (rated_ws - turb_ci))**3
    
    mask2 = (rated_ws <= wind_speed_eff) & (wind_speed_eff < turb_co)
    turb_pwr[mask2] = rated_pwr

    # Soma a potência de todas as turbinas para cada fazenda no lote
    pwrDir = np.sum(turb_pwr, axis=1) # Shape: (pop_size,)
    
    return pwrDir


def calcAEP_batch(turb_coords_batch, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    """Calcula o AEP para um lote de fazendas eólicas."""
    pop_size = turb_coords_batch.shape[0]
    num_bins = len(wind_freq)

    pwr_produced = np.zeros((pop_size, num_bins))

    # Itera sobre as direções do vento (bins)
    for i in range(num_bins):
        # Calcula a potência para todo o lote para esta direção de vento
        pwr_produced[:, i] = DirPower_batch(turb_coords_batch, wind_dir[i], wind_speed[i],
                                            turb_diam, turb_ci, turb_co,
                                            rated_ws, rated_pwr)

    hrs_per_year = 365. * 24.
    
    # Multiplica a potência produzida pela frequência do vento (broadcasting)
    # wind_freq shape: (num_bins,) -> (1, num_bins)
    # pwr_produced shape: (pop_size, num_bins)
    AEP_per_bin = hrs_per_year * (wind_freq.reshape(1, -1) * pwr_produced)
    
    # Soma o AEP de todos os bins para cada fazenda no lote
    AEP = np.sum(AEP_per_bin, axis=1) # Shape: (pop_size,)
    AEP /= 1.E6  # Convert to MWh

    return AEP

# Funções de leitura de YAML permanecem as mesmas, usando CPU/NumPy
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
