import numpy as np
from multiprocessing import Pool, cpu_count

def GaussianWake(frame_coords, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    num_turb = len(frame_coords)

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)

    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = frame_coords[i, 0] - frame_coords[j, 0]   # Calculate the x-dist
            y = frame_coords[i, 1] - frame_coords[j, 1]   # And the y-offset
            if x > 0.:                   # If Primary is downwind of the Target
                sigma = k*x + turb_diam/np.sqrt(8.)  # Calculate the wake loss
                # Simplified Bastankhah Gaussian wake model
                exponent = -0.5 * (y/sigma)**2
                radical = 1. - CT/(8.*sigma**2 / turb_diam**2)
                loss_array[j] = (1.-np.sqrt(radical)) * np.exp(exponent)
            # Note that if the Target is upstream, loss is defaulted to zero
        # Total wake losses from all upstream turbs, using sqrt of sum of sqrs
        loss[i] = np.sqrt(np.sum(loss_array**2))

    return loss

def gaussian_wake_single(i, frame_coords, turb_diam, CT, k, num_turb):
    loss_array = np.zeros(num_turb)
    x_i, y_i = frame_coords[i]
    for j in range(num_turb):
        x = x_i - frame_coords[j, 0]
        y = y_i - frame_coords[j, 1]
        if x > 0:
            sigma = k * x + turb_diam / np.sqrt(8.)
            exponent = -0.5 * (y / sigma) ** 2
            radical = 1. - CT / (8. * sigma ** 2 / turb_diam ** 2)
            loss_array[j] = (1. - np.sqrt(radical)) * np.exp(exponent)
    return np.sqrt(np.sum(loss_array ** 2))

def GaussianWakeParallel(frame_coords, turb_diam):
    num_turb = len(frame_coords)
    CT = 4.0 * 1. / 3. * (1.0 - 1. / 3.)
    k = 0.0324555
    
    with Pool(cpu_count()) as pool:
        loss = pool.starmap(gaussian_wake_single, [(i, frame_coords, turb_diam, CT, k, num_turb) for i in range(num_turb)])
    
    return np.array(loss)

# Teste de desempenho
def test_parallelization():
    np.random.seed(42)
    num_turb = 100  # Número de turbinas
    frame_coords = np.random.rand(num_turb, 2) * 1000  # Coordenadas aleatórias
    turb_diam = 100
    
    import time
    
    # Sem paralelização
    start = time.time()
    loss_serial = GaussianWake(frame_coords, turb_diam)
    time_serial = time.time() - start
    
    # Com paralelização
    start = time.time()
    loss_parallel = GaussianWakeParallel(frame_coords, turb_diam)
    time_parallel = time.time() - start
    
    # Comparação
    error = np.abs(loss_serial - loss_parallel).sum()
    print(f"Tempo sem paralelização: {time_serial:.4f} segundos")
    print(f"Tempo com paralelização: {time_parallel:.4f} segundos")
    print(f"Erro absoluto entre as versões: {error:.6f}")

test_parallelization()