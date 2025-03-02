import numpy as np
import time

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
    exponent[mask] = -0.5 * (y_diff[mask] / sigma[mask])**2

    # Calcula o radical (fator do modelo Bastankhah) somente para os valores válidos
    # Para turbinas que não estão a jusante, deixamos o valor neutro (1.0)
    radical = np.ones_like(sigma)
    radical[mask] = 1. - CT / (8. * sigma[mask]**2 / turb_diam**2)

    # Para evitar valores negativos na raiz, aplicamos np.maximum e calculamos a raiz
    radical_val = np.ones_like(sigma)
    radical_val[mask] = np.sqrt(np.maximum(radical[mask], 0))

    # Calcula a matriz de perda utilizando a equação do modelo
    loss_matrix = np.zeros_like(sigma)
    loss_matrix[mask] = (1. - radical_val[mask]) * np.exp(exponent[mask])

    # Agrega as perdas para cada turbina usando a raiz da soma dos quadrados
    loss = np.sqrt(np.sum(loss_matrix**2, axis=1))

    return loss

def GaussianWake_original(frame_coords, turb_diam):
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

# Gerando dados aleatórios para simular turbinas (100 turbinas em um espaço 2D)
num_turbinas = 100
frame_coords = np.random.rand(num_turbinas, 2) * 1000  # Coordenadas x e y entre 0 e 1000
turb_diam = 100  # Diâmetro das turbinas

# Função original (sem vetorização)
def test_original():
    start = time.time()
    loss = GaussianWake_original(frame_coords, turb_diam)
    end = time.time()
    print(f"Tempo sem vetorização: {end - start:.4f} segundos")
    return loss

# Função vetorizada
def test_vectorized():
    start = time.time()
    loss = GaussianWake_vetorizado_optimizado(frame_coords, turb_diam)
    end = time.time()
    print(f"Tempo com vetorização: {end - start:.4f} segundos")
    return loss

# Rodando os testes
loss_original = test_original()
loss_vectorized = test_vectorized()

# Comparando resultados
diff = np.linalg.norm(loss_original - loss_vectorized)
print(f"Erro absoluto entre as versões: {diff:.6f}")
