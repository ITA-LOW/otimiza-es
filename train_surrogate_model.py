# train_surrogate_model.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# --- Tenta importar TensorFlow e dá uma mensagem de erro clara se não estiver instalado ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Erro: A biblioteca 'tensorflow' é necessária para treinar o modelo.")
    print("Por favor, instale as dependências com: pip install -r requirements.txt")
    exit()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- 1. Configurações e Caminhos ---
project_root = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(project_root, 'output', 'multi_objective_sub')
data_path = os.path.join(output_dir, 'training_data.csv')
model_path = os.path.join(output_dir, 'surrogate_model.h5')
input_scaler_path = os.path.join(output_dir, 'input_scaler.pkl')
output_scaler_path = os.path.join(output_dir, 'output_scaler.pkl')
history_plot_path = os.path.join(output_dir, 'training_history.png')

# --- 2. Carregar e Preparar os Dados ---
print(f"Carregando dados de treinamento de: {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print("Erro: Arquivo 'training_data.csv' não encontrado.")
    print("Por favor, execute 'generate_training_data.py' primeiro.")
    exit()

print(f"Número de amostras carregadas: {len(df)}")
df.dropna(inplace=True) # Remove qualquer linha com NaN
print(f"Número de amostras após limpeza: {len(df)}")


# Separar features (entradas X) e labels (saídas Y)
n_turbines = 16
input_cols = [f't{j}_{coord}' for j in range(n_turbines) for coord in ['x', 'y']] + ['sub_x', 'sub_y']
output_cols = ['aep', 'cost']

X = df[input_cols].values
Y = df[output_cols].values

# Normalizar os dados é CRUCIAL para o bom treinamento da rede neural
print("Normalizando os dados...")
input_scaler = MinMaxScaler()
X_scaled = input_scaler.fit_transform(X)

output_scaler = MinMaxScaler()
Y_scaled = output_scaler.fit_transform(Y)

# Salvar os scalers para uso futuro na otimização
joblib.dump(input_scaler, input_scaler_path)
joblib.dump(output_scaler, output_scaler_path)
print(f"Scalers de entrada e saída salvos em: {output_dir}")


# Dividir em conjuntos de treinamento e validação
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
print(f"Tamanho do conjunto de treinamento: {len(X_train)}")
print(f"Tamanho do conjunto de validação: {len(X_val)}")


# --- 3. Construir a Arquitetura da Rede Neural ---
def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Dropout para prevenir overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        # Camada de saída com 2 neurônios (para AEP e Custo) e ativação linear (para regressão)
        layers.Dense(2, activation='linear') 
    ])
    
    # Usar o otimizador Adam e a perda de erro quadrático médio (padrão para regressão)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model(X_train.shape[1])
model.summary()


# --- 4. Treinar o Modelo ---
print("\nIniciando o treinamento do modelo substituto...")

# Callback para parar o treinamento se a perda de validação não melhorar
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20, # Número de épocas sem melhora antes de parar
    restore_best_weights=True # Restaura os pesos da melhor época
)

EPOCHS = 200
BATCH_SIZE = 32

history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("Treinamento concluído.")

# --- 5. Salvar o Modelo e Visualizar o Histórico ---

# Salvar o modelo treinado
model.save(model_path)
print(f"Modelo substituto salvo em: {model_path}")

# Plotar o histórico de treinamento
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Histórico de Perda do Modelo')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.xlabel('Época')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Gráfico do histórico de treinamento salvo em: {history_plot_path}")

plot_history(history)
