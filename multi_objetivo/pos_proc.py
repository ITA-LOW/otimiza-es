import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# post_processamento.py
# Este script analisa os resultados da otimização multiobjetivo,
# identifica automaticamente os 3 layouts-arquétipo (Max AEP, Min Custo, Joelho)
# e gera visualizações detalhadas para cada um.

import numpy as np
import pandas as pd
import cabling # Importa nosso módulo refatorado
import matplotlib.pyplot as plt

# =============================================================================
# FUNÇÕES HELPER
# =============================================================================

def load_custom_coords(filepath):
    """Carrega as coordenadas do nosso formato customizado 'xc: [...]' / 'yc: [...]'."""
    with open(filepath, 'r') as f:
        # Adicionado .strip() para remover possíveis linhas em branco
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"Arquivo de coordenadas incompleto: {filepath}")
        x_line, y_line = lines
        
    x_content = x_line[x_line.find('[')+1 : x_line.rfind(']')]
    x_coords = [float(s) for s in x_content.split(',')]
    y_content = y_line[y_line.find('[')+1 : y_line.rfind(']')]
    y_coords = [float(s) for s in y_content.split(',')]
    return np.column_stack((x_coords, y_coords))

def identificar_arquetipos(df):
    """
    Identifica os índices das 3 soluções arquétipo em um DataFrame da Frente de Pareto.
    
    Retorna:
        dict: Um dicionário com os índices para 'max_aep', 'min_cost', e 'joelho'.
    """
    if df.empty:
        return {}

    # 1. Identificar Campeão de AEP e Campeão de Custo (simples)
    idx_max_aep = df['AEP_Liquido_MWh'].idxmax()
    idx_min_cost = df['Custo_USD'].idxmin()

    # 2. Identificar Ponto de Joelho (abordagem geométrica)
    
    # Normalizar os dados para a escala [0, 1] para que os eixos sejam comparáveis
    aep_norm = (df['AEP_Liquido_MWh'] - df['AEP_Liquido_MWh'].min()) / (df['AEP_Liquido_MWh'].max() - df['AEP_Liquido_MWh'].min())
    cost_norm = (df['Custo_USD'] - df['Custo_USD'].min()) / (df['Custo_USD'].max() - df['Custo_USD'].min())
    
    # Pontos extremos da linha (ponto de custo mínimo e ponto de aep máximo)
    p1 = np.array([cost_norm[idx_min_cost], aep_norm[idx_min_cost]])
    p2 = np.array([cost_norm[idx_max_aep], aep_norm[idx_max_aep]])
    
    # Calcular a distância de cada ponto na frente até a linha que conecta os extremos
    all_points = np.column_stack((cost_norm, aep_norm))
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    vec_from_p1 = all_points - p1
    
    # Projeção escalar para encontrar o ponto mais próximo na linha
    scalar_proj = np.dot(vec_from_p1, line_vec_norm)
    
    # Distância perpendicular (usando produto vetorial em 2D)
    # A fórmula é |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / sqrt((x2-x1)² + (y2-y1)²)
    # que é o módulo do produto vetorial da linha e do vetor ao ponto, dividido pelo comprimento da linha.
    distances = np.abs(np.cross(p2 - p1, all_points - p1)) / np.linalg.norm(p2 - p1)

    idx_joelho = np.argmax(distances)
    
    return {
        'max_aep': idx_max_aep,
        'min_cost': idx_min_cost,
        'joelho': idx_joelho
    }

def plotar_frente_de_pareto_enriquecida(df, arquetipos_idx, output_dir):
    """
    Gera um gráfico da Frente de Pareto de alta qualidade para publicação,
    destacando os 3 layouts arquétipo.
    """
    # Extrai as séries de dados para facilitar o acesso
    aep = df['AEP_Liquido_MWh']
    cost = df['Custo_USD'] / 1e6 # Convertido para Milhões de USD

    # Extrai os dados dos 3 arquétipos
    max_aep_sol = df.loc[arquetipos_idx['max_aep']]
    min_cost_sol = df.loc[arquetipos_idx['min_cost']]
    joelho_sol = df.loc[arquetipos_idx['joelho']]

    plt.figure(figsize=(12, 9))

    # 1. Plota a população geral de forma sutil
    plt.scatter(cost, aep, c='gray', alpha=0.5, label='Soluções da Frente de Pareto')

    # 2. Plota os arquétipos com destaque
    plt.scatter(min_cost_sol['Custo_USD']/1e6, min_cost_sol['AEP_Liquido_MWh'],
                c='green', s=150, edgecolor='black', marker='D', label='Arquétipo: Custo Mínimo')
    
    plt.scatter(max_aep_sol['Custo_USD']/1e6, max_aep_sol['AEP_Liquido_MWh'],
                c='red', s=150, edgecolor='black', marker='s', label='Arquétipo: AEP Máximo')

    plt.scatter(joelho_sol['Custo_USD']/1e6, joelho_sol['AEP_Liquido_MWh'],
                c='orange', s=200, edgecolor='black', marker='*', label='Arquétipo: Joelho (Compromisso)')

    # 3. Adiciona anotações de texto (opcional, mas muito eficaz)
    plt.text(min_cost_sol['Custo_USD']/1e6 * 1.01, min_cost_sol['AEP_Liquido_MWh'] - 500,
            f"Custo Min.\n${min_cost_sol['Custo_USD']/1e6:.3f} mi",
            fontsize=10, color='darkgreen', weight='bold')
            
    plt.text(max_aep_sol['Custo_USD']/1e6 * 0.98, max_aep_sol['AEP_Liquido_MWh'] + 300,
            f"AEP Max.\n{max_aep_sol['AEP_Liquido_MWh']:,.0f} MWh",
            fontsize=10, color='darkred', weight='bold', ha='right')
            
    plt.text(joelho_sol['Custo_USD']/1e6, joelho_sol['AEP_Liquido_MWh'] + 500,
            f"Joelho",
            fontsize=10, color='darkorange', weight='bold', ha='center')

    # 4. Polimento final do gráfico
    plt.title('Frente de Pareto: Análise de Trade-off entre AEP e Custo', fontsize=18, weight='bold')
    plt.xlabel('Custo Total do Cabeamento (Milhões de USD)', fontsize=14)
    plt.ylabel('AEP Líquido (MWh/ano)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Salva a figura
    output_path = os.path.join(output_dir, "pareto_front_enriquecido.png")
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico enriquecido da Frente de Pareto salvo em: '{output_path}'")
    plt.close()

# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    # Diretórios de entrada e saída
    INPUT_DIR = "pareto_front_results"
    OUTPUT_DIR = "archetype_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar o resumo da Frente de Pareto
    summary_path = os.path.join(INPUT_DIR, "pareto_summary.csv")
    try:
        df_pareto = pd.read_csv(summary_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de resumo não encontrado em '{summary_path}'.")
        print("Por favor, execute o script 'wind_farm_GA_multi.py' primeiro.")
        return

    print("--- Identificando Layouts Arquétipo ---")
    
    # Identificar os índices dos 3 arquétipos
    arquetipos_idx = identificar_arquetipos(df_pareto)
    
    if not arquetipos_idx:
        print("A Frente de Pareto está vazia. Nenhuma análise a ser feita.")
        return

    # Imprimir um resumo
    print(f"Campeão de AEP: Solução #{df_pareto.loc[arquetipos_idx['max_aep'], 'Solution']}")
    print(f"Campeão de Custo: Solução #{df_pareto.loc[arquetipos_idx['min_cost'], 'Solution']}")
    print(f"Solução de Joelho: Solução #{df_pareto.loc[arquetipos_idx['joelho'], 'Solution']}")

    print("\n--- Gerando Visualizações Detalhadas para cada Arquétipo ---")
    
    # Loop pelos 3 arquétipos para gerar suas imagens
    for nome_arquetipo, index in arquetipos_idx.items():
        # Obter os dados da solução
        solucao = df_pareto.loc[index]
        print(f"Processando '{nome_arquetipo}' (Solução #{solucao['Solution']})...")
        
        # Carregar as coordenadas
        coord_path = solucao['File']
        coords = load_custom_coords(coord_path)
        
        # Determinar a subestação (turbina mais próxima do continente)
        subestacao_continente = np.array([[-1350, 0]])
        distancias_ao_continente = np.linalg.norm(coords - subestacao_continente, axis=1)
        sub_idx = np.argmin(distancias_ao_continente)
        
        # Analisar o cabeamento para obter o objeto 'planta'
        planta, _ = cabling.analisar_layout_completo(coords, sub_idx)
        
        # Criar um título descritivo para o gráfico
        titulo = (f"Arquétipo: {nome_arquetipo.replace('_', ' ').title()}\n"
                  f"AEP Líquido: {solucao['AEP_Liquido_MWh']:,.1f} MWh | "
                  f"Custo: ${solucao['Custo_USD']/1e6:.2f} mi")
        
        # Gerar e salvar a imagem
        output_filename = os.path.join(OUTPUT_DIR, f"arquetipo_{nome_arquetipo}.png")
        cabling.plotar_cabeamento(planta, coords, sub_idx, titulo=titulo, output_filename=output_filename)

    print("\nProcesso concluído com sucesso!")
    print(f"As imagens dos 3 layouts arquétipo foram salvas em '{OUTPUT_DIR}/'.")

    print("\n--- Gerando gráfico enriquecido da Frente de Pareto ---")
    plotar_frente_de_pareto_enriquecida(df_pareto, arquetipos_idx, OUTPUT_DIR)

if __name__ == "__main__":
    main()