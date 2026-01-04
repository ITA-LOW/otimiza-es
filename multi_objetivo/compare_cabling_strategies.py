"""
Script para comparar as estratégias de cabeamento V1 e V3.

Executa o algoritmo de otimização com ambas as abordagens e compara:
- Tempo de execução
- Custo final
- AEP líquido
- Qualidade do Pareto front
- Estatísticas descritivas

Uso:
    python3 multi_objetivo/compare_cabling_strategies.py
"""

import sys
import os
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa o módulo principal (vamos modificá-lo dinamicamente)
import importlib.util

def run_optimization_with_cabling_version(version='v3', run_id=1):
    """
    Executa a otimização com uma versão específica de cabeamento.
    
    Args:
        version: 'v1' ou 'v3'
        run_id: ID da execução (para múltiplas execuções)
    
    Returns:
        dict com métricas da execução
    """
    print(f"\n{'='*80}")
    print(f"EXECUTANDO COM CABLING_{version.upper()}")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}\n")
    
    # Carrega o módulo multi16_prioriza_aep
    module_path = os.path.join(os.path.dirname(__file__), 'multi16_prioriza_aep.py')
    spec = importlib.util.spec_from_file_location("multi16_prioriza_aep", module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Modifica o import antes de carregar
    # Precisamos fazer isso de forma diferente - vamos modificar o arquivo temporariamente
    # ou melhor, vamos criar uma cópia modificada
    
    # Lê o arquivo original
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Substitui o import e chamadas baseado na versão
    modified_content = content  # Inicializa
    
    if version == 'v1':
        # Substitui import
        modified_content = modified_content.replace(
            'import multi_objetivo.cabling_v3 as cabling_v3',
            'import multi_objetivo.cabling_v1 as cabling_v1'
        )
        # Substitui chamadas de função (cuidado com ordem para não substituir duas vezes)
        modified_content = modified_content.replace(
            'cabling_v3.analisar_layout_completo',
            'cabling_v1.analisar_layout_completo'
        )
        # Substitui parâmetro sub por substation_idx
        modified_content = modified_content.replace(
            'sub=ponto_de_coleta_idx',
            'substation_idx=ponto_de_coleta_idx'
        )
    else:  # v3
        # Garante que está usando v3 (pode já estar)
        if 'import multi_objetivo.cabling_v1 as cabling_v1' in modified_content:
            modified_content = modified_content.replace(
                'import multi_objetivo.cabling_v1 as cabling_v1',
                'import multi_objetivo.cabling_v3 as cabling_v3'
            )
        if 'cabling_v1.analisar_layout_completo' in modified_content:
            modified_content = modified_content.replace(
                'cabling_v1.analisar_layout_completo',
                'cabling_v3.analisar_layout_completo'
            )
        # Garante que está usando sub (pode já estar)
        modified_content = modified_content.replace(
            'substation_idx=ponto_de_coleta_idx',
            'sub=ponto_de_coleta_idx'
        )
    
    # Salva em arquivo temporário
    temp_file = f"multi16_prioriza_aep_temp_{version}_{run_id}.py"
    temp_path = os.path.join(os.path.dirname(__file__), temp_file)
    
    with open(temp_path, 'w') as f:
        f.write(modified_content)
    
    try:
        # Executa o módulo modificado
        spec = importlib.util.spec_from_file_location("temp_module", temp_path)
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules['temp_module'] = temp_module
        
        # Salva diretório de resultados original
        original_results_dir = "pareto_front_results"
        results_dir = f"pareto_front_results_{version}_{run_id}"
        
        # Remove diretório de resultados anterior se existir
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        
        # Executa e mede tempo
        start_time = time.time()
        
        # Executa o main do módulo
        try:
            spec.loader.exec_module(temp_module)
            if hasattr(temp_module, 'main'):
                temp_module.main()
            else:
                print("AVISO: Função main() não encontrada no módulo")
        except Exception as e:
            print(f"ERRO ao executar módulo: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        execution_time = time.time() - start_time
        
        # Move resultados para diretório específico da versão
        if os.path.exists(original_results_dir):
            shutil.move(original_results_dir, results_dir)
        
        # Lê resultados
        csv_path = os.path.join(results_dir, "pareto_summary.csv")
        if os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)
            
            metrics = {
                'version': version,
                'run_id': run_id,
                'execution_time_min': execution_time / 60,
                'execution_time_sec': execution_time,
                'n_solutions': len(df_results),
                'best_aep_mwh': df_results['AEP_Liquido_MWh'].max(),
                'best_aep_gwh': df_results['AEP_Liquido_MWh'].max() / 1000,
                'mean_aep_mwh': df_results['AEP_Liquido_MWh'].mean(),
                'std_aep_mwh': df_results['AEP_Liquido_MWh'].std(),
                'best_cost_usd': df_results['Custo_USD'].min(),
                'mean_cost_usd': df_results['Custo_USD'].mean(),
                'std_cost_usd': df_results['Custo_USD'].std(),
                'mean_n_grupos': df_results['N_Grupos'].mean(),
                'mean_perdas_mwh': df_results['Perdas_Joule_MWh'].mean(),
                'results_dir': results_dir
            }
        else:
            print(f"AVISO: Arquivo de resultados não encontrado: {csv_path}")
            metrics = {
                'version': version,
                'run_id': run_id,
                'execution_time_min': execution_time / 60,
                'execution_time_sec': execution_time,
                'error': 'Results file not found'
            }
        
        return metrics
        
    finally:
        # Limpa arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)

def compare_strategies(n_runs=1):
    """
    Compara as estratégias V1 e V3 executando ambas n_runs vezes.
    
    Args:
        n_runs: Número de execuções para cada estratégia (para estatísticas)
    """
    print("\n" + "="*80)
    print("COMPARAÇÃO DE ESTRATÉGIAS DE CABEAMENTO")
    print("V1: KMeans + Balanceamento Refinado")
    print("V3: Agrupamento Angular + Balanceamento Rápido")
    print("="*80)
    
    all_results = []
    
    # Executa cada versão n_runs vezes
    for run_id in range(1, n_runs + 1):
        print(f"\n{'#'*80}")
        print(f"EXECUÇÃO {run_id}/{n_runs}")
        print(f"{'#'*80}")
        
        # Executa V3
        try:
            metrics_v3 = run_optimization_with_cabling_version('v3', run_id)
            all_results.append(metrics_v3)
        except Exception as e:
            print(f"ERRO ao executar V3: {e}")
            import traceback
            traceback.print_exc()
        
        # Executa V1
        try:
            metrics_v1 = run_optimization_with_cabling_version('v1', run_id)
            all_results.append(metrics_v1)
        except Exception as e:
            print(f"ERRO ao executar V1: {e}")
            import traceback
            traceback.print_exc()
    
    # Cria DataFrame com todos os resultados
    df_all = pd.DataFrame(all_results)
    
    # Salva resultados brutos
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    df_all.to_csv(os.path.join(output_dir, "raw_comparison_results.csv"), index=False)
    
    # Calcula estatísticas agregadas por versão
    if n_runs > 1:
        df_summary = df_all.groupby('version').agg({
            'execution_time_min': ['mean', 'std', 'min', 'max'],
            'best_aep_gwh': ['mean', 'std', 'min', 'max'],
            'mean_aep_mwh': ['mean', 'std'],
            'best_cost_usd': ['mean', 'std', 'min', 'max'],
            'mean_cost_usd': ['mean', 'std'],
            'mean_n_grupos': ['mean', 'std'],
            'n_solutions': ['mean', 'std']
        }).round(2)
        
        df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
        df_summary.to_csv(os.path.join(output_dir, "comparison_summary.csv"))
        
        print("\n" + "="*80)
        print("RESUMO COMPARATIVO")
        print("="*80)
        print(df_summary)
    else:
        # Para uma única execução, mostra comparação direta
        df_v1 = df_all[df_all['version'] == 'v1']
        df_v3 = df_all[df_all['version'] == 'v3']
        
        if len(df_v1) > 0 and len(df_v3) > 0:
            print("\n" + "="*80)
            print("COMPARAÇÃO DIRETA")
            print("="*80)
            
            comparison = {
                'Métrica': [
                    'Tempo de Execução (min)',
                    'Melhor AEP (GWh)',
                    'AEP Médio (MWh)',
                    'Melhor Custo (USD)',
                    'Custo Médio (USD)',
                    'Número de Soluções',
                    'N Grupos Médio',
                    'Perdas Médias (MWh)'
                ],
                'V1': [
                    f"{df_v1['execution_time_min'].iloc[0]:.2f}",
                    f"{df_v1['best_aep_gwh'].iloc[0]:.2f}",
                    f"{df_v1['mean_aep_mwh'].iloc[0]:.2f}",
                    f"{df_v1['best_cost_usd'].iloc[0]:.2f}",
                    f"{df_v1['mean_cost_usd'].iloc[0]:.2f}",
                    f"{df_v1['n_solutions'].iloc[0]}",
                    f"{df_v1['mean_n_grupos'].iloc[0]:.2f}",
                    f"{df_v1['mean_perdas_mwh'].iloc[0]:.2f}"
                ],
                'V3': [
                    f"{df_v3['execution_time_min'].iloc[0]:.2f}",
                    f"{df_v3['best_aep_gwh'].iloc[0]:.2f}",
                    f"{df_v3['mean_aep_mwh'].iloc[0]:.2f}",
                    f"{df_v3['best_cost_usd'].iloc[0]:.2f}",
                    f"{df_v3['mean_cost_usd'].iloc[0]:.2f}",
                    f"{df_v3['n_solutions'].iloc[0]}",
                    f"{df_v3['mean_n_grupos'].iloc[0]:.2f}",
                    f"{df_v3['mean_perdas_mwh'].iloc[0]:.2f}"
                ]
            }
            
            # Calcula diferença percentual
            diff = []
            for i in range(len(comparison['Métrica'])):
                try:
                    v1_val = float(comparison['V1'][i])
                    v3_val = float(comparison['V3'][i])
                    if v1_val != 0:
                        pct_diff = ((v3_val - v1_val) / v1_val) * 100
                        diff.append(f"{pct_diff:+.2f}%")
                    else:
                        diff.append("N/A")
                except:
                    diff.append("N/A")
            
            comparison['Diferença (%)'] = diff
            
            df_comparison = pd.DataFrame(comparison)
            print(df_comparison.to_string(index=False))
            
            # Salva comparação
            df_comparison.to_csv(os.path.join(output_dir, "direct_comparison.csv"), index=False)
            
            # Análise de velocidade
            speedup = df_v1['execution_time_min'].iloc[0] / df_v3['execution_time_min'].iloc[0]
            print(f"\n{'='*80}")
            print(f"ANÁLISE DE VELOCIDADE")
            print(f"{'='*80}")
            print(f"V3 é {speedup:.2f}x mais rápido que V1")
            print(f"Tempo V1: {df_v1['execution_time_min'].iloc[0]:.2f} min")
            print(f"Tempo V3: {df_v3['execution_time_min'].iloc[0]:.2f} min")
            print(f"Economia de tempo: {df_v1['execution_time_min'].iloc[0] - df_v3['execution_time_min'].iloc[0]:.2f} min")
    
    print(f"\n{'='*80}")
    print("RESULTADOS SALVOS EM:")
    print(f"  - {os.path.join(output_dir, 'raw_comparison_results.csv')}")
    if n_runs > 1:
        print(f"  - {os.path.join(output_dir, 'comparison_summary.csv')}")
    else:
        print(f"  - {os.path.join(output_dir, 'direct_comparison.csv')}")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compara estratégias de cabeamento V1 e V3')
    parser.add_argument('--runs', type=int, default=1,
                       help='Número de execuções para cada estratégia (default: 1)')
    
    args = parser.parse_args()
    
    compare_strategies(n_runs=args.runs)

