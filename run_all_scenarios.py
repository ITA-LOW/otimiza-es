"""
Script para rodar otimização multiobjetivo para todos os cenários (16, 36, 64 turbinas).
Executa automaticamente a otimização completa (Fase 1 + Fase 2) para cada cenário
e salva os resultados em diretórios separados.

COMO FUNCIONA:
1. Para cada cenário (16, 36, 64 turbinas):
   - Cria uma cópia temporária do multi16_prioriza_aep.py
   - Modifica os parâmetros (IND_SIZE, CIRCLE_RADIUS, MAX_GRUPOS, YAML, diretório de saída)
   - Modifica PATIENCE da Fase 2 (para rodar mais tempo)
   - Executa o script modificado (Fase 1 + Fase 2)
   - Remove o script temporário após execução

2. Salva resultados em diretórios separados:
   - 16 turbinas → pareto_front_results/
   - 36 turbinas → pareto_front_results_36/
   - 64 turbinas → pareto_front_results_64/

3. Ao final, pode gerar figuras automaticamente

PARÂMETROS AJUSTÁVEIS (por cenário):
Cada cenário (16, 36, 64 turbinas) pode ter parâmetros diferentes do GA:
- POP_SIZE_P1/P2: Tamanho da população (Fase 1 e 2)
- NGEN_P1/P2: Número máximo de gerações
- CXPB_P1/P2: Probabilidade de crossover
- MUTPB_P1/P2: Probabilidade de mutação
- PATIENCE_P1/P2: Gerações sem melhoria antes de parar

Exemplo: Problemas maiores (64 turbinas) podem precisar de:
- POP_SIZE maior (ex: 400-500)
- NGEN maior (ex: 600-800)
- PATIENCE maior (ex: 300-400)

Autor: [Seu Nome]
Data: 2025
"""

import os
import sys
import time
import subprocess

# =============================================================================
# CONFIGURAÇÕES DOS CENÁRIOS
# =============================================================================

SCENARIOS = {
    16: {
        'IND_SIZE': 16,
        'CIRCLE_RADIUS': 1300,
        'MAX_GRUPOS': 16,
        'yaml_file': 'iea37-ex16.yaml',
        'output_dir': 'pareto_front_results',
        # Parâmetros do GA específicos para 16 turbinas
        'POP_SIZE_P1': 300,
        'NGEN_P1': 500,
        'CXPB_P1': 0.95,
        'MUTPB_P1': 0.70,
        'INDPB_P1': 0.40,
        'PATIENCE_P1': 150,
        'POP_SIZE_P2': 300,
        'NGEN_P2': 1500,
        'CXPB_P2': 0.95,
        'MUTPB_P2': 0.70,
        'INDPB_P2': 0.40,
        'PATIENCE_P2': 350,
    },
    36: {
        'IND_SIZE': 36,
        'CIRCLE_RADIUS': 2000,
        'MAX_GRUPOS': 36,
        'yaml_file': 'iea37-ex36.yaml',
        'output_dir': 'pareto_front_results_36',
        # Parâmetros do GA específicos para 36 turbinas
        'POP_SIZE_P1': 300,
        'NGEN_P1': 500,
        'CXPB_P1': 1.00,
        'MUTPB_P1': 0.35,
        'INDPB_P1': 0.10,
        'PATIENCE_P1': 150,
        'POP_SIZE_P2': 300,
        'NGEN_P2': 1500,
        'CXPB_P2': 1.00,
        'MUTPB_P2': 0.35,
        'INDPB_P2': 0.10,
        'PATIENCE_P2': 350,
    },
    64: {
        'IND_SIZE': 64,
        'CIRCLE_RADIUS': 3000,
        'MAX_GRUPOS': 64,
        'yaml_file': 'iea37-ex64.yaml',
        'output_dir': 'pareto_front_results_64',
        # Parâmetros do GA específicos para 64 turbinas
        'POP_SIZE_P1': 300,
        'NGEN_P1': 500,
        'CXPB_P1': 0.80,
        'MUTPB_P1': 0.20,
        'INDPB_P1': 0.20,
        'PATIENCE_P1': 150,
        'POP_SIZE_P2': 300,
        'NGEN_P2': 1500,
        'CXPB_P2': 0.80,
        'MUTPB_P2': 0.20,
        'INDPB_P2': 0.20,
        'PATIENCE_P2': 350,
    }
}

# =============================================================================
# FUNÇÃO PARA MODIFICAR O SCRIPT DE OTIMIZAÇÃO
# =============================================================================

def modify_optimization_script(n_turbines, scenario_config):
    """
    Modifica o script multi16_prioriza_aep.py para o cenário específico.
    Cria uma cópia temporária com os parâmetros corretos.
    
    Args:
        n_turbines: Número de turbinas
        scenario_config: Dicionário com configurações
        
    Returns:
        Caminho do script modificado
    """
    script_path = os.path.join('multi_objetivo', 'multi16_prioriza_aep.py')
    temp_script = os.path.join('multi_objetivo', f'multi16_prioriza_aep_{n_turbines}.py')
    
    # Lê o script original
    with open(script_path, 'r') as f:
        code = f.read()
    
    # Modifica as constantes principais
    code = code.replace('IND_SIZE = 16', f'IND_SIZE = {scenario_config["IND_SIZE"]}')
    code = code.replace('CIRCLE_RADIUS = 1300', f'CIRCLE_RADIUS = {scenario_config["CIRCLE_RADIUS"]}')
    code = code.replace('MAX_GRUPOS = 16', f'MAX_GRUPOS = {scenario_config["MAX_GRUPOS"]}')
    
    # Modifica o arquivo YAML
    code = code.replace('iea37-ex16.yaml', scenario_config['yaml_file'])
    
    # Modifica parâmetros do GA na função main() (linhas ~1208-1217)
    # Fase 1
    code = code.replace('POP_SIZE_P1 = 300', f'POP_SIZE_P1 = {scenario_config["POP_SIZE_P1"]}')
    code = code.replace('NGEN_P1 = 500   # EXATO: 500 gerações', f'NGEN_P1 = {scenario_config["NGEN_P1"]}   # EXATO: {scenario_config["NGEN_P1"]} gerações')
    code = code.replace('NGEN_P1 = 500', f'NGEN_P1 = {scenario_config["NGEN_P1"]}')  # Fallback caso não tenha o comentário
    code = code.replace('CXPB_P1 = 0.95', f'CXPB_P1 = {scenario_config["CXPB_P1"]}')
    code = code.replace('MUTPB_P1 = 0.7', f'MUTPB_P1 = {scenario_config["MUTPB_P1"]}')
    
    # Fase 2
    code = code.replace('POP_SIZE_P2 = 300', f'POP_SIZE_P2 = {scenario_config["POP_SIZE_P2"]}')
    code = code.replace('NGEN_P2 = 500', f'NGEN_P2 = {scenario_config["NGEN_P2"]}')
    code = code.replace('CXPB_P2 = 0.95', f'CXPB_P2 = {scenario_config["CXPB_P2"]}')
    code = code.replace('MUTPB_P2 = 0.7', f'MUTPB_P2 = {scenario_config["MUTPB_P2"]}')
    
    # Modifica INDPB (individual probability) no registro do toolbox
    # Fase 1: linha ~520 (toolbox_phase1.register("mutate", ...))
    code = code.replace('toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=100, indpb=0.4)',
                        f'toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=100, indpb={scenario_config["INDPB_P1"]})')
    
    # Fase 2: linha ~560 (toolbox_phase2.register("mutate", ...))
    code = code.replace('toolbox_phase2.register("mutate", mutate_phase2, mu=0, sigma=100, indpb=0.4)',
                        f'toolbox_phase2.register("mutate", mutate_phase2, mu=0, sigma=100, indpb={scenario_config["INDPB_P2"]})')
    
    # Também modifica nas re-registrações durante fase agressiva (linhas ~649 e ~661)
    code = code.replace('toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_AGGRESSIVE, indpb=0.4)',
                        f'toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_AGGRESSIVE, indpb={scenario_config["INDPB_P1"]})')
    code = code.replace('toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_NORMAL, indpb=0.4)',
                        f'toolbox_phase1.register("mutate", mutate_phase1, mu=0, sigma=SIGMA_NORMAL, indpb={scenario_config["INDPB_P1"]})')
    
    # Modifica PATIENCE (dentro das funções optimize_phase1 e optimize_phase2)
    # Fase 1: PATIENCE dentro de optimize_phase1 (linha ~595)
    lines = code.split('\n')
    phase1_patience_found = False
    for i, line in enumerate(lines):
        if 'def optimize_phase1' in line:
            phase1_patience_found = True
        if phase1_patience_found and 'PATIENCE = 150' in line:
            lines[i] = line.replace('PATIENCE = 150', f'PATIENCE = {scenario_config["PATIENCE_P1"]}')
            break
    code = '\n'.join(lines)
    
    # Fase 2: PATIENCE dentro de optimize_phase2 (linha ~892)
    lines = code.split('\n')
    phase2_patience_found = False
    for i, line in enumerate(lines):
        if 'def optimize_phase2' in line:
            phase2_patience_found = True
        if phase2_patience_found and 'PATIENCE = 100' in line:
            lines[i] = line.replace('PATIENCE = 100', f'PATIENCE = {scenario_config["PATIENCE_P2"]}')
            break
    code = '\n'.join(lines)
    
    # Modifica TODAS as referências a pareto_front_results (com e sem aspas)
    output_dir = scenario_config['output_dir']
    # Substitui com aspas duplas
    code = code.replace('"pareto_front_results"', f'"{output_dir}"')
    # Substitui com aspas simples
    code = code.replace("'pareto_front_results'", f"'{output_dir}'")
    # Substitui sem aspas (em join paths)
    code = code.replace('pareto_front_results/', f'{output_dir}/')
    code = code.replace('pareto_front_results\\', f'{output_dir}\\')
    
    # Salva script temporário
    with open(temp_script, 'w') as f:
        f.write(code)
    
    return temp_script


# =============================================================================
# FUNÇÃO PARA EXECUTAR OTIMIZAÇÃO
# =============================================================================

def run_optimization_for_scenario(n_turbines, scenario_config):
    """
    Executa a otimização multiobjetivo para um cenário específico.
    
    Args:
        n_turbines: Número de turbinas (16, 36 ou 64)
        scenario_config: Dicionário com configurações do cenário
        
    Returns:
        True se sucesso, False caso contrário
    """
    print("\n" + "=" * 80)
    print(f"INICIANDO OTIMIZAÇÃO PARA {n_turbines} TURBINAS")
    print("=" * 80)
    
    # Cria diretório de saída
    output_dir = scenario_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'evolution'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'evolution_phase1'), exist_ok=True)
    
    # Modifica o script
    temp_script = modify_optimization_script(n_turbines, scenario_config)
    
    try:
        # Executa o script modificado
        print(f"Executando: python3 {temp_script}")
        result = subprocess.run(
            [sys.executable, temp_script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,  # Mostra output em tempo real
            text=True
        )
        
        success = (result.returncode == 0)
        
        # Remove script temporário
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        return success
        
    except Exception as e:
        print(f"ERRO ao executar otimização: {e}")
        # Remove script temporário mesmo em caso de erro
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return False


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """
    Função principal: executa otimização para todos os cenários.
    """
    print("=" * 80)
    print("EXECUÇÃO AUTOMÁTICA - OTIMIZAÇÃO MULTIOBJETIVO")
    print("Cenários: 16, 36 e 64 turbinas")
    print("=" * 80)
    
    start_time_total = time.time()
    
    # Lista de cenários para executar
    # Pode modificar para rodar apenas alguns: [16] ou [16, 36] etc.
    scenarios_to_run = [16, 36, 64]
    
    results = {}
    
    for n_turbines in scenarios_to_run:
        if n_turbines not in SCENARIOS:
            print(f"\nAVISO: Cenário de {n_turbines} turbinas não configurado. Pulando...")
            continue
        
        scenario_config = SCENARIOS[n_turbines]
        
        print(f"\n{'='*80}")
        print(f"PREPARANDO CENÁRIO DE {n_turbines} TURBINAS")
        print(f"{'='*80}")
        print(f"  IND_SIZE: {scenario_config['IND_SIZE']}")
        print(f"  CIRCLE_RADIUS: {scenario_config['CIRCLE_RADIUS']} m")
        print(f"  MAX_GRUPOS: {scenario_config['MAX_GRUPOS']}")
        print(f"  YAML: {scenario_config['yaml_file']}")
        print(f"  Output: {scenario_config['output_dir']}")
        
        # Executa otimização
        scenario_start = time.time()
        success = run_optimization_for_scenario(n_turbines, scenario_config)
        scenario_time = time.time() - scenario_start
        
        results[n_turbines] = {
            'success': success,
            'time': scenario_time
        }
        
        if success:
            print(f"\n✓ Cenário de {n_turbines} turbinas concluído com sucesso!")
            print(f"  Tempo: {scenario_time/60:.2f} minutos ({scenario_time/3600:.2f} horas)")
        else:
            print(f"\n✗ Erro ao executar cenário de {n_turbines} turbinas")
    
    # Resumo final
    total_time = time.time() - start_time_total
    print("\n" + "=" * 80)
    print("RESUMO DA EXECUÇÃO")
    print("=" * 80)
    
    for n_turbines, result in results.items():
        status = "✓ SUCESSO" if result['success'] else "✗ ERRO"
        print(f"  {n_turbines} turbinas: {status} ({result['time']/60:.2f} min)")
    
    print(f"\nTempo total: {total_time/60:.2f} minutos ({total_time/3600:.2f} horas)")
    
    # Pergunta se quer gerar as figuras
    print("\n" + "=" * 80)
    print("PRÓXIMO PASSO: Gerar figuras do artigo")
    print("=" * 80)
    print("Execute: python3 generate_article_figures.py")
    print("=" * 80)
    
    # Opcional: gerar figuras automaticamente
    generate_figures = input("\nDeseja gerar as figuras automaticamente agora? (s/n): ").lower().strip()
    if generate_figures == 's':
        print("\nGerando figuras...")
        result = subprocess.run(
            [sys.executable, 'generate_article_figures.py'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            print("\n✓ Figuras geradas com sucesso!")
        else:
            print("\n✗ Erro ao gerar figuras")


if __name__ == "__main__":
    main()
