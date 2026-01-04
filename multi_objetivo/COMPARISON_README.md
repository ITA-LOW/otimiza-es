# Comparação de Estratégias de Cabeamento

Este script compara as duas abordagens de cabeamento (V1 e V3) para otimização de parques eólicos.

## Estratégias Comparadas

### V1: KMeans + Balanceamento Refinado
- **Agrupamento**: KMeans baseado em similaridade de cosseno
- **Balanceamento**: Algoritmo complexo com múltiplas fases
- **Dependências**: Requer sklearn
- **Características**: Mais lento, potencialmente melhor qualidade

### V3: Agrupamento Angular + Balanceamento Rápido
- **Agrupamento**: Divisão angular determinística
- **Balanceamento**: Algoritmo rápido com swaps limitados
- **Dependências**: Apenas numpy
- **Características**: Mais rápido, boa qualidade

## Uso

### Execução Básica (1 execução de cada)
```bash
python3 multi_objetivo/compare_cabling_strategies.py
```

### Múltiplas Execuções (para estatísticas)
```bash
python3 multi_objetivo/compare_cabling_strategies.py --runs 3
```

Isso executará cada estratégia 3 vezes e calculará estatísticas (média, desvio padrão, etc.).

## Saídas

O script gera os seguintes arquivos no diretório `comparison_results/`:

1. **raw_comparison_results.csv**: Resultados brutos de todas as execuções
2. **direct_comparison.csv**: Comparação direta (quando n_runs=1)
3. **comparison_summary.csv**: Estatísticas agregadas (quando n_runs>1)

### Métricas Comparadas

- **Tempo de Execução**: Tempo total em minutos
- **Melhor AEP**: Maior AEP líquido encontrado (GWh)
- **AEP Médio**: Média do AEP líquido na frente de Pareto (MWh)
- **Melhor Custo**: Menor custo encontrado (USD)
- **Custo Médio**: Média do custo na frente de Pareto (USD)
- **Número de Soluções**: Tamanho da frente de Pareto
- **N Grupos Médio**: Número médio de grupos de cabeamento
- **Perdas Médias**: Perdas Joule médias (MWh)

## Diretórios de Resultados

Cada execução salva seus resultados em:
- `pareto_front_results_v1_<run_id>/`: Resultados da V1
- `pareto_front_results_v3_<run_id>/`: Resultados da V3

## Exemplo de Saída

```
================================================================================
COMPARAÇÃO DIRETA
================================================================================
                    Métrica              V1              V3    Diferença (%)
0    Tempo de Execução (min)         45.23          15.67          -65.35%
1            Melhor AEP (GWh)       418.56         418.42           -0.03%
2          AEP Médio (MWh)     415234.12    415189.45           -0.01%
3         Melhor Custo (USD)     256686.00     257234.00           +0.21%
4        Custo Médio (USD)     275432.00     276189.00           +0.27%
5      Número de Soluções             423            421           -0.47%
6        N Grupos Médio              4.23           4.18           -1.18%
7    Perdas Médias (MWh)          234.56         235.12           +0.24%

================================================================================
ANÁLISE DE VELOCIDADE
================================================================================
V3 é 2.89x mais rápido que V1
Tempo V1: 45.23 min
Tempo V3: 15.67 min
Economia de tempo: 29.56 min
```

## Notas

- O script cria arquivos temporários durante a execução (removidos automaticamente)
- Cada execução pode levar várias horas dependendo dos parâmetros
- Recomenda-se executar com `--runs 1` primeiro para verificar se tudo está funcionando
- Para estatísticas robustas, execute pelo menos 3-5 vezes

## Interpretação dos Resultados

- **Diferença positiva (%)**: V3 é melhor que V1 nessa métrica
- **Diferença negativa (%)**: V1 é melhor que V3 nessa métrica
- **Speedup**: Quanto maior, mais rápido é V3 em relação a V1

Para um artigo científico, recomenda-se:
1. Executar múltiplas vezes (3-5) para estatísticas
2. Comparar não apenas médias, mas também desvios padrão
3. Realizar testes estatísticos (t-test) se necessário
4. Discutir trade-offs entre velocidade e qualidade

