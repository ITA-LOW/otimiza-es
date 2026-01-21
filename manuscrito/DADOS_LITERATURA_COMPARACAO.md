# Dados Comparativos da Literatura - Estrutura de Coleta

## 📊 Dados Atuais do Nosso Trabalho

### Tempos de Execução (média ± desvio padrão)

| Escala | Método | Tempo [min] | Observações |
|--------|--------|-------------|-------------|
| 16 turbinas | Baseline | 10.1 ± 0.1 | NSGA-II fase única |
| 16 turbinas | Proposed | 6.9 ± 0.2 | Hierárquico 2 fases |
| 16 turbinas | Sequential | 5.2 ± 0.1 | Sequencial 2 fases |
| 36 turbinas | Baseline | 83.7 ± 2.4 | - |
| 36 turbinas | Proposed | 49.1 ± 1.0 | - |
| 36 turbinas | Sequential | 43.3 ± 1.3 | - |
| 64 turbinas | Baseline | 165.1 ± 54.3 | Alta variabilidade |
| 64 turbinas | Proposed | 90.5 ± 28.4 | - |
| 64 turbinas | Sequential | 84.2 ± 26.9 | - |

### Net AEP (GWh)

| Escala | Método | Net AEP [GWh] | Observações |
|--------|--------|----------------|-------------|
| 16 | Baseline | 404.34 ± 8.44 | - |
| 16 | Proposed | 457.89 ± 2.39 | - |
| 16 | Sequential | 462.99 ± 0.43 | - |
| 36 | Baseline | 835.10 ± 7.30 | - |
| 36 | Proposed | 941.00 ± 5.77 | - |
| 36 | Sequential | 944.62 ± 3.16 | - |
| 64 | Baseline | 1329.29 ± 17.91 | - |
| 64 | Proposed | 1447.18 ± 21.75 | - |
| 64 | Sequential | 1503.81 ± 12.27 | - |

### CAPEX (kUSD)

| Escala | Método | CAPEX [kUSD] | Observações |
|--------|--------|--------------|-------------|
| 16 | Baseline | 190 ± 22 | - |
| 16 | Proposed | 657 ± 91 | - |
| 16 | Sequential | 731 ± 108 | - |
| 36 | Baseline | 875 ± 93 | - |
| 36 | Proposed | 1976 ± 245 | - |
| 36 | Sequential | 1838 ± 327 | - |
| 64 | Baseline | 2416 ± 37 | - |
| 64 | Proposed | 3302 ± 292 | - |
| 64 | Sequential | 3325 ± 200 | - |

### Hypervolume (×10¹²)

| Escala | Método | HV [×10¹²] | Observações |
|--------|--------|------------|-------------|
| 16 | Baseline | 8.52 ± 0.12 | - |
| 16 | Proposed | 8.99 ± 0.03 | - |
| 16 | Sequential | 8.92 ± 0.05 | - |
| 36 | Baseline | 16.34 ± 0.21 | - |
| 36 | Proposed | 17.30 ± 0.28 | - |
| 36 | Sequential | 12.85 ± 7.62 | Alta variabilidade |
| 64 | Baseline | 23.57 ± 0.33 | Problemas de cálculo |
| 64 | Proposed | 22.50 ± 3.55 | ~50% runs válidos |
| 64 | Sequential | 12.50 ± 12.82 | Alta variabilidade |

---

## 🔍 Template para Extração de Dados da Literatura

### Artigo: [Autor] ([Ano])
**Título:** [Título completo]
**DOI/Link:** [URL]
**Método/Paradigma:** [Nome do método]
**Tipo de Problema:** [WFLO / WFCRP / Integrated Co-Design / Outro]

#### Configuração Experimental
- **Escala testada:** [16/36/64 turbinas ou similar]
- **Hardware:** [CPU, RAM, se disponível]
- **População:** [Tamanho da população]
- **Gerações:** [Número de gerações]
- **Número de execuções:** [Runs independentes]

#### Resultados Quantitativos
- **Net AEP:** [valor] GWh (escala: [16/36/64])
- **CAPEX:** [valor] kUSD (escala: [16/36/64])
- **Tempo de execução:** [valor] minutos/horas (escala: [16/36/64])
- **Hypervolume:** [valor] ×10¹² (escala: [16/36/64])
- **Número de soluções Pareto:** [valor] (escala: [16/36/64])

#### Resultados Qualitativos
- **Distância subestação:** [valor] m
- **Comprimento de cabos:** [valor] km
- **Número de grupos:** [valor]
- **Perdas elétricas:** [valor] % do AEP bruto

#### Observações
- [Notas sobre limitações, diferenças metodológicas, etc.]

---

## 📋 Artigos Prioritários para Extração

### 1. Wang et al. (2019) - "Integrated optimization"
**Status:** ⏳ A extrair
**Prioridade:** Alta
**Razão:** Trabalho seminal em otimização integrada layout + infraestrutura

**Dados a buscar:**
- [ ] Net AEP para escalas similares
- [ ] CAPEX reportado
- [ ] Tempo de execução
- [ ] Métricas de qualidade (HV, IGD, etc.)

---

### 2. Jin et al. (2025) - "Integrated co-design"
**Status:** ⏳ A extrair
**Prioridade:** Alta
**Razão:** Trabalho recente em co-design layout-subestação

**Dados a buscar:**
- [ ] Comparação com abordagens sequenciais
- [ ] Redução de custo reportada
- [ ] Tempo de execução
- [ ] Métricas de trade-off

---

### 3. Jin et al. (2019) - "Power loss"
**Status:** ⏳ A extrair
**Prioridade:** Média
**Razão:** Modelos de perda de potência, redução de custo de 3.14%

**Dados a buscar:**
- [ ] Redução de custo vs sequencial
- [ ] Impacto de perdas elétricas no AEP
- [ ] Tempo de execução (se disponível)

---

### 4. Alencar et al. (2026) - "Flexible routing"
**Status:** ⏳ A extrair
**Prioridade:** Alta
**Razão:** Roteamento flexível, menciona redução computacional de 60x

**Dados a buscar:**
- [ ] Tempo de execução
- [ ] Eficiência computacional
- [ ] Qualidade de soluções
- [ ] Comparação com métodos anteriores

---

### 5. Machado et al. (2024) - "Hybrid meta-heuristics"
**Status:** ⏳ A extrair
**Prioridade:** Média
**Razão:** Meta-heurísticas híbridas para escalabilidade

**Dados a buscar:**
- [ ] Resultados em escalas similares
- [ ] Hypervolume e diversidade
- [ ] Tempo de execução
- [ ] Escalabilidade reportada

---

### 6. Moon et al. (2015) - "Optimal substation placement"
**Status:** ⏳ A extrair
**Prioridade:** Média
**Razão:** Otimização de posição de subestação, redução de custo vs sequencial

**Dados a buscar:**
- [ ] Redução de custo vs abordagens sequenciais
- [ ] Impacto no posicionamento
- [ ] Tempo de execução (se disponível)

---

### 7. Yuan & Tang (2025) - "ACO"
**Status:** ⏳ A extrair
**Prioridade:** Baixa
**Razão:** Ant Colony Optimization, redução de 3.82% de custo

**Dados a buscar:**
- [ ] Redução de custo reportada
- [ ] Tempo de execução
- [ ] Comparação com outros métodos

---

### 8. Shen et al. (2023) - "Ring topology"
**Status:** ⏳ A extrair
**Prioridade:** Baixa
**Razão:** Topologias em anel, redução de 4-8% de custo

**Dados a buscar:**
- [ ] Redução de custo
- [ ] Comparação com topologias tradicionais
- [ ] Tempo de execução (se disponível)

---

### 9. Nakhai et al. (2023) - "Electrical cost model"
**Status:** ⏳ A extrair
**Prioridade:** Baixa
**Razão:** Modelo de custo NREL, não resultados de otimização

**Dados a buscar:**
- [ ] Validação do modelo de custo
- [ ] Comparação com outros modelos (se houver)

---

### 10. IEA Task 37 Benchmark
**Status:** ⏳ A extrair
**Prioridade:** Alta
**Razão:** Benchmark padrão, resultados de referência

**Dados a buscar:**
- [ ] Resultados de referência para 16, 36, 64 turbinas
- [ ] AEP bruto esperado
- [ ] Métricas de qualidade (se disponíveis)

---

## 📊 Tabela Comparativa Consolidada (A Preencher)

| Autor | Ano | Método | Escala | Net AEP [GWh] | CAPEX [kUSD] | Tempo [min] | HV [×10¹²] | Observações |
|-------|-----|--------|--------|---------------|--------------|-------------|------------|-------------|
| Este trabalho | 2025 | Baseline | 16 | 404.34 | 190 | 10.1 | 8.52 | - |
| Este trabalho | 2025 | Proposed | 16 | 457.89 | 657 | 6.9 | 8.99 | - |
| Este trabalho | 2025 | Sequential | 16 | 462.99 | 731 | 5.2 | 8.92 | - |
| Este trabalho | 2025 | Baseline | 36 | 835.10 | 875 | 83.7 | 16.34 | - |
| Este trabalho | 2025 | Proposed | 36 | 941.00 | 1976 | 49.1 | 17.30 | - |
| Este trabalho | 2025 | Sequential | 36 | 944.62 | 1838 | 43.3 | 12.85 | - |
| Este trabalho | 2025 | Baseline | 64 | 1329.29 | 2416 | 165.1 | 23.57 | Problemas HV |
| Este trabalho | 2025 | Proposed | 64 | 1447.18 | 3302 | 90.5 | 22.50 | ~50% válidos |
| Este trabalho | 2025 | Sequential | 64 | 1503.81 | 3325 | 84.2 | 12.50 | - |
| [Wang] | 2019 | Integrated | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| [Jin] | 2025 | Co-design | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| [Jin] | 2019 | Power loss | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | Redução 3.14% |
| [Alencar] | 2026 | Flexible | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | 60x redução |
| [Moon] | 2015 | Substation | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 🎯 Próximos Passos

1. **Buscar acesso aos artigos** - Verificar disponibilidade via biblioteca/DOI
2. **Extrair dados de tabelas/gráficos** - Usar ferramentas de extração se necessário
3. **Normalizar dados** - Ajustar para escalas comparáveis
4. **Validar comparações** - Verificar se condições experimentais são similares
5. **Criar tabela LaTeX** - Formatar para inserção no documento

---

## 📝 Notas de Busca

### Buscas Realizadas
- ✅ Busca geral: "wind farm layout optimization NSGA-II results"
- ✅ Busca: "offshore wind farm cable routing optimization results"
- ✅ Busca: "IEA Task 37 benchmark results"
- ✅ Busca: "multi-objective wind farm optimization hypervolume"
- ✅ Busca específica: "Wang 2019 integrated optimization"
- ✅ Busca específica: "Jin 2025 integrated co-design"
- ✅ Busca específica: "Jin 2019 power loss 3.14%"
- ✅ Busca específica: "Alencar 2026 flexible routing 60-fold"

### Dificuldades Encontradas
- Muitos artigos não reportam tempos de execução explicitamente
- Dados podem estar em gráficos que requerem extração manual
- Diferenças em hardware dificultam comparação direta
- Alguns artigos usam escalas diferentes (número de turbinas)
- Artigos específicos podem não estar disponíveis publicamente
- Dados podem estar em tabelas suplementares não indexadas

### Estratégias Alternativas
- Comparar percentuais de melhoria ao invés de valores absolutos
- Normalizar por número de turbinas ou dimensão do problema
- Usar métricas relativas (ex: tempo relativo ao Baseline)
- Focar em tendências de escalabilidade ao invés de valores absolutos
- Usar dados de artigos gerais sobre MOEAs como referência
- Comparar com benchmarks padrão (DTLZ, ZDT) quando aplicável

### Artigos Genéricos Encontrados (Como Referência)

**Artigos sobre tempo de execução em MOEAs:**
1. "Evolutionary Multi-Objective Energy Production Optimization" (MDPI, 2025)
   - Compara NSGA-II, L-NSGA-II, MOEA/D
   - Reporta tempos de execução para 100 gerações, população 20
   - Problema: 3 objetivos

2. "Efficient workflow scheduling using IMOMA" (Nature, 2025)
   - Compara IMOMA vs NSGA-II, SPEA-II, MOPSO
   - Tempos médios para 50, 75, 100, 200 tarefas
   - Runtime competitivo reportado

3. "Novel hybrid evolutionary algorithm" (Nature, 2025)
   - Tempos de execução para instâncias grandes
   - Exemplo: ~12,600 segundos para 1000 cidades

4. "Metaheuristics for RCPSP_TDRC" (ScienceDirect, 2023)
   - Limite de 2 minutos por execução
   - Compara NSGA-II, PESA-II, MOEA/D, IBEA

### Próximas Ações Recomendadas

1. **Acesso direto aos artigos:**
   - Verificar acesso via biblioteca universitária
   - Buscar PDFs dos artigos específicos (Wang 2019, Jin 2025, etc.)
   - Extrair dados de tabelas e gráficos manualmente

2. **Extrair dados de gráficos:**
   - Usar ferramentas como WebPlotDigitizer para extrair dados de gráficos
   - Converter valores aproximados quando necessário

3. **Contato com autores:**
   - Considerar contatar autores para dados adicionais
   - Solicitar dados brutos se disponíveis

4. **Usar dados genéricos:**
   - Comparar com artigos gerais sobre NSGA-II, MOEA/D
   - Usar como referência de escalabilidade geral

---

**Última atualização:** 2025-01-XX
**Status:** Em progresso - Estrutura criada, busca inicial realizada
**Próximo passo:** Acesso direto aos PDFs dos artigos específicos
