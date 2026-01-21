# Plano de Ação: Otimização de Visualizações e Comparação com Literatura

## 📊 Resumo Executivo

**Objetivo:** Reduzir de 28 imagens PNG para 4 imagens melhoradas + tabelas comparativas, incluindo comparação com resultados da literatura.

**Imagens a manter e melhorar:**
1. ✅ Tempo de execução (melhorar gráfico)
2. ✅ Frentes de Pareto (manter)
3. ✅ Hipervolume (melhorar apresentação)
4. ✅ Layouts de exemplo (manter)

**Conversões:**
- Gráficos de escalabilidade (3) → Tabela comparativa expandida
- Análise qualitativa (4) → Tabela qualitativa expandida (já existe, melhorar)
- Outros gráficos → Tabelas ou texto

---

## 🎯 FASE 1: Análise e Coleta de Dados Comparativos

### 1.1 Autores Identificados no Documento para Comparação

**Autores de otimização integrada:**
- **Wang2019Integrated** - Otimização integrada layout + infraestrutura
- **Jin2025Integrated** - Co-design layout-subestação
- **Jin2019PowerLoss** - Modelos de perda de potência
- **Moon2015Optimal** - Otimização de posição de subestação
- **Nakhai2023Electrical** - Modelo de custo NREL para cabos

**Autores de roteamento de cabos:**
- **Alencar2026Flexible** - Roteamento flexível, heurísticas de reparo
- **Machado2024Hybrid** - Meta-heurísticas híbridas
- **Yuan2025ACO** - Ant Colony Optimization
- **Shen2023Ring** - Topologias em anel

**Autores de layout:**
- **Silva2025Layout** - Framework modular GA
- **Baker2019Best** - IEA Task 37 benchmark

### 1.2 Métricas para Comparação com Literatura

**Métricas quantitativas a extrair/comparar:**
- Net AEP (GWh) para 16, 36, 64 turbinas
- CAPEX de cabos (kUSD)
- Tempo de execução (minutos)
- Hypervolume (×10¹²)
- Número de soluções Pareto
- Taxa de redução de custo vs abordagens sequenciais

**Métricas qualitativas:**
- Distância subestação do centro (m)
- Comprimento total de cabos (km)
- Número de grupos de cabos
- Perdas elétricas (% do AEP bruto)

### 1.3 Busca de Dados na Literatura

**Estratégia de busca:**
1. Extrair valores de tabelas/gráficos dos artigos citados
2. Buscar artigos recentes (2020-2025) com benchmarks IEA Task 37
3. Comparar com trabalhos que usam NSGA-II em problemas similares
4. Identificar valores de referência para hypervolume em problemas multi-objetivo

**Artigos prioritários para extração de dados:**
- Wang et al. (2019) - Integrated optimization
- Jin et al. (2025) - Co-design framework
- Alencar et al. (2026) - Flexible routing
- Machado et al. (2024) - Hybrid meta-heuristics

---

## 📈 FASE 2: Melhorias nas 4 Imagens Principais

### 2.1 Tempo de Execução (Melhorar)

**Problemas atuais:**
- Pode estar pouco claro visualmente
- Falta comparação com literatura
- Escala pode não ser ideal

**Melhorias propostas:**
- ✅ Gráfico de barras agrupadas com erro padrão
- ✅ Escala logarítmica no eixo Y (se necessário)
- ✅ Incluir linha de referência com tempos reportados na literatura
- ✅ Cores consistentes (Baseline: azul, Proposed: verde, Sequential: laranja)
- ✅ Anotar percentuais de redução (31.6%, 41.2%, 45.2%)
- ✅ Adicionar tabela complementar com valores numéricos

**Formato:** PNG de alta qualidade ou TikZ/pgfplots (mais leve)

### 2.2 Frentes de Pareto (Manter)

**Status atual:** OK, mas pode melhorar

**Melhorias propostas:**
- ✅ Consolidar as 3 subfiguras em uma única figura mais compacta
- ✅ Adicionar linhas de referência (se houver dados da literatura)
- ✅ Melhorar legenda e anotações
- ✅ Destacar soluções knee-point
- ✅ Adicionar valores de hypervolume nas legendas

**Formato:** PNG ou PDF vetorial (mais leve que PNG)

### 2.3 Hipervolume (Melhorar Apresentação)

**Problemas atuais:**
- 3 gráficos separados podem ser consolidados
- Falta contexto comparativo
- Pode não mostrar claramente a vantagem do método proposto

**Melhorias propostas:**
- ✅ Consolidar em uma única figura com 3 subplots (16, 36, 64 turbinas)
- ✅ Adicionar linha de referência (Baseline médio ou valor da literatura)
- ✅ Mostrar bandas de confiança (média ± desvio padrão)
- ✅ Destacar transição entre Fase 1 e Fase 2 (se aplicável)
- ✅ Adicionar tabela complementar com valores finais de HV
- ✅ Incluir comparação com valores reportados na literatura (se disponível)

**Formato:** PNG ou TikZ/pgfplots

### 2.4 Layouts de Exemplo (Manter)

**Status atual:** OK, mas pode otimizar

**Melhorias propostas:**
- ✅ Manter as 9 subfiguras (3 estratégias × 3 escalas)
- ✅ Otimizar compressão das imagens PNG
- ✅ Adicionar anotações claras (AEP, CAPEX) em cada layout
- ✅ Garantir cores consistentes entre figuras
- ✅ Considerar reduzir para 6 layouts (2 escalas mais representativas) se necessário

**Formato:** PNG otimizado (compressão balanceada)

---

## 📋 FASE 3: Conversão para Tabelas

### 3.1 Tabela de Escalabilidade (Substituir 3 gráficos)

**Conteúdo:**
- Net AEP, CAPEX, Tempo de execução para 16, 36, 64 turbinas
- Valores médios ± desvio padrão
- Percentuais de melhoria (Proposed vs Baseline)
- Comparação com valores da literatura (coluna adicional)

**Estrutura proposta:**
```
| Escala | Método | Net AEP [GWh] | CAPEX [kUSD] | Tempo [min] | Ref. Literatura |
|--------|--------|---------------|--------------|-------------|------------------|
| 16     | Baseline | ... | ... | ... | [Wang2019] |
| 16     | Proposed | ... | ... | ... | - |
| 16     | Sequential | ... | ... | ... | - |
| ...    | ...    | ... | ... | ... | ... |
```

### 3.2 Tabela Qualitativa Expandida (Substituir 4 gráficos)

**Melhorias na tabela existente:**
- ✅ Adicionar coluna com valores da literatura (se disponível)
- ✅ Incluir percentuais de diferença entre métodos
- ✅ Adicionar notas explicativas sobre padrões observados
- ✅ Formatação melhorada para legibilidade

**Métricas a incluir:**
- Distância subestação do centro
- Comprimento total de cabos
- Número de grupos de cabos
- Perdas elétricas (%)
- Seções de cabos utilizadas

### 3.3 Tabela Comparativa com Literatura

**Nova tabela a criar:**
- Comparar resultados com trabalhos anteriores
- Métricas: Net AEP, CAPEX, Hypervolume, Tempo
- Incluir referências bibliográficas
- Destacar onde nosso método supera/igual a literatura

**Estrutura:**
```
| Autor | Ano | Método | Escala | Net AEP | CAPEX | HV | Tempo | Observações |
|-------|-----|--------|--------|---------|-------|----|----|----|------------|
| Wang  | 2019| NSGA-II| 16     | ...     | ...   | ...| ...| ...| ... |
| Jin   | 2025| Co-design| 36  | ...     | ...   | ...| ...| ...| ... |
| Este trabalho | 2025 | Proposed | 16 | ... | ... | ... | ... | ... |
```

---

## 🔍 FASE 4: Pesquisa e Extração de Dados da Literatura

### 4.1 Artigos Prioritários para Extração

1. **Wang et al. (2019)** - "Integrated optimization"
   - Buscar: Net AEP, CAPEX, tempo para escalas similares
   - Método: NSGA-II integrado

2. **Jin et al. (2025)** - "Integrated co-design"
   - Buscar: Resultados de co-design layout-subestação
   - Comparar redução de custo vs sequencial

3. **Alencar et al. (2026)** - "Flexible routing"
   - Buscar: Tempos de execução, qualidade de soluções
   - Comparar eficiência computacional

4. **Machado et al. (2024)** - "Hybrid meta-heuristics"
   - Buscar: Resultados em escalas similares
   - Comparar hypervolume e diversidade

5. **Moon et al. (2015)** - "Optimal substation placement"
   - Buscar: Redução de custo vs abordagens sequenciais
   - Comparar posicionamento de subestação

### 4.2 Estratégia de Busca Web

**Termos de busca:**
- "offshore wind farm layout optimization NSGA-II results"
- "IEA Task 37 benchmark results comparison"
- "wind farm cable routing optimization AEP CAPEX"
- "multi-objective wind farm optimization hypervolume comparison"

**Fontes:**
- Google Scholar
- IEEE Xplore
- ScienceDirect
- Artigos já citados no documento

### 4.3 Formato de Dados Extraídos

**Template para cada artigo:**
```
Artigo: [Autor] ([Ano])
Método: [Nome do método]
Escala: [16/36/64 turbinas]
Net AEP: [valor] GWh
CAPEX: [valor] kUSD
Tempo: [valor] minutos
Hypervolume: [valor] ×10¹²
Observações: [notas relevantes]
```

---

## 📝 FASE 5: Implementação no LaTeX

### 5.1 Remover Imagens

**Imagens a remover:**
- ❌ `scalability_net_aep.png` → Tabela
- ❌ `scalability_capex.png` → Tabela
- ❌ `scalability_execution_time.png` → Já temos gráfico melhorado
- ❌ `qualitative_substation_distance.png` → Tabela expandida
- ❌ `qualitative_cable_length.png` → Tabela expandida
- ❌ `qualitative_cable_groups.png` → Tabela expandida
- ❌ `qualitative_electrical_losses.png` → Tabela expandida

**Total removido:** 7 imagens PNG

### 5.2 Melhorar Imagens Mantidas

**Imagens a melhorar:**
- ✅ `execution_time_comparison.png` → Regenerar com melhorias
- ✅ `pareto_compacto_16.png, 36.png, 64.png` → Consolidar e melhorar
- ✅ `hv_16.png, 36.png, 64.png` → Consolidar e melhorar
- ✅ Layouts (9 imagens) → Otimizar compressão

### 5.3 Criar/Atualizar Tabelas

**Tabelas a criar/atualizar:**
1. `table_scalability_comparison.tex` - Nova (substitui 3 gráficos)
2. `table_qualitative_metrics.tex` - Expandir (já existe)
3. `table_literature_comparison.tex` - Nova (comparação com literatura)
4. `table_hypervolume_summary.tex` - Nova (resumo de HV)

### 5.4 Atualizar Texto do Documento

**Seções a atualizar:**
- Seção 4.1: Referenciar nova tabela de escalabilidade
- Seção 4.2: Manter referência às frentes de Pareto
- Seção 4.3: Referenciar tabela de resumo de HV + gráfico melhorado
- Seção 4.4: Manter gráfico de tempo melhorado
- Seção 4.5: Referenciar tabela qualitativa expandida
- Seção 4.6: Manter layouts de exemplo
- Nova subseção: "Comparison with Literature" (antes de Discussion)

---

## ✅ Checklist de Implementação

### Preparação
- [ ] Extrair dados dos artigos da literatura
- [ ] Organizar dados em formato estruturado
- [ ] Identificar valores de referência para comparação

### Melhorias de Imagens
- [ ] Regenerar gráfico de tempo de execução melhorado
- [ ] Consolidar e melhorar gráficos de hipervolume
- [ ] Melhorar apresentação das frentes de Pareto
- [ ] Otimizar compressão dos layouts de exemplo

### Criação de Tabelas
- [ ] Criar tabela de escalabilidade comparativa
- [ ] Expandir tabela qualitativa
- [ ] Criar tabela de comparação com literatura
- [ ] Criar tabela resumo de hypervolume

### Atualização do Documento
- [ ] Remover referências às imagens deletadas
- [ ] Adicionar referências às novas tabelas
- [ ] Atualizar texto com comparações da literatura
- [ ] Adicionar nova subseção de comparação
- [ ] Verificar compilação LaTeX

### Validação
- [ ] Verificar que todas as informações estão preservadas
- [ ] Confirmar que o documento compila sem erros
- [ ] Validar que o tamanho do PDF foi reduzido
- [ ] Revisar qualidade visual das imagens mantidas

---

## 📊 Resultado Esperado

**Antes:**
- 28 imagens PNG
- 2 tabelas
- Tamanho do PDF: [a verificar]

**Depois:**
- 4 imagens melhoradas (tempo, pareto, HV, layouts)
- 5-6 tabelas (incluindo comparação com literatura)
- Tamanho do PDF: Reduzido significativamente
- Qualidade: Melhorada com comparações contextuais

---

## 🚀 Próximos Passos Imediatos

1. **Começar pela pesquisa de literatura** - Extrair dados comparativos
2. **Criar tabela de comparação com literatura** - Base para discussão
3. **Melhorar gráfico de tempo de execução** - Primeira imagem a otimizar
4. **Consolidar gráficos de hipervolume** - Segunda prioridade
5. **Atualizar seções do documento** - Integrar novas tabelas

---

**Data de criação:** [Hoje]
**Status:** Pronto para implementação
**Prioridade:** Alta
