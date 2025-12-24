"""
Grid Search Otimizado
Baseado na análise dos padrões cíclicos identificados
"""

# INDPB: 0.05 a 0.30 com granularidade 0.025
# Foco na zona ótima identificada (0.05-0.20 são mais frequentes nos picos)
indpb_values = [i / 1000.0 for i in range(50, 301, 25)]  # 0.05 a 0.30, passo 0.025
# Resultado: [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]

# MUTPB: 0.05 a 1.00 com granularidade 0.05
# Granularidade média pois MUTPB tem menor impacto direto
mutpb_values = [i / 100.0 for i in range(5, 101, 5)]  # 0.05 a 1.00, passo 0.05
# Resultado: [0.05, 0.10, 0.15, ..., 0.95, 1.00]

# CXPB: 0.55 a 1.00 com granularidade 0.01 (CRÍTICO - padrão cíclico)
# Zona ótima identificada: 0.60-0.95 (média 0.866 no top 10%)
# Granularidade fina necessária para capturar padrão cíclico
cxpb_values = [i / 100.0 for i in range(55, 101, 1)]  # 0.55 a 1.00, passo 0.01
# Resultado: [0.55, 0.56, 0.57, ..., 0.99, 1.00]

# Total de combinações
total = len(indpb_values) * len(mutpb_values) * len(cxpb_values)
print(f"Grid Search Otimizado:")
print(f"  INDPB: {len(indpb_values)} valores → {indpb_values}")
print(f"  MUTPB: {len(mutpb_values)} valores → Range: [{min(mutpb_values):.2f}, {max(mutpb_values):.2f}]")
print(f"  CXPB: {len(cxpb_values)} valores → Range: [{min(cxpb_values):.2f}, {max(cxpb_values):.2f}]")
print(f"  Total: {total:,} combinações")


