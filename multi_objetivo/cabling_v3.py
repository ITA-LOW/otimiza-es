import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import math
import json
# Nota: Não usa KMeans - usa agrupamento angular determinístico

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================

def calculate_distance(x1, y1, x2, y2):
    """
    Calcula distância euclidiana entre dois pontos no plano.
    Usa math.hypot para evitar overflow numérico.
    """
    return math.hypot(x2 - x1, y2 - y1)

# ======================================================
# CLASSES ELÉTRICAS (INALTERADAS)
# ======================================================

class Cable:
    """
    Representa um segmento de cabo elétrico no sistema de cabeamento.
    
    Cada cabo transporta potência acumulada (Pn) de todas as turbinas a montante
    e precisa ter seção transversal suficiente para suportar a corrente resultante.
    """
    # Fator de potência típico para turbinas eólicas com conversores modernos
    # Valores típicos: 0.9-0.95 (carga indutiva devido aos conversores)
    # Usamos 0.95 como valor conservador, representando sistemas modernos com
    # controle ativo de fator de potência (conforme prática em parques eólicos offshore)
    POWER_FACTOR = 0.95  # cos(φ) - adimensional
    
    # Tabela de resistência elétrica por seção transversal (Ω/km)
    # Valores típicos para cabos submarinos de média tensão
    SECTION_TABLE = {
        50: 0.49, 70: 0.34, 95: 0.25, 120: 0.20,
        150: 0.16, 185: 0.13, 240: 0.10,
    }

    def __init__(self, lc, Vn, Pn):
        """
        Inicializa um cabo com parâmetros elétricos básicos.
        
        Args:
            lc: Comprimento do cabo (metros)
            Vn: Tensão nominal do sistema (Volts) - típico: 33kV
            Pn: Potência acumulada transportada (Watts) - soma das potências das turbinas a montante
        """
        self.lc = lc  # Comprimento do cabo (metros)
        self.Vn = Vn  # Tensão nominal (Volts)
        self.Pn = Pn  # Potência acumulada transportada (Watts)
        self.dI = 2.3  # Densidade de corrente máxima permitida (A/mm²) - padrão industrial
        
        # Calcula corrente elétrica: I = P / (√3 * V * cos(φ)) para sistema trifásico
        # Fórmula completa: P = √3 * V * I * cos(φ) => I = P / (√3 * V * cos(φ))
        # Referência: Fórmula padrão de engenharia elétrica para sistemas trifásicos
        # (ver, por exemplo, materiais didáticos de universidades e normas IEC)
        # √3 é o fator de correção para sistemas trifásicos (tensão de linha)
        # cos(φ) = POWER_FACTOR considera o fator de potência típico de turbinas eólicas
        # com conversores modernos (valores típicos: 0.9-0.95, usando 0.95 como conservador)
        self.I = self.Pn / (math.sqrt(3) * self.Vn * Cable.POWER_FACTOR)
        
        # Área mínima necessária para suportar a corrente (mm²)
        # Baseado na densidade de corrente máxima permitida
        self.A_continuous = self.I / self.dI
        
        # Valores que serão atribuídos quando a seção for escolhida
        self.A = None  # Seção transversal escolhida (mm²)
        self.R_km = None  # Resistência por quilômetro (Ω/km)
        self.R = None  # Resistência total do cabo (Ω)
        self.Pj = None  # Perdas Joule (Watts) - Pj = 3 * I² * R (trifásico)
        self.C = 0  # Custo por metro (USD/m)
        self.Ctot = 0  # Custo total do cabo (USD)

    def assign_section(self, section):
        """
        Atribui uma seção transversal comercial ao cabo e calcula propriedades elétricas.
        
        Args:
            section: Seção transversal em mm² (deve estar em SECTION_TABLE)
        
        Processo:
        1. Atribui seção e busca resistência correspondente na tabela
        2. Calcula resistência total: R = R_km * (comprimento em km)
        3. Calcula perdas Joule: Pj = 3 * I² * R (fator 3 para sistema trifásico)
        """
        self.A = section
        self.R_km = self.SECTION_TABLE[section]  # Resistência por km da tabela
        self.R = self.R_km * (self.lc / 1000)  # Resistência total (comprimento em km)
        # Perdas Joule em sistema trifásico: Pj = 3 * I² * R
        self.Pj = 3 * (self.I ** 2) * self.R


class Turbine:
    """
    Representa uma turbina eólica no parque.
    Armazena potência nominal e posição espacial.
    """
    def __init__(self, Pt, x, y):
        """
        Inicializa uma turbina.
        
        Args:
            Pt: Potência nominal da turbina (Watts) - típico: 3.35 MW
            x, y: Coordenadas espaciais da turbina (metros)
        """
        self.P = Pt  # Potência nominal (Watts)
        self.x = x  # Coordenada X (metros)
        self.y = y  # Coordenada Y (metros)


class Plant:
        
    # Multiplicador NREL (0.3476 USD/m por mm2) * 3 condutores
    NREL_UNIT_COST = 0.3476 * 3  # 1.0428

    INDUSTRIAL_CABLE_COSTS = {
        50: 52.14,   # 50 * 1.0428
        70: 72.99,   # 70 * 1.0428
        95: 99.07,   # 95 * 1.0428
        120: 125.14, # 120 * 1.0428
        150: 156.42, # 150 * 1.0428
        185: 192.92, # 185 * 1.0428
        240: 250.27,  # 240 * 1.0428
        300: 312.84,
        400: 400*1.0428,
        500: 500*1.0428,
        630: 630*1.0428,
        800: 800*1.0428
    }

    """
    INDUSTRIAL_CABLE_COSTS Justification:
    These values represent the discrete instantiation of the NREL marine-energy cost model 
    (Nakhai et al., 2023). The baseline linear model defines costs as:
    Cost [USD/m] = 0.3476 * CSA * N_cond.

    For a three-phase inter-array system (N_cond = 3), the linear trend is mapped to 
    commercially available cross-sections (50 to 240 mm2) to simulate real-world 
    procurement constraints within the NSGA-II optimization loop. This ensures 
    Objective f2 (CAPEX) reflects discrete industrial steps rather than continuous 
    approximations.
    """



    def __init__(self, Vn, Tr, paths):
        """
        Inicializa a planta de cabeamento e calcula todas as propriedades.
        
        Args:
            Vn: Tensão nominal do sistema (Volts)
            Tr: Lista de objetos Turbine (uma por turbina + subestação)
            paths: Lista de caminhos de cabeamento (cada path é lista de índices)
        
        Processo de inicialização (executado automaticamente):
        1. lay_cables(): Cria objetos Cable para cada segmento
        2. uniform_section(): Escolhe seção única para todos os cabos (baseado no maior A_continuous)
        3. calculate_losses(): Calcula perdas Joule totais
        4. calculate_cost(): Calcula custo total de cabeamento
        """
        self.Vn = Vn  # Tensão nominal (Volts)
        self.Tr = Tr  # Lista de turbinas (objetos Turbine)
        self.paths = paths  # Lista de caminhos de cabeamento
        self.Cb = []  # Lista de listas de cabos (um por path)
        self.cables_flat = []  # Lista plana de todos os cabos (para cálculos)
        self.Pjtot = 0  # Perdas Joule totais (Watts)
        self.Ctot = 0  # Custo total de cabeamento (USD)

        # Executa sequência de cálculos
        self.lay_cables()  # Cria objetos Cable para cada segmento
        self.uniform_section()  # Escolhe seção única para todos os cabos
        self.calculate_losses()  # Calcula perdas Joule totais
        self.calculate_cost()  # Calcula custo total

    def lay_cables(self):
        """
        Cria objetos Cable para cada segmento de cabeamento.
        
        Para cada path (caminho de cabeamento):
        - Percorre segmentos consecutivos do path
        - Acumula potência das turbinas a montante (Pacc)
        - Calcula comprimento de cada segmento
        - Cria objeto Cable com potência acumulada
        
        A potência acumulada é crítica: cada segmento transporta a soma das
        potências de todas as turbinas conectadas a montante dele no path.
        Isso determina a corrente elétrica e, consequentemente, a seção necessária.
        """
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Pacc = 0  # Potência acumulada (soma das potências a montante)
            
            # Percorre segmentos consecutivos do path
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]  # Índices dos pontos conectados
                
                # Acumula potência da turbina atual (a montante)
                # A potência acumulada aumenta conforme caminhamos para a subestação
                Pacc += self.Tr[a].P
                
                # Calcula comprimento do segmento (distância euclidiana)
                L = calculate_distance(
                    self.Tr[a].x, self.Tr[a].y,
                    self.Tr[b].x, self.Tr[b].y
                )
                
                # Cria cabo com potência acumulada até este ponto
                cable_path.append(Cable(L, self.Vn, Pacc))
            
            self.Cb.append(cable_path)

        # Cria lista plana de todos os cabos (útil para cálculos agregados)
        self.cables_flat = [c for p in self.Cb for c in p]

    def uniform_section(self):
        """
        Escolhe uma seção transversal única para TODOS os cabos da planta.
        
        Estratégia: Seção uniforme (todos os cabos têm a mesma bitola)
        - Vantagem: Simplifica instalação e reduz custos de estoque
        - Desvantagem: Pode ser superdimensionado em alguns segmentos
        
        Processo:
        1. Encontra o maior A_continuous necessário (maior corrente)
        2. Escolhe a menor seção comercial que atende este requisito
        3. Aplica esta seção a todos os cabos (uniformização)
        
        Isso garante que todos os cabos suportem a corrente máxima necessária,
        mesmo que alguns segmentos possam usar seções menores teoricamente.
        """
        # Encontra a maior área mínima necessária entre todos os cabos
        # Este é o cabo que precisa da maior seção (maior corrente)
        Amax = max(c.A_continuous for c in self.cables_flat)
        
        # Inicializa com a maior seção disponível (fallback)
        chosen = max(Cable.SECTION_TABLE)
        
        # Escolhe a menor seção comercial que atende o requisito
        # Percorre seções em ordem crescente e escolhe a primeira >= Amax
        for sec in sorted(Cable.SECTION_TABLE):
            if sec >= Amax:
                chosen = sec
                break
        
        # Aplica a seção escolhida a TODOS os cabos (uniformização)
        for c in self.cables_flat:
            c.assign_section(chosen)

    def calculate_losses(self):
        """
        Calcula perdas Joule totais da planta.
        
        Perdas Joule: Pj = 3 * I² * R (sistema trifásico)
        - São perdas por efeito Joule (aquecimento) nos cabos
        - Dependem da corrente (I) e resistência (R)
        - Reduzem a energia entregue (AEP líquido = AEP bruto - perdas)
        
        A soma de todas as perdas individuais dá a perda total do sistema.
        """
        self.Pjtot = sum(c.Pj for c in self.cables_flat)

    def calculate_cost(self):
        """
        Calcula custo total de cabeamento usando tabela de custos industriais.
        
        Estratégia: Custo por metro depende apenas da seção do cabo
        - Todos os cabos têm a mesma seção (uniform_section)
        - Custo por metro é obtido da tabela INDUSTRIAL_CABLE_COSTS
        - Custo total = soma de (comprimento * custo_por_metro) para cada cabo
        
        A tabela de custos reflete valores reais de mercado para cabos submarinos
        de média tensão, baseada no modelo NREL (Nakhai et al., 2023).
        """
        # Obtém seção do primeiro cabo (todos têm a mesma seção)
        sec = self.cables_flat[0].A
        
        # Busca custo por metro na tabela de custos industriais
        custo_m = self.INDUSTRIAL_CABLE_COSTS[sec]
        
        # Calcula custo total: soma de (comprimento * custo_por_metro) para cada cabo
        self.Ctot = 0
        for c in self.cables_flat:
            c.C = custo_m  # Custo por metro (USD/m)
            c.Ctot = c.lc * custo_m  # Custo total do cabo (USD)
            self.Ctot += c.Ctot  # Acumula custo total da planta

    def get_max_calculated_section(self):
        return self.cables_flat[0].A

# ======================================================
# AGRUPAMENTO ANGULAR (SEM BALANCEAMENTO)
# ======================================================

def agrupar_por_setor_angular(coords, sub, n_grupos):
    """
    Agrupa turbinas por setores angulares contíguos em relação à subestação.
    
    Esta estratégia de agrupamento é determinística e evita cruzamentos:
    - Divide o espaço ao redor da subestação em n_grupos fatias angulares
    - Cada fatia contém turbinas cujos ângulos estão em um intervalo contíguo
    - Garante que grupos não se sobreponham (evita cruzamentos de cabos)
    
    Args:
        coords: Array de coordenadas (turbinas + subestação)
        sub: Índice da subestação no array coords
        n_grupos: Número de grupos desejado (fatias angulares)
    
    Returns:
        Lista de grupos, onde cada grupo é uma lista de índices de turbinas
    """
    # Calcula vetores da subestação para cada ponto (turbinas + subestação)
    # v[i] = vetor do ponto i em relação à subestação
    v = coords - coords[sub]
    
    # Calcula ângulos polares de todos os pontos em relação à subestação
    # arctan2(y, x) retorna ângulo em [-π, π] (radianos)
    # Ângulo 0 = direita, π/2 = cima, -π/2 = baixo, ±π = esquerda
    ang = np.arctan2(v[:, 1], v[:, 0])
    
    # Remove a subestação do cálculo (seu ângulo não importa para agrupamento)
    # Cria array de índices excluindo a subestação
    indices_turbinas = np.array([i for i in range(len(coords)) if i != sub])
    angulos_turbinas = ang[indices_turbinas]
    
    # Ordena índices por ângulo crescente (ordem angular no sentido anti-horário)
    # Isso garante que turbinas adjacentes angularmente fiquem próximas na lista
    idx_ordenado = indices_turbinas[np.argsort(angulos_turbinas)]
    
    # Divide em n_grupos fatias angulares contíguas usando array_split
    # Cada fatia contém aproximadamente o mesmo número de turbinas
    # Exemplo: 16 turbinas, 4 grupos = 4 turbinas por grupo (fatias de ~90°)
    grupos = np.array_split(idx_ordenado, n_grupos)
    
    # Converte para listas e remove grupos vazios (caso n_grupos > número de turbinas)
    grupos_filtrados = [list(g) for g in grupos if len(g) > 0]
    
    return grupos_filtrados

# ======================================================
# FUNÇÃO PRINCIPAL
# ======================================================

def analisar_layout_completo(coords, sub, n_grupos=15, Vn=33e3, P_turb=3.35e6):
    """
    Analisa layout completo com agrupamento angular estrito.
    
    Esta função é o ponto de entrada principal para análise de cabeamento.
    Processo completo:
    1. Agrupa turbinas por setores angulares (evita cruzamentos)
    2. Cria paths ordenados radialmente dentro de cada grupo
    3. Calcula propriedades elétricas (corrente, resistência, perdas)
    4. Calcula custos de cabeamento
    5. Retorna resultados agregados
    
    Args:
        coords: Array de coordenadas (turbinas + subestação) - shape (N, 2)
        sub: Índice da subestação no array coords
        n_grupos: Número de grupos de cabeamento (strings) - default: 15
        Vn: Tensão nominal do sistema (Volts) - default: 33kV
        P_turb: Potência nominal de cada turbina (Watts) - default: 3.35 MW
    
    Returns:
        planta: Objeto Plant com todos os detalhes de cabeamento
        resultados: Dicionário com métricas agregadas (custo, perdas, comprimento, etc.)
    """
    N = len(coords)
    
    # PASSO 1: Agrupa turbinas por setores angulares contíguos
    # Cada grupo é uma fatia angular ao redor da subestação
    # Isso garante que grupos não se cruzem (evita cruzamentos de cabos)
    grupos = agrupar_por_setor_angular(coords, sub, n_grupos)
    
    # PASSO 2: Cria paths de cabeamento dentro de cada grupo
    # Dentro de cada grupo, ordena turbinas por distância radial (mais longe primeiro)
    # Isso cria ordem estrita: turbina mais distante -> turbina mais próxima -> subestação
    paths = []
    for g in grupos:
        if len(g) == 0:
            continue
        
        # Calcula distâncias de cada turbina do grupo à subestação
        distancias = [np.linalg.norm(coords[t] - coords[sub]) for t in g]
        
        # Ordena por distância decrescente (mais longe primeiro)
        # Estratégia: Conecta turbinas em cascata, da mais distante para a mais próxima
        # Isso minimiza comprimento total de cabo dentro de cada grupo
        ordenado = [t for _, t in sorted(zip(distancias, g), reverse=True)]
        
        # Adiciona subestação no final do path (ponto de coleta)
        paths.append(ordenado + [sub])

    # PASSO 3: Cria objetos Turbine e inicializa Plant
    # A Plant automaticamente calcula: cabos, seções, perdas e custos
    # NOTA: A subestação (índice 'sub') também recebe P_turb, mas isso não afeta
    # o cálculo porque a subestação está sempre no final dos paths e sua potência
    # nunca é acumulada (o loop em lay_cables vai até len(path)-1).
    # Idealmente, a subestação deveria ter P=0, mas a implementação atual funciona
    # corretamente devido à ordem dos elementos no path.
    turbinas = [Turbine(P_turb, x, y) for x, y in coords]
    planta = Plant(Vn, turbinas, paths)

    # PASSO 4: Converte custos e calcula métricas finais
    COT_DOLAR = 0.1722  # Cotação USD/BRL usada nas versões anteriores

    # Comprimento total de cabo (soma de todos os segmentos)
    comprimento_total = sum(c.lc for c in planta.cables_flat)
    
    # Perdas anuais: perdas Joule * horas por ano (8760) / conversão para MWh
    # 8760 = horas em um ano (365 * 24)
    perda_anual_mwh = planta.Pjtot * 8760 / 1e6
    
    # Perdas totais em kW (para compatibilidade com versões anteriores)
    perda_total_kw = planta.Pjtot / 1e3
    
    # Seção do cabo escolhida (todos têm a mesma seção)
    secao = planta.get_max_calculated_section()
    
    # Custo total em BRL e conversão para USD
    custo_total = planta.Ctot  # Custo em BRL (da tabela INDUSTRIAL_CABLE_COSTS)
    custo_total_usd = custo_total * COT_DOLAR  # Conversão para USD

    # Dicionário de resultados (compatível com versões anteriores)
    resultados = {
        # --- chaves históricas (V1 / V2) ---
        "custo_total_usd": custo_total_usd,  # Custo total em USD
        "comprimento_total_m": comprimento_total,  # Comprimento total em metros
        "perda_total_kw": perda_total_kw,  # Perdas totais em kW
        "perda_anual_mwh": perda_anual_mwh,  # Perdas anuais em MWh
        "secao_cabo_mm2": secao,  # Seção do cabo em mm²

        # --- chaves novas (V3, mantidas) ---
        "custo_total": custo_total,  # Custo total em BRL (antes da conversão)
        "secao_mm2": secao  # Seção do cabo (alias para compatibilidade)
    }

    return planta, resultados

# ======================================================
# VISUALIZAÇÃO
# ======================================================

def plotar(planta, coords, sub):
    """
    Visualiza o layout de cabeamento com paths coloridos.
    
    Função auxiliar para visualização rápida durante desenvolvimento/teste.
    Plota cada path (string) com cor diferente e destaca a subestação.
    
    Args:
        planta: Objeto Plant com paths de cabeamento
        coords: Array de coordenadas (turbinas + subestação)
        sub: Índice da subestação
    """
    plt.figure(figsize=(10,10))
    
    # Plota cada path (string) com cor diferente
    for i, p in enumerate(planta.paths):
        x = [coords[k,0] for k in p]  # Coordenadas X do path
        y = [coords[k,1] for k in p]  # Coordenadas Y do path
        plt.plot(x, y, '-o', label=f'String {i+1}')  # Linha com marcadores
    
    # Destaca a subestação com marcador especial (estrela amarela)
    plt.scatter(coords[sub,0], coords[sub,1], s=300, c='yellow', marker='*')
    
    plt.axis('equal')  # Mantém proporção 1:1 (importante para visualização espacial)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("--- Teste cabling_v3 ---")

    # ===== layout de teste (64 torres, igual ao seu caso) =====
    x_opt = [661.0802488476509, 1028.5071476706316, 508.96385923664684, -811.2093133111011, -568.8308215348413, 135.25320530398835, 1485.0472312776456, 1730.1250520990923, 646.5554140208285, -199.42376448479095, -951.9509284418007, -1507.1615974960082, -1633.9909479438822, -1019.2818235431666, -697.4332126339438, -9.574190686191963, 871.7251315158168, 1595.0527644567876, 2456.803622909096, 1872.184838009039, 1996.8734031880808, 1210.9016644860078, 275.7769809516732, -549.3369238438046, -1043.5460752583974, -1634.7387307270328, -2099.842348708364, -2273.785647659477, -2398.041377663485, -1559.36963751651, -1162.3466840457904, -320.7091804891101, 509.84634451260695, 1326.450406770889, 1701.2674238845875, 2127.5522847938364, 2987.7405455505045, 2855.3512539844323, 2718.252786650593, 2285.6660195388986, 1857.8551884527724, 1392.4968740426127, 787.4717401929862, 137.43093561094906, -883.513574615608, -1161.334709046542, -1836.7986360836767, -2535.152451105885, -2655.315492118568, -2767.015069397514, -2867.3249981604527, -2976.727159156992, -2557.9521380151386, -2196.4428749717517, -1832.5764298269705, -1332.5418611408882, -482.1124677414058, -187.26001448925123, 643.5174096131757, 1121.401691010576, 1814.909194566765, 2336.8564045169005, 2731.7185913993453, 2926.420801140257]
    y_opt = [45.596410184418396, -102.84602376647021, 807.3970577337197, 371.58736397695736, -614.669773766668, -770.1489622929139, 232.24072673593452, 625.1181844633061, 1458.0268983660428, 1285.1783613061, 1125.1660557112616, 793.4735755352228, 223.32673728296464, -915.2891031021514, -1417.8546962289251, -1544.680665315384, -967.8222111218745, -472.52660284680076, 88.53209187034501, 1071.154478230013, 1684.9560690782678, 2204.5089768888347, 2025.8720291878376, 2520.185778550141, 1851.0238946485588, 1476.242136149511, 953.884503172178, -204.64533466558544, -760.1295456700283, -1262.8861935451305, -1858.0034535025784, -2030.4682372594812, -2183.7635495548434, -1687.768977244719, -1425.9208202323532, -828.0335781258101, -270.487321948783, 408.4430996270802, 1241.3588655768012, 1941.1338020272867, 2355.491613240731, 2657.2281471571628, 2835.123001089535, 2983.760720328497, 2648.145616002615, 2766.083599057293, 2364.8884384732605, 1604.047221383611, 1331.4834111951925, 535.9495707647394, -63.18170584095447, -372.9241616847378, -1096.6510703643166, -1689.189485811573, -2347.7758236565655, -2687.8025040381153, -2880.949347604215, -2994.090617101495, -2790.3310807019598, -2548.4676073629403, -2388.700552838015, -1868.8530123801156, -1239.9438061023466, -659.9615826554246]

    coords = np.array(list(zip(x_opt, y_opt)))

    substation_idx = 57  # igual ao seu caso

    # ===== executar algoritmo =====
    plant, res = analisar_layout_completo(
        coords,
        sub=substation_idx,
        n_grupos=64
    )

    # ===== resultados =====
    print(json.dumps(res, indent=2))

    # ===== plot =====
    plotar(plant, coords, substation_idx)
