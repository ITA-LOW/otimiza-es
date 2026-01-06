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
    return math.hypot(x2 - x1, y2 - y1)

# ======================================================
# CLASSES ELÉTRICAS (INALTERADAS)
# ======================================================

class Cable:
    SECTION_TABLE = {
        50: 0.49, 70: 0.34, 95: 0.25, 120: 0.20,
        150: 0.16, 185: 0.13, 240: 0.10,
    }

    def __init__(self, lc, Vn, Pn):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.dI = 2.3
        self.I = self.Pn / (math.sqrt(3) * self.Vn)
        self.A_continuous = self.I / self.dI
        self.A = None
        self.R_km = None
        self.R = None
        self.Pj = None
        self.C = 0
        self.Ctot = 0

    def assign_section(self, section):
        self.A = section
        self.R_km = self.SECTION_TABLE[section]
        self.R = self.R_km * (self.lc / 1000)
        self.Pj = 3 * (self.I ** 2) * self.R


class Turbine:
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y


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
        240: 250.27  # 240 * 1.0428
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
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.Cb = []
        self.cables_flat = []
        self.Pjtot = 0
        self.Ctot = 0

        self.lay_cables()
        self.uniform_section()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Pacc = 0
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                Pacc += self.Tr[a].P
                L = calculate_distance(
                    self.Tr[a].x, self.Tr[a].y,
                    self.Tr[b].x, self.Tr[b].y
                )
                cable_path.append(Cable(L, self.Vn, Pacc))
            self.Cb.append(cable_path)

        self.cables_flat = [c for p in self.Cb for c in p]

    def uniform_section(self):
        Amax = max(c.A_continuous for c in self.cables_flat)
        chosen = max(Cable.SECTION_TABLE)
        for sec in sorted(Cable.SECTION_TABLE):
            if sec >= Amax:
                chosen = sec
                break
        for c in self.cables_flat:
            c.assign_section(chosen)

    def calculate_losses(self):
        self.Pjtot = sum(c.Pj for c in self.cables_flat)

    def calculate_cost(self):
        sec = self.cables_flat[0].A
        custo_m = self.INDUSTRIAL_CABLE_COSTS[sec]
        self.Ctot = 0
        for c in self.cables_flat:
            c.C = custo_m
            c.Ctot = c.lc * custo_m
            self.Ctot += c.Ctot

    def get_max_calculated_section(self):
        return self.cables_flat[0].A

# ======================================================
# AGRUPAMENTO ANGULAR (SEM BALANCEAMENTO)
# ======================================================

def agrupar_por_setor_angular(coords, sub, n_grupos):
    """
    Agrupa turbinas por setores angulares contíguos em relação à subestação.
    Garante que cada grupo seja uma fatia angular estritamente separada.
    """
    # Calcula vetores da subestação para cada turbina
    v = coords - coords[sub]
    
    # Calcula ângulos de todas as turbinas em relação à subestação
    ang = np.arctan2(v[:, 1], v[:, 0])
    
    # Remove a subestação do cálculo (seu ângulo não importa)
    # Cria array de índices excluindo a subestação
    indices_turbinas = np.array([i for i in range(len(coords)) if i != sub])
    angulos_turbinas = ang[indices_turbinas]
    
    # Ordena índices por ângulo (ordem angular crescente)
    idx_ordenado = indices_turbinas[np.argsort(angulos_turbinas)]
    
    # Divide em n_grupos fatias angulares contíguas usando array_split
    grupos = np.array_split(idx_ordenado, n_grupos)
    
    # Converte para listas e remove grupos vazios
    grupos_filtrados = [list(g) for g in grupos if len(g) > 0]
    
    return grupos_filtrados

# ======================================================
# FUNÇÃO PRINCIPAL
# ======================================================

def analisar_layout_completo(coords, sub, n_grupos=15, Vn=33e3, P_turb=3.35e6):
    """
    Analisa layout completo com agrupamento angular estrito.
    Cada grupo é uma fatia angular contígua, sem balanceamento que cause cruzamentos.
    """
    N = len(coords)
    
    # Agrupa turbinas por setores angulares contíguos
    grupos = agrupar_por_setor_angular(coords, sub, n_grupos)
    
    # Cria paths: dentro de cada grupo, ordena por distância radial decrescente
    paths = []
    for g in grupos:
        if len(g) == 0:
            continue
        
        # Calcula distâncias de cada turbina do grupo à subestação
        distancias = [np.linalg.norm(coords[t] - coords[sub]) for t in g]
        
        # Ordena por distância decrescente (mais longe primeiro)
        # Isso garante ordem radial estrita: da turbina mais distante para a mais próxima
        ordenado = [t for _, t in sorted(zip(distancias, g), reverse=True)]
        
        # Adiciona subestação no final do path
        paths.append(ordenado + [sub])

    turbinas = [Turbine(P_turb, x, y) for x, y in coords]
    planta = Plant(Vn, turbinas, paths)

    COT_DOLAR = 0.1722  # mesmo valor usado nas versões anteriores

    comprimento_total = sum(c.lc for c in planta.cables_flat)
    perda_anual_mwh = planta.Pjtot * 8760 / 1e6
    perda_total_kw = planta.Pjtot / 1e3
    secao = planta.get_max_calculated_section()
    custo_total = planta.Ctot
    custo_total_usd = custo_total * COT_DOLAR

    resultados = {
        # --- chaves históricas (V1 / V2) ---
        "custo_total_usd": custo_total_usd,
        "comprimento_total_m": comprimento_total,
        "perda_total_kw": perda_total_kw,
        "perda_anual_mwh": perda_anual_mwh,
        "secao_cabo_mm2": secao,

        # --- chaves novas (V3, mantidas) ---
        "custo_total": custo_total,
        "secao_mm2": secao
    }

    return planta, resultados

# ======================================================
# VISUALIZAÇÃO
# ======================================================

def plotar(planta, coords, sub):
    plt.figure(figsize=(10,10))
    for i, p in enumerate(planta.paths):
        x = [coords[k,0] for k in p]
        y = [coords[k,1] for k in p]
        plt.plot(x, y, '-o', label=f'String {i+1}')
    plt.scatter(coords[sub,0], coords[sub,1], s=300, c='yellow', marker='*')
    plt.axis('equal')
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
