import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math
from optimizer import optimize_parameters

# ================================================
# CLASSES PARA OBJETOS TIPO PARQUE EÓLICO
# ================================================

# Função pra calcular distância Euclidiana
# Alternativa: Função para calcular a curva catenária aumentaria a precisão do tamanho do cabeamento. Problema: saber a altura 
# entre o ponto de fixação do cabo e o assoalho continental.

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Cable:

    """Cabo entre turbinas"""
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.dI = 2.3  # A/mm²
        self.I = self.Pn / (math.sqrt(3) * self.Vn)
        self.A = max(self.I / self.dI, 50)
        self.p = 0.0173e-6 # resistividade do cobre a 20°
        self.alpha = 0.00393 # coef de temperatuea do cobre a 20°()
        self.p90 = self.p * (1 + self.alpha * (90 - 20))
        self.R = (self.p90 * self.lc) / (self.A * 1e-6)
        #self.R = (self.lc)*0.1/1000
        self.Pj = self.__Pj__()
        self.C = 2e3
        self.Ctot = self.__Ctot__()

    def __Pj__(self):
        self.Pj = 3 * (self.I ** 2) * self.R
        return self.Pj
    
    def __Ctot__(self):
        return self.lc * self.C

class Turbine:
    """Turbina com coordenadas"""
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y

class Plant:

    # Tabela de custos por metro para seções transversais industriais, pedir ajuda para o prof. Lenon pra resolver isso aqui
    INDUSTRIAL_CABLE_COSTS = {
        50: 2e3, 
        70: 3e3,
        95: 4e3,
        120: 5e3,
        150: 6e3,
        185: 7e3,
        240: 9e3,
        300: 11e3,
        400: 14e3
    }

    """Parque eólico com métodos da professora integrados"""
    def __init__(self, Vn, Tr, paths):
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.Cb = []
        self.cables_flat = []
        self.Pjtot = 0
        self.Ctot = 0
        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()
        
    def lay_cables(self):
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Ptransmitted = 0
            for k in range(len(path)-1):
                current = path[k]
                next_node = path[k+1]
                Ptransmitted += self.Tr[current].P
                length = calculate_distance(
                    self.Tr[current].x, self.Tr[current].y,
                    self.Tr[next_node].x, self.Tr[next_node].y
                )
                cable_path.append(Cable(length, self.Vn, Ptransmitted))
            self.Cb.append(cable_path)
        self.cables_flat = [item for sublist in self.Cb for item in sublist]

    def calculate_losses(self): #TODO VERIFICAR PERDAS POR JOULE QUAL A SEÇÃO RETA QUE ESTÁ SENDO USADA
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)
    
     # Método MODIFICADO para calcular o custo total usando a nova lógica
    def calculate_cost(self):
        # 1. Obter a maior seção transversal calculada necessária para toda a planta
        maior_secao_necessaria = self.get_max_calculated_section()
        print(f"\nMaior seção calculada na planta: {maior_secao_necessaria:.2f} mm²") # Opcional: print para debug

        # 2. Encontrar a seção industrial padrão imediatamente acima ou igual à necessária
        # e obter seu custo por metro. Esta lógica e os valores são inventados.
        # As chaves do dicionário são as seções industriais padrão.
        secoes_industriais_ordenadas = sorted(self.INDUSTRIAL_CABLE_COSTS.keys())

        custo_por_metro_industrial = None
        secao_industrial_usada = None

        for secao_padrao in secoes_industriais_ordenadas:
            if secao_padrao >= maior_secao_necessaria:
                custo_por_metro_industrial = self.INDUSTRIAL_CABLE_COSTS[secao_padrao]
                secao_industrial_usada = secao_padrao
                break # Encontrou a primeira seção industrial adequada

        # Se a maior seção necessária for maior que a maior seção na tabela inventada, usa o custo da maior seção da tabela.
        if custo_por_metro_industrial is None:
            secao_industrial_usada = secoes_industriais_ordenadas[-1]
            custo_por_metro_industrial = self.INDUSTRIAL_CABLE_COSTS[secao_industrial_usada]
            print(f"A maior seção calculada ({maior_secao_necessaria:.2f} mm²) excede a maior seção na tabela ({secao_industrial_usada} mm²).") # Opcional: print
        else:
             print(f"Usando seção industrial: {secao_industrial_usada} mm² com custo por metro: R$ {custo_por_metro_industrial:,.2f}") # Opcional: print

        # 3. Recalcular o custo total (Ctot) para CADA cabo na planta
        # Usando o MESMO custo por metro industrial determinado acima.
        total_cost = 0
        # self.cables_flat contém todos os objetos Cable [2, 3].
        for cable in self.cables_flat:
            # Atualiza o custo por metro no objeto Cable (opcional, mas bom para print_cables)
            cable.C = custo_por_metro_industrial
            # Recalcula o Ctot do cabo usando o novo custo por metro industrial
            cable.Ctot = cable.lc * cable.C # Ctot é calculado em Cable.__Ctot__ inicialmente [1], mas estamos sobrescrevendo aqui
            total_cost += cable.Ctot # Soma os custos individuais recalculados

        # 4. Armazenar o custo total da planta
        self.Ctot = total_cost
    
    def get_max_calculated_section(self):
        """Retorna a maior seção transversal calculada (self.A) entre todos os cabos da planta."""
        if not self.cables_flat:
            # Retorna 0 ou outro valor apropriado se a lista de cabos estiver vazia
            return 0.0
        # Itera sobre todos os cabos na lista self.cables_flat e encontra o valor máximo de self.A
        # self.A é calculado em Cable.__init__ [1]
        max_section = max(cable.A for cable in self.cables_flat)
        return max_section

    def print_turbines(self):
        print(f"Number of turbines: {len(self.Tr)}")
        print(f"{'Details':<15}{'P (MW)':>15}{'x(m)':>15}{'y(m)':>15}")
        for k, T in enumerate(self.Tr):
            print(
                f"{k:<15}"            
                f"{T.P/1e6:>15.2f}"    
                f"{T.x:>15.2f}"        
                f"{T.y:>15.2f}"        
            )
        print()
        
    def print_cables(self):
        print("Cable routing report")
        print(f"{'Connection':<15}{'Length(m)':>15}{'Section(mm²)':>15}"
            f"{'P (MW)':>15}{'P_J (kW)':>15}{'Cost milions (R$)':>22}{'Resistance':>22}")
        for j, path in enumerate(self.Cb):
            nodes = self.paths[j]
            for k, cable in enumerate(path):
                conn = f"{nodes[k]} --> {nodes[k+1]}"
                print(
                    f"{conn:<15}"
                    f"{cable.lc:15.1f}"
                    f"{cable.A:15.1f}"
                    f"{cable.Pn/1e6:15.2f}"
                    f"{cable.Pj/1e3:15.2f}"
                    f"{cable.Ctot/1e6:21.2f}"
                    f"{cable.R:21.2f}"
                )
    
# ================================================
# LÓGICA DE AGRUPAMENTO E PATHS
# ================================================

def agrupar_torres_por_cosseno(coordenadas, indice_subestacao, n_grupos=8):
    """Agrupa turbinas por similaridade angular"""
    vetores = coordenadas - coordenadas[indice_subestacao]
    mascara = np.arange(len(coordenadas)) != indice_subestacao
    
    kmeans = KMeans(n_clusters=n_grupos, random_state=42)
    grupos = kmeans.fit_predict(cosine_similarity(vetores[mascara]))
    
    indices_originais = np.where(mascara)[0].tolist()
    return [sorted([indices_originais[i] for i in np.where(grupos == id)[0]]) 
            for id in range(n_grupos)]

def balancear_grupos(grupos, vectors, alvo=8):
    """Balanceia grupos mantendo similaridade"""
    def similaridade_grupo(turbina, grupo):
        return np.mean([cosine_similarity([vectors[turbina]], [vectors[t]])[0][0] for t in grupo]) if grupo else 0
    
    balanced = [g.copy() for g in grupos]
    
    for _ in range(10):
        tamanhos = [len(g) for g in balanced]
        if all(t == alvo for t in tamanhos): break
        
        fontes = sorted([(len(g), i) for i, g in enumerate(balanced) if len(g) > alvo], reverse=True)
        
        for _, fonte in fontes:
            while len(balanced[fonte]) > alvo:
                turbinas = balanced[fonte]
                sims = [similaridade_grupo(t, turbinas) for t in turbinas]
                turbina = turbinas.pop(np.argmin(sims))
                melhor_dest = np.argmax([similaridade_grupo(turbina, g) if i != fonte and len(g) < alvo else -1 
                                      for i, g in enumerate(balanced)])
                balanced[melhor_dest].append(turbina)
    
    # Fase 2: Swaps seguros
    melhorias = True
    while melhorias:
        melhorias = False
        for i in range(len(balanced)):
            for j in range(i+1, len(balanced)):
                # Cria cópias para iterar
                copia_i = list(balanced[i])
                copia_j = list(balanced[j])
                
                for t1 in copia_i:
                    for t2 in copia_j:
                        # Verifica existência atual
                        if t1 not in balanced[i] or t2 not in balanced[j]:
                            continue
                            
                        sim_original = (similaridade_grupo(t1, balanced[i]) +
                                      similaridade_grupo(t2, balanced[j]))
                        sim_novo = (similaridade_grupo(t2, balanced[i]) +
                                  similaridade_grupo(t1, balanced[j]))
                        
                        if sim_novo > sim_original:
                            try:
                                balanced[i].remove(t1)
                                balanced[j].remove(t2)
                                balanced[i].append(t2)
                                balanced[j].append(t1)
                                melhorias = True
                            except ValueError:
                                continue
    return balanced

def gen_groups(vectors, n_groups):
    # usa coord/substation do módulo
    return agrupar_torres_por_cosseno(coordinates, substation_idx, n_groups)

def make_route(grupos):
    return [
        sorted(g, key=lambda t: -np.linalg.norm(coordinates[t]-coordinates[substation_idx]))
        + [substation_idx]
        for g in grupos
    ]

def calc_loss(paths):
    plant = Plant(33e3,
                  [Turbine(3.35e6, *coordinates[i]) for i in range(len(coordinates))],
                  paths)
    return plant.Pjtot

def calc_cost(paths):
    plant = Plant(33e3,
                  [Turbine(3.35e6, *coordinates[i]) for i in range(len(coordinates))],
                  paths)
    return plant.Ctot

# ================================================
# DADOS E EXECUÇÃO
# ================================================

""" #opt 16
x_opt = [-134.92021516964914, 475.55640005887045, 47.341265591348645, -291.87618347006656, -467.2781430286087, 349.0231420338762, 1203.3903276016408, 1080.9073355869466, 247.26455478961583, -725.2478601140702, -1100.162845125174, -1294.5429885693968, -949.8004833332534, -609.7234640375709, 181.263092795678, 1148.844649045738]
y_opt = [-801.113357244488, 275.0418014163371, 914.8073394319176, 440.4982633422105, -288.84091129919307, -451.7884456528136, 94.70985786260522, 722.1799528444428, 1276.1492031053472, 1078.8503732558997, 588.5083327203014, -117.19792126546781, -633.071166755402, -1130.428394544164, -1287.2676446864946, -608.0829014957618]
 """

""" #opt 36
x_opt = [-27.067259971937148, 859.9474892894142, 159.75467304589642, -764.3261925007537, -630.2551072688764, 338.0167388830049, 1385.446321490046, 1022.7870107979952, 671.2374491011798, -179.86081069819423, -685.0770394572637, -1232.6743863724075, -1398.6025271805358, -974.5813363405996, -843.9649134535559, 131.0752180452796, 226.46857507937392, 1230.960459631853, 1999.5386345844709, 1906.6824056217906, 1541.2586483326731, 1170.6475873952145, 331.3339286754586, -511.19217557434087, -1046.9373856960315, -1583.5271477143244, -1834.4147530618209, -1989.5035161179699, -1931.2444380577317, -1563.6453899004166, -1111.222989553677, -407.29369501511314, 486.35441964532856, 1060.056332658777, 1585.2161717422005, 1754.1290155705983]
y_opt = [47.03479638372782, -147.96706358119582, 898.5826258845099, 352.0311484994638, -331.74933628005664, -508.69694106427136, 197.94200170223692, 730.120817802791, 1261.0107977656942, 1423.1371297558192, 1053.0541627231373, 685.7874145210852, -134.62972841727978, -683.5478245764709, -1204.855130884686, -1407.1659516076497, -931.2431727696841, -698.9220887857573, 42.956170200073366, 545.0677675831894, 1085.2623971270032, 1621.5992763065585, 1785.4180281802073, 1933.5673138137022, 1584.6023525757832, 1221.6552924348794, 534.9722283126675, 204.63551104220318, -503.918716192347, -1063.494074446592, -1662.884072454275, -1794.6508063491021, -1939.9637553948983, -1568.2178126817287, -1219.4626036178643, -330.220704217598]
 """

#opt 64
x_opt = [661.0802488476509, 1028.5071476706316, 508.96385923664684, -811.2093133111011, -568.8308215348413, 135.25320530398835, 1485.0472312776456, 1730.1250520990923, 646.5554140208285, -199.42376448479095, -951.9509284418007, -1507.1615974960082, -1633.9909479438822, -1019.2818235431666, -697.4332126339438, -9.574190686191963, 871.7251315158168, 1595.0527644567876, 2456.803622909096, 1872.184838009039, 1996.8734031880808, 1210.9016644860078, 275.7769809516732, -549.3369238438046, -1043.5460752583974, -1634.7387307270328, -2099.842348708364, -2273.785647659477, -2398.041377663485, -1559.36963751651, -1162.3466840457904, -320.7091804891101, 509.84634451260695, 1326.450406770889, 1701.2674238845875, 2127.5522847938364, 2987.7405455505045, 2855.3512539844323, 2718.252786650593, 2285.6660195388986, 1857.8551884527724, 1392.4968740426127, 787.4717401929862, 137.43093561094906, -883.513574615608, -1161.334709046542, -1836.7986360836767, -2535.152451105885, -2655.315492118568, -2767.015069397514, -2867.3249981604527, -2976.727159156992, -2557.9521380151386, -2196.4428749717517, -1832.5764298269705, -1332.5418611408882, -482.1124677414058, -187.26001448925123, 643.5174096131757, 1121.401691010576, 1814.909194566765, 2336.8564045169005, 2731.7185913993453, 2926.420801140257]
y_opt = [45.596410184418396, -102.84602376647021, 807.3970577337197, 371.58736397695736, -614.669773766668, -770.1489622929139, 232.24072673593452, 625.1181844633061, 1458.0268983660428, 1285.1783613061, 1125.1660557112616, 793.4735755352228, 223.32673728296464, -915.2891031021514, -1417.8546962289251, -1544.680665315384, -967.8222111218745, -472.52660284680076, 88.53209187034501, 1071.154478230013, 1684.9560690782678, 2204.5089768888347, 2025.8720291878376, 2520.185778550141, 1851.0238946485588, 1476.242136149511, 953.884503172178, -204.64533466558544, -760.1295456700283, -1262.8861935451305, -1858.0034535025784, -2030.4682372594812, -2183.7635495548434, -1687.768977244719, -1425.9208202323532, -828.0335781258101, -270.487321948783, 408.4430996270802, 1241.3588655768012, 1941.1338020272867, 2355.491613240731, 2657.2281471571628, 2835.123001089535, 2983.760720328497, 2648.145616002615, 2766.083599057293, 2364.8884384732605, 1604.047221383611, 1331.4834111951925, 535.9495707647394, -63.18170584095447, -372.9241616847378, -1096.6510703643166, -1689.189485811573, -2347.7758236565655, -2687.8025040381153, -2880.949347604215, -2994.090617101495, -2790.3310807019598, -2548.4676073629403, -2388.700552838015, -1868.8530123801156, -1239.9438061023466, -659.9615826554246]


""" #baseline 16
x_opt = [0., 650., 200.861, -525.861, -525.861, 200.861, 1300., 1051.7221, 401.7221, -401.7221, -1051.7221,
           -1300., -1051.7221, -401.7221, 401.7221, 1051.7221]
y_opt = [0., 0., 618.1867, 382.0604, -382.0604, -618.1867, 0., 764.1208, 1236.3735, 1236.3735, 764.1208, 0.,
           -764.1208, -1236.3735, -1236.3735, -764.1208] """


""" #baseline 36
x_opt = [0., 666.6667, 206.0113, -539.3447, -539.3447, 206.0113, 1333.3333, 1154.7005, 666.6667, 0.,
           -666.6667, -1154.7005, -1333.3333, -1154.7005, -666.6667, 0., 666.6667, 1154.7005, 2000., 1879.3852,
           1532.0889, 1000., 347.2964, -347.2964, -1000., -1532.0889, -1879.3852, -2000., -1879.3852, -1532.0889,
           -1000., -347.2964, 347.2964, 1000., 1532.0889, 1879.3852]
y_opt = [0., 0., 634.0377, 391.8568, -391.8568, -634.0377, 0., 666.6667, 1154.7005, 1333.3333,
           1154.7005, 666.6667, 0., -666.6667, -1154.7005, -1333.3333, -1154.7005, -666.6667, 0., 684.0403,
           1285.5752, 1732.0508, 1969.6155, 1969.6155, 1732.0508, 1285.5752, 684.0403, 0., -684.0403, -1285.5752,
           -1732.0508, -1969.6155, -1969.6155, -1732.0508, -1285.5752, -684.0403] """


""" #baseline 64
x_opt = [0., 750., 231.7627, -606.7627, -606.7627, 231.7627, 1500., 1299.0381, 750., 0.,
           -750., -1299.0381, -1500., -1299.0381, -750., 0., 750., 1299.0381, 2250, 2114.3084,
           1723.6, 1125., 390.7084, -390.7084, -1125., -1723.6, -2114.3084, -2250., -2114.3084, -1723.6,
           -1125, -390.7084, 390.7084, 1125., 1723.6, 2114.3084, 3000., 2924.7837, 2702.9066, 2345.4944,
           1870.4694, 1301.6512, 667.5628, 0., -667.5628, -1301.6512, -1870.4694, -2345.4944, -2702.9066, -2924.7837,
           -3000., -2924.7837, -2702.9066, -2345.4944, -1870.4694, -1301.6512, -667.5628, 0., 667.5628, 1301.6512,
           1870.4694, 2345.4944, 2702.9066, 2924.7837]
y_opt = [0., 0., 713.2924, 440.8389, -440.8389, -713.2924, 0., 750., 1299.0381, 1500,
           1299.0381, 750., 0., -750., -1299.0381, -1500., -1299.0381, -750., 0., 769.5453,
           1446.2721, 1948.5572, 2215.8174, 2215.8174, 1948.5572, 1446.2721, 769.5453, 0., -769.5453, -1446.2721,
           -1948.5572, -2215.8174, -2215.8174, -1948.5572, -1446.2721, -769.5453, 0., 667.5628, 1301.6512, 1870.4694,
           2345.4944, 2702.9066, 2924.7837, 3000., 2924.7837, 2702.9066, 2345.4944, 1870.4694, 1301.6512, 667.5628,
           0., -667.5628, -1301.6512, -1870.4694, -2345.4944, -2702.9066, -2924.7837, -3000., -2924.7837, -2702.9066,
           -2345.4944, -1870.4694, -1301.6512, -667.5628] """

""" x_opt = [-136.98835237100258, 475.28496493442435, 44.594104173051306, -293.5414392169158, -469.94233983256936, 346.9677885250473, 1206.3018471071273, 1081.0309704906497, 247.13894652525005, -725.3010986951364, -1102.747073031985, -1294.4449970898518, -950.4035830514036, -613.0687013550835, 178.56824550804876, 1148.2883108089245]
y_opt =[-805.3715385981511, 271.66288218328634, 914.5028781442215, 437.5925009499115, -293.59229050514307, -456.0016333844102, 90.32256539522399, 722.0609675352182, 1276.2924199059453, 1078.8597296348412, 586.6722706600424, -120.05061224720494, -635.927944113068, -1133.0055090088122, -1287.6775146314112, -609.4538171638917]
 """

substation_idx = 57
coordinates = np.array(list(zip(x_opt, y_opt)))


if __name__ == "__main__":
    grupos = agrupar_torres_por_cosseno(coordinates, substation_idx)
    vectors = coordinates - coordinates[substation_idx]
    grupos_balanceados = balancear_grupos(grupos, vectors)
    
    paths = []
    for grupo in grupos_balanceados:
        distancias = [np.linalg.norm(coordinates[t] - coordinates[substation_idx]) for t in grupo]
        ordenado = [x for _, x in sorted(zip(distancias, grupo), reverse=True)]
        paths.append(ordenado + [substation_idx])
    
    Tr_opt = [Turbine(3.35e6, x, y) for x, y in zip(x_opt, y_opt)]
    plant = Plant(Vn=33e3, Tr=Tr_opt, paths=paths)
    
    plant.print_turbines() # Gera report de turbinas
    plant.print_cables() # Gera report de cabeamento
    
    cotDolar = 0.1722 # Conversão do real para dólar 21/04/25
    
    print("\n=== Resultados ===")
    print(f"Perdas totais: {plant.Pjtot/1e6:.2f} MW")
    print(f"Perdas anuais: {plant.Pjtot*8760/1e6:.2f} MWh")
    print(f"Custo total: $ {plant.Ctot*cotDolar/1e6:.2f} milions")

    plt.figure(figsize=(12,12))
    cores = [
    '#FF0000',  # vermelho
    '#FFD700',  # amarelo
    '#0000FF',  # azul
    '#008000',  # verde
    '#FFA500',  # laranja
    '#00FFFF',  # ciano
    '#FF00FF',  # magenta
    '#800080',  # roxo
    '#A52A2A',  # marrom
    '#000000'   # preto
    ]
    
    for i, path in enumerate(paths):
        x = [coordinates[node][0] for node in path]
        y = [coordinates[node][1] for node in path]
        plt.plot(x, y, 'o-', linewidth=2, color=cores[i], 
                markersize=8, markeredgecolor='black', label=f'String {i+1}')
        
        for node in path:
            plt.text(coordinates[node][0] + 10, coordinates[node][1] + 10,
                    str(node), fontsize=12, color='black', weight='bold')
    
    plt.scatter(coordinates[substation_idx][0], coordinates[substation_idx][1],
               c='yellow', s=300, marker='*', edgecolors='black', label='Substation')
    
    plt.title("Cable connections", fontsize=24)
    plt.xlabel("X (m)", fontsize=22)
    plt.ylabel("Y (m)", fontsize=22)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

"""" coordinates = np.column_stack((x_opt, y_opt))
substation_idx = 57
vectors = coordinates - coordinates[substation_idx]


if __name__ == "__main__":
    n_groups_list = [8, 9, 10, 11, 12]
    alvo_list     = [8, 9, 10]
    weight_loss   = 0.6
    weight_cost   = 0.4

    best, df = optimize_parameters(
        vectors=vectors,
        n_groups_list=n_groups_list,
        alvo_list=alvo_list,
        generate_initial_groups=gen_groups,
        balancear_grupos=balancear_grupos,
        define_route=make_route,
        compute_joule_loss=calc_loss,
        compute_cable_cost=calc_cost,
        weight_loss=weight_loss,
        weight_cost=weight_cost
    )

    print("### Melhor Parâmetros ###")
    print(best)

    # usa best para gerar solução final
    ng_opt   = int(best['n_groups'])
    alvo_opt = int(best['alvo'])
    grupos0  = agrupar_torres_por_cosseno(coordinates, substation_idx, ng_opt)
    grupos   = balancear_grupos(grupos0, vectors, alvo=alvo_opt)

    paths = make_route(grupos)
    Tr_list = [Turbine(3.35e6, *coordinates[i]) for i in range(len(coordinates))]
    plant   = Plant(Vn=33e3, Tr=Tr_list, paths=paths)

    plant.print_turbines()
    plant.print_cables() """

# 1³ artigo:
    #otimização sequencial
    #otimização paralela
    

# 2° artigo: tema: custo
    # otimizações individuais
    # topologias
        # quantidade de substações
        # tipo de torres - mais ou menos potencia nominal
        # tensão do parque eolico (hj é 33kV)
        # tipo de cabo (seção reta diferente de cabo em diferentes trechos)