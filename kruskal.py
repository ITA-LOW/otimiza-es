""" import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ===================== DADOS DE ENTRADA =====================
x_opt = [456.7204089099218, 438.3919220444296, -399.92448768287204, -520.3471492587372, -406.78323514290537,
         116.0430277199342, 1299.9147666148804, 935.4797522168615, 580.9914866745228, -288.6249657521407,
         -1091.8161063628386, -984.6468715598825, -796.0997345716727, -222.36747893782228, 639.3216959582929,
         991.2686388396767]

y_opt = [188.21833363635656, -207.8664662389235, 872.844372473487, 350.4894749171311, -381.96064772293386,
         -749.2747718669738, -14.886219683036972, 536.1257300488144, 1080.69374416935, 1267.5549807167085,
         705.6469300480154, 2.399217995711785, -926.1619079629315, -1280.8406240861154, -1118.7085204533382,
         -576.566173271566]

substation_idx = 13  # Índice da subestação
coordinates = np.array(list(zip(x_opt, y_opt)))

# ===================== FUNÇÕES PRINCIPAIS =====================
def agrupar_torres_por_cosseno(coordenadas, indice_subestacao, n_grupos=4):
    #Agrupa turbinas por similaridade angular relativa à subestação
    vetores = coordenadas - coordenadas[indice_subestacao]
    mascara = np.arange(len(coordenadas)) != indice_subestacao
    vetores_turbinas = vetores[mascara]
    
    kmeans = KMeans(n_clusters=n_grupos, random_state=42)
    grupos = kmeans.fit_predict(cosine_similarity(vetores_turbinas))
    
    indices_originais = np.where(mascara)[0].tolist()
    return [sorted([indices_originais[i] for i in np.where(grupos == id)[0]]) for id in range(n_grupos)]

def balancear_grupos(grupos, vectors, alvo=4):
    #Balanceia grupos mantendo máxima similaridade interna
    def similaridade_grupo(turbina, grupo):
        if not grupo: return 0
        return np.mean([cosine_similarity([vectors[turbina]], [vectors[t]])[0][0] for t in grupo])
    
    balanced = [g.copy() for g in grupos]
    
    # Fase 1: Redistribuição de turbinas excedentes
    for _ in range(10):
        tamanhos = [len(g) for g in balanced]
        if all(t == alvo for t in tamanhos): break
        
        # Priorizar grupos com maior excesso
        fontes = sorted([(len(g), i) for i, g in enumerate(balanced) if len(g) > alvo], reverse=True)
        
        for _, fonte in fontes:
            while len(balanced[fonte]) > alvo:
                # Encontrar turbina menos similar
                turbinas = balanced[fonte]
                sims = [similaridade_grupo(t, turbinas) for t in turbinas]
                idx = np.argmin(sims)
                turbina = turbinas.pop(idx)
                
                # Encontrar melhor destino
                melhor_dest, melhor_sim = -1, -1
                for d, grupo in enumerate(balanced):
                    if d == fonte or len(grupo) >= alvo: continue
                    sim = similaridade_grupo(turbina, grupo)
                    if sim > melhor_sim:
                        melhor_sim, melhor_dest = sim, d
                
                if melhor_dest != -1:
                    balanced[melhor_dest].append(turbina)
    
    # Fase 2: Otimização por trocas pareadas
    melhorias = True
    while melhorias:
        melhorias = False
        for i in range(len(balanced)):
            for j in range(i+1, len(balanced)):
                for t1 in balanced[i]:
                    for t2 in balanced[j]:
                        # Calcular ganho potencial
                        sim_original = (similaridade_grupo(t1, balanced[i]) +
                                      similaridade_grupo(t2, balanced[j]))
                        sim_novo = (similaridade_grupo(t2, balanced[i]) +
                                   similaridade_grupo(t1, balanced[j]))
                        
                        if sim_novo > sim_original:
                            balanced[i].remove(t1)
                            balanced[j].remove(t2)
                            balanced[i].append(t2)
                            balanced[j].append(t1)
                            melhorias = True
    
    return balanced

def plot_grupos(grupos, titulo):
    #Visualiza grupos com cores diferentes
    cores = ['#FF0000', '#FFD700', '#0000FF', '#008000']
    plt.figure(figsize=(12,10))
    
    for i, grupo in enumerate(grupos):
        pontos = coordinates[grupo]
        plt.scatter(pontos[:,0], pontos[:,1], c=cores[i], s=100,
                   edgecolor='black', label=f'String {i+1}')
        
        for t in grupo:
            x, y = coordinates[t]
            plt.plot([coordinates[substation_idx][0], x],
                     [coordinates[substation_idx][1], y], 
                     color=cores[i], alpha=0.3)
            plt.text(x, y, str(t), fontsize=8, ha='center', va='bottom')
    
    plt.scatter(*coordinates[substation_idx], c='black', marker='s',
               s=200, label='Subestação')
    plt.title(titulo, fontsize=14)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ===================== PROCESSAMENTO PRINCIPAL =====================
if __name__ == "__main__":
    # Passo 1: Agrupamento inicial
    grupos = agrupar_torres_por_cosseno(coordinates, substation_idx)
    print("Grupos originais:")
    for i, g in enumerate(grupos, 1):
        print(f"String {i} ({len(g)}): {sorted(g)}")
    
    # Passo 2: Balanceamento
    vectors = coordinates - coordinates[substation_idx]
    grupos_balanceados = balancear_grupos(grupos, vectors)
    print("\nGrupos balanceados:")
    for i, g in enumerate(grupos_balanceados, 1):
        print(f"String {i} ({len(g)}): {sorted(g)}")
    
    # Passo 3: Visualização
    plot_grupos(grupos, "Agrupamento Original")
    plot_grupos(grupos_balanceados, "Agrupamento Balanceado") """

###################3

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math

# ================================================
# CLASSES MODIFICADAS PARA ORDEM CORRETA
# ================================================

# Function to calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Cable:
    """Cabo entre turbinas (ordem reversa) com elementos da professora"""
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.dI = 2.3  # A/mm²
        self.I = self.Pn / (math.sqrt(3) * self.Vn)
        self.A = max(self.I / self.dI, 50)
        self.p = 0.0173e-6
        self.alpha = 0.393
        self.p90 = self.p * (1 + self.alpha * (90 - 20))
        self.R = (self.p90 * self.lc) / (self.A * 1e-6)
        self.Pj = self.__Pj__()
        self.C = 2e3  # Custo por metro da professora
        self.Ctot = self.__Ctot__()

    def __Pj__(self):
        self.Pj = 3 * self.lc * self.A * self.p90 * self.I ** 2
        #self.Pj = 3 * (self.I ** 2) * self.R
        return self.Pj
    
    def __Ctot__(self):
        return self.lc * self.C

class Turbine:
    """Turbina com coordenadas (mantido igual aos dois códigos)"""
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y

class Plant:
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
    
    def calculate_losses(self):
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)
    
    def calculate_cost(self):
        self.Ctot = sum(cable.Ctot for cable in self.cables_flat)
    
    # Novos métodos da professora
    def print_turbines(self):
        print(f"Number of turbines: {len(self.Tr)}")
        print(f"Details:     \t  P (MW)  \t  x(m)    \t  y(m)")
        for k, T in enumerate(self.Tr):
            print(f"{k}     \t  {T.P/1e6}  \t  {T.x}    \t  {T.y}")
    
    def print_cables(self):
        print(f"Number of cables: {len(self.cables_flat)}")
        print(f"Connecting turbines \t length(m)  \t  P transmitted (MW) \t P_J (kW)    \t  Cost (R$)")
        for j, path in enumerate(self.Cb):
            for k, cable in enumerate(path):
                nodes = self.paths[j]
                print(f"{nodes[k]} and {nodes[k+1]}      \t {cable.lc:.1f}       \t  {cable.Pn / 1e6:.2f}  \t \t  {cable.Pj / 1e3:.2f}  \t  {cable.Ctot:.2f} ")

# ================================================
# LÓGICA DE AGRUPAMENTO E PATHS CORRIGIDA
# ================================================

def agrupar_torres_por_cosseno(coordenadas, indice_subestacao, n_grupos=5):
    """Agrupa turbinas por similaridade angular"""
    vetores = coordenadas - coordenadas[indice_subestacao]
    mascara = np.arange(len(coordenadas)) != indice_subestacao
    
    kmeans = KMeans(n_clusters=n_grupos, random_state=42)
    grupos = kmeans.fit_predict(cosine_similarity(vetores[mascara]))
    
    indices_originais = np.where(mascara)[0].tolist()
    return [sorted([indices_originais[i] for i in np.where(grupos == id)[0]]) 
            for id in range(n_grupos)]

def balancear_grupos(grupos, vectors, alvo=7):
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

# ================================================
# DADOS E EXECUÇÃO COM ORDENAÇÃO CORRETA
# ================================================

""" #opt 16
x_opt = [-134.92021516964914, 475.55640005887045, 47.341265591348645, -291.87618347006656, -467.2781430286087, 349.0231420338762, 1203.3903276016408, 1080.9073355869466, 247.26455478961583, -725.2478601140702, -1100.162845125174, -1294.5429885693968, -949.8004833332534, -609.7234640375709, 181.263092795678, 1148.844649045738]
y_opt = [-801.113357244488, 275.0418014163371, 914.8073394319176, 440.4982633422105, -288.84091129919307, -451.7884456528136, 94.70985786260522, 722.1799528444428, 1276.1492031053472, 1078.8503732558997, 588.5083327203014, -117.19792126546781, -633.071166755402, -1130.428394544164, -1287.2676446864946, -608.0829014957618]
 """

""" #opt 64
x_opt = [570.2196253107965, 988.7458935877285, 378.0303332681764, -759.3155250479742, -598.2889195429211, 115.32944434129905, 1532.3474738757197, 1643.5754371923329, 678.879799806922, -173.55352441282383, -916.1554820213375, -1520.336702274256, -1656.0964201253623, -1066.3971891539818, -734.3109890881726, -27.214314445697077, 835.2961913225896, 1481.114965117314, 2508.9102213512488, 1838.8250664860288, 1982.6306020270194, 1190.5031614289394, 284.0517695394151, -522.2662221863669, -1017.9689963107899, -1647.3553230293264, -2107.1561710494943, -2289.018909349188, -2410.231506660626, -1586.7148792449693, -1178.6696847490875, -346.2583407817287, 484.41346116127204, 1307.8690553506744, 1648.080001373377, 2062.5555975073166, 2985.1750277445926, 2825.930810347797, 2698.252045387348, 2285.1701009186177, 1857.1436671352888, 1393.364924533131, 785.5170709216653, 156.3766791927061, -866.2704743160956, -1161.0093964183166, -1837.0551863866824, -2534.988300917158, -2657.951994806712, -2774.707037690126, -2870.958194264016, -2976.722108850463, -2568.194957023111, -2214.3747583954546, -1852.6865267101298, -1331.842786502385, -513.0141853388171, -199.6937876100551, 624.0558336421687, 1096.5704467470196, 1814.6751946298264, 2239.4993436135064, 2667.053902064622, 2926.5882089172774]
y_opt = [73.55437004768822, -164.62667704749305, 927.200624920488, 326.43241497478294, -590.5247034804956, -748.7336317242593, 205.93715274116667, 681.1782764396164, 1457.020463768242, 1284.1894629413725, 1132.5106915430454, 781.8303756970773, 203.83081826441367, -898.7973414916097, -1403.8716073243959, -1538.9404749148136, -959.9077069522826, -478.37188793537734, -2.1494773056994765, 1119.0634422406865, 1706.7883586022754, 2206.5633979217973, 2038.6335679526617, 2517.4327558675277, 1855.9005261169104, 1472.1102250670704, 948.6905347373213, -201.42966862107835, -748.0495928759142, -1245.358548213717, -1855.3323308088009, -2026.6862871731812, -2180.3250044134697, -1679.3001351597231, -1437.0719260083981, -836.60730852379, -293.81228003861327, 444.749562120928, 1283.1027163716471, 1942.7454419723892, 2355.833686691143, 2656.5703348759052, 2823.0403843967597, 2969.284035070354, 2646.3034602789244, 2766.1454368963205, 2364.489405600105, 1603.875781354132, 1324.0804326540697, 521.8951158822695, -55.04139616334269, -372.7074088966376, -1082.58347210687, -1677.00424065585, -2333.4021142185134, -2688.0826056424203, -2880.8311349796195, -2993.257050812749, -2799.325955855874, -2545.2297623242275, -2388.862921753052, -1849.0690130950666, -1236.4176993969857, -656.5836405808162]
 """

""" #baseline 16
x_opt = [0., 650., 200.861, -525.861, -525.861, 200.861, 1300., 1051.7221, 401.7221, -401.7221, -1051.7221,
           -1300., -1051.7221, -401.7221, 401.7221, 1051.7221]
y_opt = [0., 0., 618.1867, 382.0604, -382.0604, -618.1867, 0., 764.1208, 1236.3735, 1236.3735, 764.1208, 0.,
           -764.1208, -1236.3735, -1236.3735, -764.1208] """


#baseline 36
x_opt = [0., 666.6667, 206.0113, -539.3447, -539.3447, 206.0113, 1333.3333, 1154.7005, 666.6667, 0.,
           -666.6667, -1154.7005, -1333.3333, -1154.7005, -666.6667, 0., 666.6667, 1154.7005, 2000., 1879.3852,
           1532.0889, 1000., 347.2964, -347.2964, -1000., -1532.0889, -1879.3852, -2000., -1879.3852, -1532.0889,
           -1000., -347.2964, 347.2964, 1000., 1532.0889, 1879.3852]
y_opt = [0., 0., 634.0377, 391.8568, -391.8568, -634.0377, 0., 666.6667, 1154.7005, 1333.3333,
           1154.7005, 666.6667, 0., -666.6667, -1154.7005, -1333.3333, -1154.7005, -666.6667, 0., 684.0403,
           1285.5752, 1732.0508, 1969.6155, 1969.6155, 1732.0508, 1285.5752, 684.0403, 0., -684.0403, -1285.5752,
           -1732.0508, -1969.6155, -1969.6155, -1732.0508, -1285.5752, -684.0403]


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

substation_idx = 31
coordinates = np.array(list(zip(x_opt, y_opt)))


if __name__ == "__main__":
    # Sua lógica original de agrupamento e paths
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
    
    # Usando novos métodos da professora
    plant.print_turbines()
    plant.print_cables()
    
    print("\n=== Resultados Consolidados ===")
    print(f"Perdas totais: {plant.Pjtot/1e6:.2f} MW")
    print(f"Perdas anuais: {plant.Pjtot*8760/1e6:.2f} MWh")
    print(f"Custo total: R$ {plant.Ctot/1e6:.2f} milhões")

    # 5. Plotagem corrigida
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