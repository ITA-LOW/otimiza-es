# =========================================================
# cabling.py — Versão PLANAR (sem cruzamento de cabos)
# =========================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import json

# =========================================================
# CLASSES E FUNÇÕES BÁSICAS (INALTERADAS)
# =========================================================

def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

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
        self.R = None
        self.Pj = None
        self.Ctot = 0

    def assign_section(self, sec):
        self.A = sec
        R_km = self.SECTION_TABLE[sec]
        self.R = R_km * (self.lc / 1000)
        self.Pj = 3 * self.I**2 * self.R

class Turbine:
    def __init__(self, P, x, y):
        self.P = P
        self.x = x
        self.y = y

class Plant:
    INDUSTRIAL_CABLE_COSTS = {
        50: 69.52, 70: 97.33, 95: 132.09,
        120: 166.85, 150: 208.56, 185: 257.22, 240: 333.70
    }

    def __init__(self, Vn, turbines, paths):
        self.Vn = Vn
        self.Tr = turbines
        self.paths = paths
        self.cables = []
        self.Pjtot = 0
        self.Ctot = 0
        self._build()
        self._uniform_section()
        self._losses()
        self._cost()

    def _build(self):
        self.cables = []
        for path in self.paths:
            P = 0
            for i in range(len(path)-1):
                a, b = path[i], path[i+1]
                P += self.Tr[a].P
                d = calculate_distance(
                    self.Tr[a].x, self.Tr[a].y,
                    self.Tr[b].x, self.Tr[b].y
                )
                self.cables.append(Cable(d, self.Vn, P))

    def _uniform_section(self):
        Amax = max(c.A_continuous for c in self.cables)

        chosen = max(Cable.SECTION_TABLE)  # fallback seguro
        for sec in sorted(Cable.SECTION_TABLE):
            if sec >= Amax:
                chosen = sec
                break

        for c in self.cables:
            c.assign_section(chosen)

    def _losses(self):
        self.Pjtot = sum(c.Pj for c in self.cables)

    def _cost(self):
        sec = self.cables[0].A
        cost_m = self.INDUSTRIAL_CABLE_COSTS[sec]
        self.Ctot = sum(c.lc * cost_m for c in self.cables)

# =========================================================
# HEURÍSTICA PLANAR (SEM CRUZAMENTO)
# =========================================================

def build_planar_strings(coordinates, sub_idx, n_strings):
    coords = coordinates
    center = coords[sub_idx]

    angles = np.arctan2(
        coords[:,1] - center[1],
        coords[:,0] - center[0]
    )

    turbines = [i for i in range(len(coords)) if i != sub_idx]
    turbines.sort(key=lambda i: angles[i])

    strings = np.array_split(turbines, n_strings)
    return [list(s) for s in strings]

def order_string_radial(group, coords, sub_idx):
    return sorted(
        group,
        key=lambda i: np.linalg.norm(coords[i] - coords[sub_idx]),
        reverse=True
    )

# =========================================================
# PIPELINE COMPLETA
# =========================================================

def analisar_layout_completo(coords, sub_idx, n_strings=8,
                             Vn=33e3, P_turbina=3.35e6):

    groups = build_planar_strings(coords, sub_idx, n_strings)

    paths = []
    for g in groups:
        ordered = order_string_radial(g, coords, sub_idx)
        paths.append(ordered + [sub_idx])

    turbines = [Turbine(P_turbina, x, y) for x, y in coords]
    plant = Plant(Vn, turbines, paths)

    return plant, {
        "custo_total": plant.Ctot,
        "perda_total_W": plant.Pjtot
    }

# =========================================================
# PLOT
# =========================================================

def plot_layout(plant, coords, sub_idx):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10.colors

    for i, path in enumerate(plant.paths):
        x = [coords[n][0] for n in path]
        y = [coords[n][1] for n in path]
        plt.plot(x, y, '-o', color=colors[i%10], label=f'String {i+1}')

    plt.scatter(coords[sub_idx][0], coords[sub_idx][1],
                s=300, c='gold', marker='*', label='Substation')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# =========================================================
# TESTE
# =========================================================

if __name__ == "__main__":
    x_opt = [661.0802488476509, 1028.5071476706316, 508.96385923664684, -811.2093133111011, -568.8308215348413, 135.25320530398835, 1485.0472312776456, 1730.1250520990923, 646.5554140208285, -199.42376448479095, -951.9509284418007, -1507.1615974960082, -1633.9909479438822, -1019.2818235431666, -697.4332126339438, -9.574190686191963, 871.7251315158168, 1595.0527644567876, 2456.803622909096, 1872.184838009039, 1996.8734031880808, 1210.9016644860078, 275.7769809516732, -549.3369238438046, -1043.5460752583974, -1634.7387307270328, -2099.842348708364, -2273.785647659477, -2398.041377663485, -1559.36963751651, -1162.3466840457904, -320.7091804891101, 509.84634451260695, 1326.450406770889, 1701.2674238845875, 2127.5522847938364, 2987.7405455505045, 2855.3512539844323, 2718.252786650593, 2285.6660195388986, 1857.8551884527724, 1392.4968740426127, 787.4717401929862, 137.43093561094906, -883.513574615608, -1161.334709046542, -1836.7986360836767, -2535.152451105885, -2655.315492118568, -2767.015069397514, -2867.3249981604527, -2976.727159156992, -2557.9521380151386, -2196.4428749717517, -1832.5764298269705, -1332.5418611408882, -482.1124677414058, -187.26001448925123, 643.5174096131757, 1121.401691010576, 1814.909194566765, 2336.8564045169005, 2731.7185913993453, 2926.420801140257]
    y_opt = [45.596410184418396, -102.84602376647021, 807.3970577337197, 371.58736397695736, -614.669773766668, -770.1489622929139, 232.24072673593452, 625.1181844633061, 1458.0268983660428, 1285.1783613061, 1125.1660557112616, 793.4735755352228, 223.32673728296464, -915.2891031021514, -1417.8546962289251, -1544.680665315384, -967.8222111218745, -472.52660284680076, 88.53209187034501, 1071.154478230013, 1684.9560690782678, 2204.5089768888347, 2025.8720291878376, 2520.185778550141, 1851.0238946485588, 1476.242136149511, 953.884503172178, -204.64533466558544, -760.1295456700283, -1262.8861935451305, -1858.0034535025784, -2030.4682372594812, -2183.7635495548434, -1687.768977244719, -1425.9208202323532, -828.0335781258101, -270.487321948783, 408.4430996270802, 1241.3588655768012, 1941.1338020272867, 2355.491613240731, 2657.2281471571628, 2835.123001089535, 2983.760720328497, 2648.145616002615, 2766.083599057293, 2364.8884384732605, 1604.047221383611, 1331.4834111951925, 535.9495707647394, -63.18170584095447, -372.9241616847378, -1096.6510703643166, -1689.189485811573, -2347.7758236565655, -2687.8025040381153, -2880.949347604215, -2994.090617101495, -2790.3310807019598, -2548.4676073629403, -2388.700552838015, -1868.8530123801156, -1239.9438061023466, -659.9615826554246]


    coords = np.column_stack((x_opt, y_opt))
    sub = 35

    plant, res = analisar_layout_completo(coords, sub)
    print(json.dumps(res, indent=2))
    plot_layout(plant, coords, sub)
