import matplotlib.pyplot as plt
import math


class Cable:
    """
    Cable connecting two turbines
        Inputs:
            lc: cable length in meters
            Vn: Rated voltage in Volts
            Pn: Active power transmitted by the cable in Watts
            Qi: Reactive power at the cable input (turbine) in VAr
    """
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.I = self.Pn / (math.sqrt(3) * self.Vn) # corrent do cabo
        self.A = self.I/self.dI  # cross-section of the cable
        self.Pj = self.__Pj__()
        self.Ctot = self.__Ctot__()

    p = 0.0173e-6   # Ohm m2/m resistividade do cobre recozido normal à 20°C
    alpha = 0.393   # coeficiente de variação da resistividade do cobre
    p90 = p*(1 + alpha*(90-20))  # Ohm m2/m resistividade do cobre recozido normal à 90°C
    dI = 2.3        # A/mm2,  densidade de corrente média entre 880 e 1400 A (https://flexcopper.com.br/tabela-de-densidade-de-corrente) ! Precisa melhorar, norma IEC 60287 !
    A = 0
    # R = 31.7e-6  # Ohm/m at 90ºC from AC 132-kV 3phase 800 mm2 submarine cable, rated power 50 MW  (Brakelmann 2003)
    C = 2e3  # custo em R$/m (Perez-Rua 2019: 282 £/m)
    #TODO: refine this calculation:
    #   - Size the cable cross-section in function of Pn (norma IEC)
    #   - include Q
    #   - the effect of current and voltage propagation along the cable
    def __Pj__(self):
        self.Pj = 3 * self.lc * self.A * self.p90 * self.I ** 2
        #self.Pj = 3 * self.lc * self.R * (self.Pn/(math.sqrt(3)*self.Vn)) ** 2
        return self.Pj

    def __Ctot__(self):
        self.Ctot = self.lc * self.C
        return self.Ctot



class Turbine:
    """
    Turbine in the power plant
        Inputs:
            Pt: rated power in W
            x: x position in m
            y: y position in m
    """
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y


class Plant:
    """
        Offshore power plants
            Variables:
                Tr: array of turbines in the plant
                Cb: array of cables in the plant
    """
    def __init__(self, Vn, Tr, paths):
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()
        return
    Cb = []
    cables_flat = []
    Pjtot = 0
    Ctot = 0

    def lay_cables(self):
        self.Cb = []
        for m in self.paths:
            cable = []
            Ptransmitted = 0
            for k, n in enumerate(m[0:-1]):
                length = calculate_distance(self.Tr[n].x, self.Tr[n].y, self.Tr[m[k+1]].x, self.Tr[m[k+1]].y)
                Ptransmitted = Ptransmitted + self.Tr[n].P
                cable.append(Cable(lc=length, Vn=self.Vn, Pn=Ptransmitted))

            #print(f"path: {m}")
            #print(f"Number of cables: {len(cable)}")
            self.Cb.append(cable)

        #print(f"Number layed cable of paths: {len(self.Cb)}")
        self.cables_flat = [item for sublist in self.Cb for item in sublist]
        return

    def calculate_losses(self):
        self.Pjtot = 0
        for c in self.cables_flat:
            self.Pjtot = self.Pjtot + c.Pj

    def calculate_cost(self):
        self.Ctot = 0
        for c in self.cables_flat:
            self.Ctot = self.Ctot + c.Ctot

    def print_turbines(self):
        print(f"Number of turbines in the plant: {len(self.Tr)}")
        print(f"Details:     \t  P (MW)  \t  x(m)    \t  y(m)")
        for k, T in enumerate(self.Tr):
            print(f"{k}     \t  {T.P/1e6}  \t  {T.x}    \t  {T.y}")
        return

    def print_cables(self):
        print(f"Number of cables in the plant: {len(self.cables_flat)}")
        print(f"Connecting turbines \t length(m)  \t  P transmitted (MW) \t P_J (kW)    \t  Cost (R$)")
        for j, path in enumerate(self.Cb):
            for k, cable in enumerate(path):
                print(f"{self.paths[j][k]} and {self.paths[j][k+1]}      \t {cable.lc}       \t  {cable.Pn / 1e6}  \t \t  {cable.Pj / 1e3}  \t  {cable.Ctot} ")
        return

# Function to calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def plot_coordinates(x, y, title="coordinates", color='blue'):
    plt.scatter(x, y, color=color)
    # Adding labels for each point
    for i in range(len(y)):
        plt.text(x[i] + 0.1, y[i] + 0.1, f'({i})')
    # Label the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    # Title and grid
    plt.title(title)
    plt.grid(True)


####################################
# Plotting the example plants

# Coordinates of inicial position:
x_ini = [0., 650., 200.861, -525.861, -525.861, 200.861, 1300., 1051.7221, 401.7221, -401.7221, -1051.7221,
         -1300., -1051.7221, -401.7221, 401.7221, 1051.7221]
y_ini = [0., 0., 618.1867, 382.0604, -382.0604, -618.1867, 0., 764.1208, 1236.3735, 1236.3735, 764.1208, 0.,
         -764.1208, -1236.3735, -1236.3735, -764.1208]

# Coordinates of optimized position:
x_opt = [456.7204089099218, 438.3919220444296, -399.92448768287204, -520.3471492587372, -406.78323514290537,
         116.0430277199342, 1299.9147666148804, 935.4797522168615, 580.9914866745228, -288.6249657521407,
         -1091.8161063628386, -984.6468715598825, -796.0997345716727, -222.36747893782228, 639.3216959582929,
         991.2686388396767]
y_opt = [188.21833363635656, -207.8664662389235, 872.844372473487, 350.4894749171311, -381.96064772293386,
         -749.2747718669738, -14.886219683036972, 536.1257300488144, 1080.69374416935, 1267.5549807167085,
         705.6469300480154, 2.399217995711785, -926.1619079629315, -1280.8406240861154, -1118.7085204533382,
         -576.566173271566]

# Coordinates of circle:
radius = 1300
angle = 2*math.pi/16
x_c = []
y_c = []
for i in range(16):
    x_c.append(radius*math.cos(angle*i))
    y_c.append(radius*math.sin(angle*i))

# Plotting the points
plt.figure(1)
plot_coordinates(x_ini, y_ini, title='Inicial position of wind turbines within the plant')

plt.figure(2)
plot_coordinates(x_opt, y_opt, title='Optimized position of wind turbines within the plant')

plt.figure(3)
plot_coordinates(x_c, y_c, title='Circle of wind turbines within the plant')

############
# DEFINE OFFSHORE PLANTS
#
# Rated voltage:
Vn = 30e3
#
# INICIAL:
# Turbines:
Tr_ini = []
for i in range(16):
    Prated = 3.35e6
    Tr_ini.append(Turbine(Prated, x_ini[i], y_ini[i]))
# Graph:
# connection o the shore: from 13
# paths:
path1_ini = [10, 11, 12, 13]
path2_ini = [9, 3, 4, 13]
path3_ini = [8, 2, 0, 1, 5, 13]
path4_ini = [7, 6, 15, 14, 13]
path_ini = [path1_ini, path2_ini, path3_ini, path4_ini]

plant_ini = Plant(Vn, Tr_ini, path_ini)
print("\n INICIAL")
plant_ini.print_turbines()
plant_ini.print_cables()

# OPTIMIZED
# Turbines:
Tr_opt = []
for i in range(16):
    Prated = 3.35e6
    Tr_opt.append(Turbine(Prated, x_opt[i], y_opt[i]))
# Optimized graph:
# connection to the shore: from 13
# paths:
path1_opt = [10, 11, 12, 13]
path2_opt = [9, 2, 3, 4, 13]
path3_opt = [8, 0, 1, 5, 13]
path4_opt = [7, 6, 15, 14, 13]
path_opt = [path1_opt, path2_opt, path3_opt, path4_opt]

plant_opt = Plant(Vn, Tr_opt, path_opt)
print("\n OPTIMIZED")
plant_opt.print_turbines()
plant_opt.print_cables()

# CIRCLE
# Turbines:
Tr_c = []
for i in range(16):
    Prated = 3.35e6
    Tr_c.append(Turbine(Prated, x_c[i], y_c[i]))
# Circle graph:
# connection o the shore: from 13
# paths:
path1_c = [5, 4, 3, 2, 1, 0, 15, 14, 13]
path2_c = [6, 7, 8, 9, 10, 11, 12, 13]
path_c = [path1_c, path2_c]

plant_c = Plant(Vn, Tr_c, path_c)
print("\n CIRCLE")
plant_c.print_turbines()
plant_c.print_cables()


##################################
# COMPARISONS

# Losses
print("\n\t\t Joule losses on cabling (kW)  \t custo em um ano com 0.3 R$/kWh (millions of R$)")
print(f"Inicial: {plant_ini.Pjtot/1e3}  \t \t {plant_ini.Pjtot / 1e3 * 0.3 * 24 * 365 / 1e6}")
print(f"Optimal: {plant_opt.Pjtot/1e3}  \t \t {plant_opt.Pjtot / 1e3 * 0.3 * 24 * 365 / 1e6} ")
print(f"Circle: {plant_c.Pjtot/1e3}     \t \t {plant_c.Pjtot / 1e3 * 0.3 * 24 * 365 / 1e6}")

# Cost
print("\n\t\t Estimated cost of cabling (millions of R$)")
print(f"Inicial: {plant_ini.Ctot/1e6}")
print(f"Optimal: {plant_opt.Ctot/1e6}")
print(f"Circle: {plant_c.Ctot/1e6}")


# Show the plot
plt.show()

def plot_optimized_paths(x_opt, y_opt, paths):
    plt.figure(figsize=(12, 12))
    substation_pos = (x_opt[13], y_opt[13])
    
    # Plotar todos os pontos
    plt.scatter(x_opt, y_opt, c='gray', s=50, label='Torres')
    plt.scatter(*substation_pos, c='k', marker='s', s=150, label='Subestação (13)')
    
    # Plotar caminhos
    colors = ['red', 'blue', 'green', 'orange']
    for i, path in enumerate([path1_opt, path2_opt, path3_opt, path4_opt]):
        x_path = [x_opt[node] for node in path]
        y_path = [y_opt[node] for node in path]
        plt.plot(x_path, y_path, color=colors[i], linewidth=2, marker='o', label=f'String {i+1}')
    
    plt.title("Conexões Otimizadas das Strings à Subestação")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

plot_optimized_paths(x_opt, y_opt, [path1_opt, path2_opt, path3_opt, path4_opt])