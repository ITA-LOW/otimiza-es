import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
import math
from iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =============================================
# Classes para modelagem de cabos e usina
# =============================================
class Cable:
    def __init__(self, lc, Vn, Pn, Qi=0):
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.Qi = Qi
        self.I = self.Pn / (math.sqrt(3) * self.Vn) if self.Pn > 0 else 0.0
        self.dI = 2.3
        self.A = self.I / self.dI if self.I > 0 else 0.0
        if self.A > 0 and self.A < 50: # Seção mínima de 50mm²
            self.A = 50.0

        self.p = 0.0173e-6  # resistividade do cobre a 20°C (ohm.m)
        self.alpha = 0.00393 # coeficiente de temperatura do cobre (1/°C)
        self.p90 = self.p * (1 + self.alpha * (90 - 20)) # resistividade a 90°C
        self.C = 2e3  # Custo por metro (exemplo, R$/m)
        self.Pj = self._calc_pj()
        self.Ctot = self._calc_ctot()

    def _calc_pj(self):
        if self.I == 0 or self.A == 0:
            return 0.0
        # R = self.p90 * self.lc / (self.A * 1e-6) # Resistência do cabo (ohm)
        # Pj = 3 * (self.I ** 2) * R (W)
        return 3 * (self.I ** 2) * self.p90 * self.lc / (self.A * 1e-6)

    def _calc_ctot(self):
        return self.lc * self.C

class Turbine:
    def __init__(self, Pt, x, y):
        self.P = Pt # Potência nominal (W)
        self.x = x  # Coordenada x (m)
        self.y = y  # Coordenada y (m)

class Plant:
    def __init__(self, Vn, turbines_list_obj, paths_list_indices):
        self.Vn = Vn # Tensão nominal da rede interna (V)
        self.turbines_obj_list = turbines_list_obj
        self.paths_idx_list = paths_list_indices
        self.Cb = []
        self.cables_flat = []
        self.cable_map = {} # Opcional, para fácil acesso aos cabos por par de nós
        self.Pjtot = 0 # Perdas Joule totais (W)
        self.Ctot = 0  # Custo total dos cabos (R$)

        self.lay_cables()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        self.Cb = []
        self.cables_flat = []
        self.cable_map = {}

        for path_indices in self.paths_idx_list:
            if not path_indices or len(path_indices) < 2: # Precisa de pelo menos 2 nós para um cabo
                continue

            cable_path_objects = []
            # path_indices é uma lista de índices de nós, e.g., [t_distante, ..., t_prox_sub, sub_idx]
            for i in range(len(path_indices) - 1):
                node_from_idx = path_indices[i]
                node_to_idx = path_indices[i+1]

                # Calcula a potência transmitida por este cabo:
                # Soma das potências de todas as turbinas "a montante" neste string,
                # ou seja, todas as turbinas desde o início do string até node_from_idx.
                Pn_for_this_cable = 0
                for k in range(i + 1): 
                    turbine_in_string_idx = path_indices[k]
                    # A subestação (se incluída nos paths como um nó Turbine com P=0) não contribui.
                    if self.turbines_obj_list[turbine_in_string_idx].P > 0:
                         Pn_for_this_cable += self.turbines_obj_list[turbine_in_string_idx].P
                
                # Coordenadas dos nós para calcular o comprimento do cabo
                x1, y1 = self.turbines_obj_list[node_from_idx].x, self.turbines_obj_list[node_from_idx].y
                x2, y2 = self.turbines_obj_list[node_to_idx].x, self.turbines_obj_list[node_to_idx].y
                length = math.hypot(x2 - x1, y2 - y1) # Distância Euclidiana

                cable = Cable(lc=length, Vn=self.Vn, Pn=Pn_for_this_cable)
                cable_path_objects.append(cable)
                self.cable_map[(node_from_idx, node_to_idx)] = cable # Opcional
            
            self.Cb.append(cable_path_objects)
        self.cables_flat = [cable for path_obj_list in self.Cb for cable in path_obj_list]

    def calculate_losses(self):
        self.Pjtot = sum(cable.Pj for cable in self.cables_flat)

    def calculate_cost(self):
        self.Ctot = sum(cable.Ctot for cable in self.cables_flat)

# =============================================
# Funções de agrupamento e roteamento
# =============================================
def agrupar_torres_por_cosseno(coordenadas_all_nodes_np, idx_subestacao_no_array, n_grupos_desejados_explicitamente):
    vetores_para_sub = coordenadas_all_nodes_np - coordenadas_all_nodes_np[idx_subestacao_no_array]
    mascara_turbinas = np.arange(len(coordenadas_all_nodes_np)) != idx_subestacao_no_array
    
    vetores_turbinas_apenas = vetores_para_sub[mascara_turbinas]
    
    if len(vetores_turbinas_apenas) == 0: return []

    n_grupos_real_para_kmeans = min(n_grupos_desejados_explicitamente, len(vetores_turbinas_apenas))
    if n_grupos_real_para_kmeans == 0: return []

    matriz_similaridade_cosseno = cosine_similarity(vetores_turbinas_apenas)
    
    if n_grupos_real_para_kmeans > len(matriz_similaridade_cosseno):
        n_grupos_real_para_kmeans = len(matriz_similaridade_cosseno) # n_clusters não pode ser > n_samples
        if n_grupos_real_para_kmeans == 0: return []

    kmeans = KMeans(n_clusters=n_grupos_real_para_kmeans, random_state=42, n_init='auto')
    labels_grupos_kmeans = kmeans.fit_predict(matriz_similaridade_cosseno)
    
    indices_originais_das_turbinas = np.where(mascara_turbinas)[0]
    
    grupos_de_indices_originais = []
    for id_g in range(n_grupos_real_para_kmeans):
        indices_locais_no_grupo = np.where(labels_grupos_kmeans == id_g)[0]
        indices_globais_no_grupo = [indices_originais_das_turbinas[i] for i in indices_locais_no_grupo]
        if indices_globais_no_grupo:
             grupos_de_indices_originais.append(sorted(indices_globais_no_grupo))
    
    return grupos_de_indices_originais

def balancear_grupos(grupos_originais_idx_globais, coords_todos_nos_np, idx_sub_no_array_global, alvo_tam_grupo_explicito):
    def similaridade_turbina_com_grupo(idx_turbina_global, grupo_indices_globais_atual):
        if not grupo_indices_globais_atual: return 0.0
        vetor_turbina = (coords_todos_nos_np[idx_turbina_global] - coords_todos_nos_np[idx_sub_no_array_global]).reshape(1, -1)
        sims = []
        for idx_membro_global in grupo_indices_globais_atual:
            if idx_membro_global == idx_turbina_global: continue # Não comparar consigo mesma
            vetor_membro = (coords_todos_nos_np[idx_membro_global] - coords_todos_nos_np[idx_sub_no_array_global]).reshape(1, -1)
            sims.append(cosine_similarity(vetor_turbina, vetor_membro)[0][0])
        return np.mean(sims) if sims else 0.0 # Média das similaridades, ou 0 se não há com quem comparar

    grupos_balanceados = [list(g) for g in grupos_originais_idx_globais if g] # Copia e remove vazios
    if not grupos_balanceados: return []

    for iter_count in range(30): # Número de iterações para tentar balancear
        tamanhos_atuais = [len(g) for g in grupos_balanceados]
        
        terminou_balanceamento = True
        # Condição de parada: nenhum grupo excede o alvo E (nenhum grupo está abaixo do alvo OU os que estão abaixo não podem receber mais)
        for i, tamanho_g_i in enumerate(tamanhos_atuais):
            if tamanho_g_i > alvo_tam_grupo_explicito:
                terminou_balanceamento = False; break
            if tamanho_g_i < alvo_tam_grupo_explicito:
                # Existe algum outro grupo que pode doar? (i.e., > alvo_tam_grupo_explicito)
                if any(t > alvo_tam_grupo_explicito for t in tamanhos_atuais):
                    terminou_balanceamento = False; break
        if terminou_balanceamento: break
        
        # Tenta mover de grupos maiores que o alvo para grupos menores que o alvo
        for idx_g_fonte in range(len(grupos_balanceados)):
            # Itera enquanto o grupo fonte for maior que o alvo e tiver turbinas
            while len(grupos_balanceados[idx_g_fonte]) > alvo_tam_grupo_explicito and grupos_balanceados[idx_g_fonte]:
                g_fonte_atual = grupos_balanceados[idx_g_fonte]
                sims_no_g_fonte = [similaridade_turbina_com_grupo(idx_t_global, g_fonte_atual) for idx_t_global in g_fonte_atual]
                if not sims_no_g_fonte : break # Grupo fonte esvaziou inesperadamente
                
                idx_turbina_para_mover_global = g_fonte_atual.pop(np.argmin(sims_no_g_fonte))
                
                melhor_idx_g_destino = -1
                max_sim_com_destino = -float('inf')
                
                # Candidatos a destino: grupos diferentes do fonte e menores que o alvo
                indices_g_candidatos_destino = [
                    j for j, g_dest in enumerate(grupos_balanceados) 
                    if j != idx_g_fonte and len(g_dest) < alvo_tam_grupo_explicito
                ]
                
                if not indices_g_candidatos_destino: # Nenhum grupo pode receber
                    g_fonte_atual.append(idx_turbina_para_mover_global) # Devolve a turbina
                    g_fonte_atual.sort() # Mantém consistência se a ordem importa
                    break # Sai do while para este grupo fonte

                for idx_g_cand_dest in indices_g_candidatos_destino:
                    sim_com_cand_dest = similaridade_turbina_com_grupo(idx_turbina_para_mover_global, grupos_balanceados[idx_g_cand_dest])
                    if sim_com_cand_dest > max_sim_com_destino:
                        max_sim_com_destino = sim_com_cand_dest
                        melhor_idx_g_destino = idx_g_cand_dest
                
                if melhor_idx_g_destino != -1:
                    grupos_balanceados[melhor_idx_g_destino].append(idx_turbina_para_mover_global)
                    grupos_balanceados[melhor_idx_g_destino].sort()
                else: # Não encontrou bom destino, devolve
                    g_fonte_atual.append(idx_turbina_para_mover_global)
                    g_fonte_atual.sort()
                    break # Evita loop se não há para onde mover de forma benéfica
                            
    return [g for g in grupos_balanceados if g] # Remove grupos que podem ter ficado vazios

def make_route_for_ga(grupos_idx_turbinas_globais, todos_coords_np, idx_sub_no_array_global):
    paths = []
    coords_subestacao = todos_coords_np[idx_sub_no_array_global]
    for grupo_indices_globais in grupos_idx_turbinas_globais:
        if not grupo_indices_globais: continue # Pula grupo vazio
        
        distancias_e_indices = []
        for idx_turbina_global in grupo_indices_globais:
            dist = np.linalg.norm(todos_coords_np[idx_turbina_global] - coords_subestacao)
            distancias_e_indices.append((dist, idx_turbina_global))
        
        distancias_e_indices.sort(key=lambda x: x[0], reverse=True) # Mais distante primeiro
        
        path_ordenado_indices_globais = [idx_global for dist, idx_global in distancias_e_indices]
        path_ordenado_indices_globais.append(idx_sub_no_array_global) # Adiciona subestação no final
        paths.append(path_ordenado_indices_globais)
    return paths

# =============================================
# Configuração do Algoritmo Genético
# =============================================
IND_SIZE = 16                   # Número de TURBINAS a serem otimizadas
CIRCLE_RADIUS = 1300            
N_DIAMETERS = 260               
TURB_POWER = 3.35e6             
CABLE_VOLTAGE = 33e3            

SUBSTATION_X_FIXED = -1500 
SUBSTATION_Y_FIXED = 0    
SUBSTATION_POWER = 0      

EXPLICIT_N_GROUPS = 4 
EXPLICIT_TARGET_GROUP_SIZE = 4 
# Garanta que IND_SIZE, EXPLICIT_N_GROUPS, EXPLICIT_TARGET_GROUP_SIZE façam sentido juntos.
# Por ex., se IND_SIZE=16, N_GROUPS=4, TARGET_SIZE=4 é perfeito.
# Se IND_SIZE=16, N_GROUPS=5, TARGET_SIZE=3, alguns grupos terão 3, um terá 1 (16 = 3*5 + 1).
# Ou se TARGET_SIZE=4, então 16 = 4*4, mas com 5 grupos não fecha.
# A lógica de balanceamento tentará o melhor possível.

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) 
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def create_individual_from_coordinates(initial_turbine_coords_list_of_lists):
    flat_coords = np.array(initial_turbine_coords_list_of_lists).flatten().tolist()
    return creator.Individual(flat_coords)

initial_coordinates_yaml_data, _, _ = getTurbLocYAML('iea37-ex16.yaml') 
initial_turbine_coords_for_ga = initial_coordinates_yaml_data[:IND_SIZE].tolist()

toolbox.register("individual", create_individual_from_coordinates, initial_turbine_coords_list_of_lists=initial_turbine_coords_for_ga)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2 

def enforce_circle(individual_turbine_coords_flat_list): 
    for i in range(IND_SIZE):
        x_idx, y_idx = 2*i, 2*i + 1
        x, y = individual_turbine_coords_flat_list[x_idx], individual_turbine_coords_flat_list[y_idx]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = math.atan2(y, x)
            individual_turbine_coords_flat_list[x_idx] = CIRCLE_RADIUS * math.cos(angle)
            individual_turbine_coords_flat_list[y_idx] = CIRCLE_RADIUS * math.sin(angle)

def mutate(individual_turbine_coords_flat_list, mu, sigma, indpb_gene):
    mutated_individual = False
    for i in range(len(individual_turbine_coords_flat_list)):
        if random.random() < indpb_gene: # Probabilidade de mutar cada gene (coordenada)
            individual_turbine_coords_flat_list[i] += random.gauss(mu, sigma)
            mutated_individual = True
    
    if mutated_individual: # Só reforça o círculo se houve mutação nas coordenadas
        enforce_circle(individual_turbine_coords_flat_list) # Garante que as turbinas fiquem no círculo
                                                        # enforce_circle modifica a lista no local.
    return individual_turbine_coords_flat_list, 

def mate(ind1_turb_coords_flat, ind2_turb_coords_flat, alpha=0.5):
    for i in range(2*IND_SIZE): 
        gene1 = ind1_turb_coords_flat[i]
        gene2 = ind2_turb_coords_flat[i]
        gamma = random.uniform(-alpha, 1+alpha) # Blend crossover
        val1 = (1-gamma)*gene1 + gamma*gene2
        val2 = gamma*gene1 + (1-gamma)*gene2
        ind1_turb_coords_flat[i] = val1
        ind2_turb_coords_flat[i] = val2
    # Após o crossover, garantir que os filhos respeitem as restrições do círculo
    enforce_circle(ind1_turb_coords_flat)
    enforce_circle(ind2_turb_coords_flat)
    return ind1_turb_coords_flat, ind2_turb_coords_flat

toolbox.register("mate", mate, alpha=0.5)
toolbox.register("mutate", mutate, mu=0, sigma=150, indpb_gene=0.1) # indpb_gene: prob de mutar cada gene
toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_multiobjetivo(individual_turbine_coords_flat,
                           turb_atrbt_data=getTurbAtrbtYAML("iea37-335mw.yaml"),
                           wind_rose_data=getWindRoseYAML("iea37-windrose.yaml")):
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data
    
    turb_coords_only_np = np.array(individual_turbine_coords_flat).reshape((IND_SIZE, 2))

    mask_inside = (turb_coords_only_np[:,0]**2 + turb_coords_only_np[:,1]**2) <= CIRCLE_RADIUS**2
    penalty_out_of_circle = np.sum(~mask_inside) * 1e7 
    
    if IND_SIZE > 1:
        diff = turb_coords_only_np.reshape(IND_SIZE, 1, 2) - turb_coords_only_np.reshape(1, IND_SIZE, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)
        i_indices, j_indices = np.triu_indices(IND_SIZE, k=1)
        # Penaliza se a distância entre qualquer par de turbinas for menor que N_DIAMETERS * turb_diam
        # N_DIAMETERS aqui parece ser uma distância fixa, não um multiplicador de diâmetro.
        penalty_close = np.sum(dist_matrix[i_indices, j_indices] < N_DIAMETERS) * 1e7 
    else:
        penalty_close = 0
    penalty_layout = penalty_out_of_circle + penalty_close

    aep_turbines_gross = calcAEP(turb_coords_only_np, wind_freq, wind_speed, wind_dir,
                                 turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    aep_liquido_total = np.sum(aep_turbines_gross) - penalty_layout

    cable_loss_val = 1e9 
    try:
        coords_all_nodes_for_cabling_np = np.vstack((turb_coords_only_np, 
                                                     np.array([[SUBSTATION_X_FIXED, SUBSTATION_Y_FIXED]])))
        substation_fixed_idx_in_list = IND_SIZE 

        if EXPLICIT_N_GROUPS > 0 and IND_SIZE > 0:
            grupos_cosseno_indices_globais = agrupar_torres_por_cosseno(
                coords_all_nodes_for_cabling_np, 
                substation_fixed_idx_in_list, 
                n_grupos_desejados_explicitamente=EXPLICIT_N_GROUPS
            )
        else:
            grupos_cosseno_indices_globais = []
        
        if not grupos_cosseno_indices_globais:
             grupos_balanceados_indices_globais = []
        else:
            grupos_balanceados_indices_globais = balancear_grupos(
                grupos_cosseno_indices_globais, 
                coords_all_nodes_for_cabling_np, 
                substation_fixed_idx_in_list, 
                alvo_tam_grupo_explicito=EXPLICIT_TARGET_GROUP_SIZE
            )
        
        paths_to_fixed_sub_indices_globais = make_route_for_ga(
            grupos_balanceados_indices_globais, 
            coords_all_nodes_for_cabling_np, 
            substation_fixed_idx_in_list
        )
        
        valid_paths_final = [p for p in paths_to_fixed_sub_indices_globais if p and 
                             all(0 <= node_idx <= IND_SIZE for node_idx in p)]
        
        turbines_for_plant_obj_list = []
        for i in range(IND_SIZE):
            turbines_for_plant_obj_list.append(Turbine(Pt=TURB_POWER, x=turb_coords_only_np[i,0], y=turb_coords_only_np[i,1]))
        turbines_for_plant_obj_list.append(Turbine(Pt=SUBSTATION_POWER, x=SUBSTATION_X_FIXED, y=SUBSTATION_Y_FIXED))
        
        plant_instance = Plant(CABLE_VOLTAGE, turbines_for_plant_obj_list, valid_paths_final)
        cable_loss_val = plant_instance.Pjtot
        
    except Exception as e:
        # print(f"Erro durante avaliação do indivíduo: {e}") # Descomentar para debug detalhado
        pass 

    return aep_liquido_total, -cable_loss_val # Maximiza AEP, Maximiza (-Perdas)

toolbox.register("evaluate", evaluate_multiobjetivo)

def plotar_nova_solucao(coords_todos_nos_plot_np, paths_list_plot, idx_sub_no_array_plot, titulo="Layout Otimizado"):
    plt.figure(figsize=(12, 12))
    # Lista de cores mais extensa
    cores = ['#FF0000', '#0000FF', '#008000', '#FFA500', '#800080', '#00FFFF', '#FFD700', 
             '#A52A2A', '#808080', '#000000', '#FFC0CB', '#40E0D0', '#C0C0C0', '#FF6347', 
             '#32CD32', '#8A2BE2', '#FF8C00', '#2E8B57', '#DC143C', '#F0E68C']

    for i, path in enumerate(paths_list_plot):
        if not path: continue
        cor_atual = cores[i % len(cores)]
        x_path = [coords_todos_nos_plot_np[node_idx][0] for node_idx in path]
        y_path = [coords_todos_nos_plot_np[node_idx][1] for node_idx in path]
        plt.plot(x_path, y_path, 'o-', color=cor_atual, 
                 linewidth=1.5, markersize=7, markeredgecolor='black', label=f'String {i+1}')

    turb_coords_plot = np.array([coords_todos_nos_plot_np[i] for i in range(len(coords_todos_nos_plot_np)) if i != idx_sub_no_array_plot])
    if len(turb_coords_plot) > 0:
        plt.scatter(turb_coords_plot[:, 0], turb_coords_plot[:, 1], c='blue', s=60, label='Turbinas', zorder=3, alpha=0.7)

    plt.scatter(coords_todos_nos_plot_np[idx_sub_no_array_plot, 0], coords_todos_nos_plot_np[idx_sub_no_array_plot, 1], 
                c='red', s=250, marker='*', edgecolors='black', label='Subestação Fixa', zorder=5)

    for i, (x, y) in enumerate(coords_todos_nos_plot_np):
        label_text = f"S" if i == idx_sub_no_array_plot else f"T{i}"
        plt.text(x + 25, y + 25, label_text, fontsize=9, color='black', 
                 weight='bold' if i == idx_sub_no_array_plot else 'normal',
                 bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))
        
    plt.xlabel("X (m)", fontsize=14)
    plt.ylabel("Y (m)", fontsize=14)
    plt.title(titulo, fontsize=16)
    plt.legend(loc='best', fontsize=10, framealpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("layout_otimizado_subestacao_fixa.png")
    plt.show()

def main():
    random.seed(42)
    np.random.seed(42) # Para consistência no KMeans e outras operações numpy
    start_time = time.time()
    
    using_pool = False # Default para False
    if multiprocessing.cpu_count() > 1: # Tenta usar multiprocessing se houver mais de 1 CPU
        try:
            # Necessário para Windows se for empacotado com PyInstaller, mas geralmente não para scripts diretos
            # multiprocessing.freeze_support() # Descomente se tiver problemas ao criar executáveis
            pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() -1 )) # Deixa 1 CPU livre
            toolbox.register("map", pool.map)
            using_pool = True
            print(f"Usando multiprocessing com {pool._processes} processos.")
        except RuntimeError as e:
            print(f"RuntimeError ao iniciar Multiprocessing Pool: {e}. Rodando em modo serial.")
        except Exception as e: # Captura outras exceções potenciais
            print(f"Erro genérico ao iniciar Multiprocessing Pool: {e}. Rodando em modo serial.")
    else:
        print("Rodando em modo serial (CPU count <= 1 ou Pool não iniciado).")


    pop_size = 60   # População
    ngen = 150     # Número de gerações
    cxpb = 0.9      # Probabilidade de crossover
    mutpb_ind = 0.4   # Probabilidade de um INDIVÍDUO sofrer mutação (diferente de indpb_gene)
                    # A função mutate já usa indpb_gene para cada gene. DEAP usa mutpb para o indivíduo.
                    # Vamos ajustar mutate para que o mutpb de eaSimple controle se o indivíduo é mutado.
    
    # Ajustando a função mutate para refletir mutpb do eaSimple
    # A função mutate agora só aplica mutações genéticas se chamada.
    # O eaSimple usará mutpb para decidir SE chama a função mutate.
    # O indpb_gene dentro de mutate decide a prob de cada gene ser mutado.
    toolbox.unregister("mutate") # Remove registro antigo
    toolbox.register("mutate", mutate, mu=0, sigma=100, indpb_gene=0.2) # sigma era 150, indpb_gene era 0.1


    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1) # Hall of Fame para guardar o melhor indivíduo
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Aplicar restrição do círculo à população inicial
    for ind_coords_list in pop:
        enforce_circle(ind_coords_list)

    # Algoritmo Genético Simples
    pop, logbook = algorithms.eaSimple(pop, toolbox, 
                                         cxpb=cxpb, mutpb=mutpb_ind,
                                         ngen=ngen, stats=stats, 
                                         halloffame=hof, verbose=True)
    
    if using_pool and pool is not None: # Verifica se pool foi definido
        pool.close()
        pool.join()
    
    if not hof:
        print("Hall of Fame está vazio. Nenhuma solução viável foi encontrada ou a otimização falhou.")
        return None, None, None

    # --- Resultados da Melhor Solução ---
    best_individual_turb_coords_flat = hof[0]
    best_turb_coords_np = np.array(best_individual_turb_coords_flat).reshape((IND_SIZE, 2))
    
    best_coords_all_nodes_for_cabling_np = np.vstack((best_turb_coords_np,
                                                      np.array([[SUBSTATION_X_FIXED, SUBSTATION_Y_FIXED]])))
    best_substation_fixed_idx_in_list = IND_SIZE

    if EXPLICIT_N_GROUPS > 0 and IND_SIZE > 0:
        grupos_cosseno_best = agrupar_torres_por_cosseno(
            best_coords_all_nodes_for_cabling_np, 
            best_substation_fixed_idx_in_list, 
            n_grupos_desejados_explicitamente=EXPLICIT_N_GROUPS
        )
    else: 
        grupos_cosseno_best = []
    
    if not grupos_cosseno_best: 
        grupos_balanceados_best = []
    else:
        grupos_balanceados_best = balancear_grupos(
            grupos_cosseno_best, 
            best_coords_all_nodes_for_cabling_np, 
            best_substation_fixed_idx_in_list, 
            alvo_tam_grupo_explicito=EXPLICIT_TARGET_GROUP_SIZE
        )
    
    best_paths_list_final = make_route_for_ga(
        grupos_balanceados_best, 
        best_coords_all_nodes_for_cabling_np, 
        best_substation_fixed_idx_in_list
    )
    
    valid_paths_for_print_plot = [p for p in best_paths_list_final if p and 
                                  all(0 <= node_idx <= IND_SIZE for node_idx in p)]
    
    turbines_for_plant_best_obj_list = []
    for i in range(IND_SIZE):
        turbines_for_plant_best_obj_list.append(Turbine(TURB_POWER, best_turb_coords_np[i,0], best_turb_coords_np[i,1]))
    turbines_for_plant_best_obj_list.append(Turbine(SUBSTATION_POWER, SUBSTATION_X_FIXED, SUBSTATION_Y_FIXED))
            
    plant_best_instance = Plant(CABLE_VOLTAGE, turbines_for_plant_best_obj_list, valid_paths_for_print_plot)
    
    aep_val_fitness, loss_val_neg_fitness = hof[0].fitness.values
    loss_val_actual = -loss_val_neg_fitness # Perda real é positiva

    print("\n" + "="*30 + " MELHOR SOLUÇÃO ENCONTRADA " + "="*30)
    print(f"Parâmetros de Agrupamento: {EXPLICIT_N_GROUPS} Grupos, Alvo de {EXPLICIT_TARGET_GROUP_SIZE} turbinas/grupo.")
    print(f"Fitness: AEP Líquido (turbinas) = {aep_val_fitness/1e3:.2f} MWh)")
    print(f"         Perdas nos Cabos      = {loss_val_actual/1e3:.2f} kW")
    print("\nCoordenadas X (turbinas otimizadas):", best_turb_coords_np[:,0].round(2).tolist())
    print("Coordenadas Y (turbinas otimizadas):", best_turb_coords_np[:,1].round(2).tolist())
    print(f"\nSubestação Fixa em: X={SUBSTATION_X_FIXED}, Y={SUBSTATION_Y_FIXED}")
    
    print("\nStrings de cabos do melhor indivíduo (conectando à subestação fixa):")
    if not valid_paths_for_print_plot: print("  Nenhum string de cabo válido gerado.")
    else:
        for idx, path_indices in enumerate(valid_paths_for_print_plot):
            path_str = " -> ".join([f"T{n}" if n < IND_SIZE else "S_FIXA" for n in path_indices])
            print(f" String {idx+1}: {path_str}")
    
    print("\nResultados dos Cabos para a Melhor Solução:")
    print(f" Perdas Joule totais nos cabos: {plant_best_instance.Pjtot / 1e3:.3f} kW")
    print(f" Custo total estimado dos cabos: {plant_best_instance.Ctot / 1e6:.3f} Milhões (R$)") 
    
    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}min {total_time%60:.2f}s")
    print("="*80)
    
    plotar_nova_solucao(best_coords_all_nodes_for_cabling_np, 
                        valid_paths_for_print_plot, 
                        best_substation_fixed_idx_in_list,
                        titulo=f"Layout Otimizado (Sub. Fixa, {EXPLICIT_N_GROUPS} Grupos, Alvo {EXPLICIT_TARGET_GROUP_SIZE} T/Grupo)")
    
    return pop, stats, hof

if __name__ == "__main__":
    # Esta guarda é importante para o multiprocessing no Windows
    pop_result, stats_result, hof_result = main()

    # Para análise posterior, você pode querer salvar o logbook ou mais dados do hof.
    # print("\nLogbook:")
    # print(logbook)