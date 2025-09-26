import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class WindFarm:
    # O componente "ferramenta de visualização" não considera produção de
    # energia, o componente "simulador" do framework proposto pelo LAIA faz isso.
    def __init__(self, turbine_coords, wind_direction=270.0,
                 wind_speed_free_stream=10.59,
                 turbine_diameter=240, wake_k=0.0324555, ct_coeff=8/9):
        self.original_positions = np.array(turbine_coords)
        self.n_turbines = len(turbine_coords)
        self.wind_direction = wind_direction
        self.U_inf = wind_speed_free_stream
        self.D = turbine_diameter
        self.k_y = wake_k
        self.CT = ct_coeff
        self.turbine_velocities = {}

    # Rotaciona o sistema de coordenadas para alinhar a direção do vento com o
    # eixo X. Isso simplifica os cálculos, pois o modelo de esteira pode
    # assumir que o fluxo é sempre da esquerda para a direita.
    def _rotate_coordinates(self, coords):
        angle_rad = np.radians(self.wind_direction - 270.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = coords[:, 0] * cos_a - coords[:, 1] * sin_a
        y_rot = coords[:, 0] * sin_a + coords[:, 1] * cos_a
        return np.column_stack((x_rot, y_rot))


    # Calcula o déficit de velocidade (perda percentual) em um ponto (x, y)
    # causado pela esteira de uma única turbina localizada em (x0, y0),
    # usando o modelo analítico de Bastankhah.
    # Retorna 0 se o ponto não estiver a jusante da turbina.
    def _bastankhah_wake_deficit(self, x, y, x0, y0):
        d = x - x0
        is_downstream = d > 0
        sigma_y = np.full_like(d, np.inf)
        sigma_y[is_downstream] = self.k_y * d[is_downstream] + self.D/np.sqrt(8)
        radical_term = 1 - self.CT / (8 * (sigma_y / self.D)**2)
        radical_term = np.maximum(0, radical_term)
        exponent = -0.5 * ((y - y0) / sigma_y)**2
        deficit = (1 - np.sqrt(radical_term)) * np.exp(exponent)
        return np.where(is_downstream, deficit, 0.0)

    # Método que soma a contribuição da perda de velocidade de todas as
    # turbinas a jusante de uma determinada turbina.
    # Usa o método de soma quadrática proposto por Katic para estimar a
    # contribuição de todas as turbinas.
    def calculate_wake_effects(self):
        rotated_turbine_pos = self._rotate_coordinates(self.original_positions)
        velocities = []
        for i in range(self.n_turbines):
            xi, yi = rotated_turbine_pos[i]
            sum_deficits_sq = 0.0
            for j in range(self.n_turbines):
                if i == j: continue
                xj, yj = rotated_turbine_pos[j]
                deficit = self._bastankhah_wake_deficit(xi, yi, xj, yj)
                sum_deficits_sq += deficit**2
            total_deficit = min(np.sqrt(sum_deficits_sq), 1.0)
            velocities.append(self.U_inf * (1 - total_deficit))
        self.turbine_velocities = dict(zip(map(tuple, self.original_positions),
                                           velocities))


    # Gera uma malha (meshgrid) cobrindo a área do parque e calcula a
    # velocidade do vento em cada ponto.
    # O cálculo é feito aplicando o modelo de Bastankhah para cada turbina e
    # combinando os déficits de esteira usando o método da soma quadrática.
    def get_velocity_field(self, resolution=300):
        buffer = self.D * 4
        x_min, y_min = self.original_positions.min(axis=0) - buffer
        x_max, y_max = self.original_positions.max(axis=0) + buffer
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(grid_x, grid_y)
        grid_coords = np.vstack([X.ravel(), Y.ravel()]).T
        rotated_grid_coords = self._rotate_coordinates(grid_coords)
        rotated_turbine_pos = self._rotate_coordinates(self.original_positions)
        total_deficit_sq = np.zeros(resolution * resolution)
        for xj_rot, yj_rot in rotated_turbine_pos:
            deficit = self._bastankhah_wake_deficit(
                rotated_grid_coords[:, 0], rotated_grid_coords[:, 1],
                xj_rot, yj_rot)
            total_deficit_sq += deficit**2
        velocity_field = self.U_inf * (1 - np.sqrt(np.minimum(total_deficit_sq,
                                                              1.0)))
        return X, Y, velocity_field.reshape(resolution, resolution)

    # Retorna uma tabela com as velocidade dos ventos em cada turbina.
    def summarize_results(self):
        if not self.turbine_velocities: self.calculate_wake_effects()
        print("-"*75)
        print(f"Wind direction: {self.wind_direction}° | \
              Freestream velocity: {self.U_inf:.2f} m/s")
        print("-" * 75)
        print(f"{'Turbine #':<12} {'Position (x, y)':<25} \
              {'Velocity (m/s)':<20}")
        print("-" * 75)
        for i, pos in enumerate(self.original_positions):
            vel = self.turbine_velocities[tuple(pos)]
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            print(f"T{i+1:<11} {pos_str:<25} {vel:<20.4f}")
        print("-" * 75)


    # Desenha uma turbina para visualização
    def _draw_turbine(self, ax, x_pos, y_pos, turbine_id=None, velocity=None,
                      with_text=False, fontsize=8):
        rotor_angle_rad = np.radians(270 - self.wind_direction + 90)
        radius = self.D / 2

        dx = radius * np.cos(rotor_angle_rad)
        dy = radius * np.sin(rotor_angle_rad)

        ax.plot([x_pos - dx, x_pos + dx], [y_pos - dy, y_pos + dy],
                color='black', linewidth=2.5, zorder=6)

        ax.scatter(x_pos, y_pos, s=30, c='white', edgecolor='black', zorder=7)

        if with_text and turbine_id is not None and velocity is not None:
            label_text = f"T{turbine_id}\n{velocity:.2f} m/s"
            offset_angle_rad = np.radians(270 - self.wind_direction + 90)
            offset_distance = radius * 1.3
            offset_x = offset_distance * np.cos(offset_angle_rad)
            offset_y = offset_distance * np.sin(offset_angle_rad)
            ha = 'left' if offset_x >= 0 else 'right'
            va = 'bottom' if offset_y >= 0 else 'top'
            ax.text(x_pos + offset_x, y_pos + offset_y, label_text, ha=ha,
                    va=va, fontsize=fontsize, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6,
                              edgecolor='none', pad=1), zorder=8)

    # Plota o campo de velocidade sob efeito esteira
    def plot_layout_with_wake_field(self, title=None, save_path=None):
        if not self.turbine_velocities: self.calculate_wake_effects()

        # ==========================================
        # PAINEL DE CONTROLE DE ESTILO DO GRÁFICO
        # Manipule todos os tamanhos de fonte aqui
        # ==========================================
        base = 30
        font_config = {
            'title': base*1,
            'axis_label': base*0.9,
            'tick_label': base*0.7,
            'turbine_label': 13,  # Legenda T1, T2...
            'wind_label': base*0.6,
            'cbar_label': base*0.5
        }
        # ==========================================

        X, Y, V = self.get_velocity_field()

        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        aspect_ratio = y_range / x_range if x_range != 0 else 1

        fig_width = 14
        fig_height = fig_width * aspect_ratio
        if fig_height > 14:
            fig_height = 14
            fig_width = fig_height / aspect_ratio

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cmap_reversed = 'coolwarm_r'
        contour = ax.contourf(X, Y, V, levels=100, cmap=cmap_reversed, zorder=1)

        for i, pos in enumerate(self.original_positions):
            vel = self.turbine_velocities[tuple(pos)]
            self._draw_turbine(ax, pos[0], pos[1], turbine_id=i+1,
                               velocity=vel,
                               with_text=True,
                               fontsize=font_config['turbine_label'])

        flow_angle_rad = np.radians(270 - self.wind_direction)
        ax.arrow(0.05, 0.9, 0.06 * np.cos(flow_angle_rad),
                 0.06 * np.sin(flow_angle_rad),
                 transform=ax.transAxes, width=0.008,
                 head_width=0.025, head_length=0.018,
                 fc='black', ec='black', zorder=10)
        ax.text(0.05, 0.95, 'Vento', transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=font_config['wind_label'], fontweight='bold')

        if title is None:
            title_text = f'Campo de Velocidade (Direção: {self.wind_direction}°)'
        else:
            title_text = title
        ax.set_title(title_text, fontsize=font_config['title'], weight='bold')

        #ax.set_xlabel('X coordinate [m]', fontsize=font_config['axis_label'])
        ax.set_xlabel('Coordenada X [m]', fontsize=font_config['axis_label'])
        #ax.set_ylabel('Y coordinate [m]', fontsize=font_config['axis_label'])
        ax.set_ylabel('Coordenada Y [m]', fontsize=font_config['axis_label'])
        ax.tick_params(axis='both', which='major',
                       labelsize=font_config['tick_label'])

        ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(contour, ax=ax, orientation='vertical',
                            pad=0.02, shrink=0.855)
        cbar.set_label('Velocidade do Vento [m/s]',
                       fontsize=font_config['cbar_label'], weight='bold')
        cbar.ax.tick_params(labelsize=font_config['tick_label'])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":

    coord_x= [7.600652423822722, 3116.7628945630076, 6251.08377862495,
    9955.234924528468, 12220.878621983331, 14753.071739705838,
    16364.074191108346,
    17806.783788593042, 18770.52232227931, 20464.88627757359,
    22074.646650558403, 21425.71289640738, 18017.157871035764,
    14539.437028707142,
    12172.435748866577, 8554.6772392993, 7522.570117677625, 5759.437463164007,
    4319.018266672723, 2672.765734916511, 1174.0174118899406,
    1778.2669709625022, 4624.829199187698, 3711.6303547565985,
    5176.115456817468, 6557.61103926695, 5734.0598993438, 7561.1819887301,
    6919.730430097865, 7289.6856460625695, 8454.590700053164, 7916.800006188747,
    9321.771057084483, 8299.08912551373, 8971.388737801544,
    9634.798405886977, 10898.486336196864, 11344.062160443982, 10615.7854492212,
    10146.963465690995, 11711.774340453376, 10968.05425235548,
    13201.343583421796, 12746.77002317665, 13488.32943151204, 12861.98824697515,
    14113.493859643517, 12562.734706804149, 15037.0589222853714,
    15666.381037291807, 14319.422645858598, 16057.109394744397, 15123.188832354,
    15421.818769771877, 17341.77042853011, 16970.249923801817,
    18324.897851394944, 17643.67593095796, 19564.297874410982, 18596.4148513817]
    coord_y= [14.732611426795984, 343.866378530809, 682.5535699890619,
    0.0007435589174407535, 377.4777790528864, 739.7737607369219, 3910.69116116975,
    6420.93945386492, 9606.259444255254, 11890.770373087307,
    14708.133571280992, 15728.486336957276, 15999.980548868123, 15693.364648301,
    15999.998244615332, 15553.77551325293, 13812.646966172913,
    10706.659729081883, 8146.502927266327, 4833.510051503408, 1998.979481569771,
    1185.6540780885232, 2786.9584575414183, 4068.9203702775076,
    6594.952098625447, 2335.9207285026823, 4499.687526642956, 5962.316488137563,
    9864.029858240096, 11773.150743916593, 1011.1949437589434,
    3224.8463426863154, 5587.334186847504, 8913.535538935479, 11455.46784985946,
    15298.792392066802, 1380.216349499651, 5138.490727279522,
    7565.253302485559, 10059.635592517368, 11102.03205957391, 14285.39480581248,
    1839.5551205411628, 4114.881544019736, 8303.578105559098,
    10351.0136991541, 12503.35951515289, 14561.66865213871, 2182.4932539490383,
    4884.94744695806, 6859.975270006579, 7957.078887063167,
    10674.360553553412, 14266.403321339176, 7159.176138675647, 9245.96824341844,
    11501.06595005533, 13671.570259100376, 13268.346538175194,
        15087.297636759702]

    turbine_coordinates = list(zip(coord_x, coord_y))

    wind_direction = 22.0
    turbine_diameter = 240
    wind_speed_free_stream = 10

    farm = WindFarm(turbine_coordinates, wind_direction=wind_direction,
                    turbine_diameter=turbine_diameter,
                    wind_speed_free_stream=wind_speed_free_stream)

    farm.summarize_results()
    farm.plot_layout_with_wake_field()