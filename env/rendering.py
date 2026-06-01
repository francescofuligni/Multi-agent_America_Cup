import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SailingRenderer:
    def __init__(self, env):
        self.env = env
        self.fig = None
        self.ax = None

    def render_frame(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8,8))

        self.fig.clf() # Fully clear the figure to avoid frame overlapping
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, self.env.field_width)
        self.ax.set_ylim(0, self.env.field_length)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#a0d8ef')
        
        # --- PRESTART LINE (coerente con env) ---
        start_y = self.env.start_line_y

        # linea di partenza reale
        self.ax.axhline(
            start_y,
            color='yellow',
            linewidth=2,
            linestyle='--',
            alpha=0.8,
            label='Start Line'
        )

        # zona prestart (timer phase)
        if self.env.start_phase:
            # area sotto la linea dove le barche si preparano
            staging_height = 120.0  # coerente con tuo spawn (-20 / -80)

            pre_zone = patches.Rectangle(
                (0, start_y - staging_height),
                self.env.field_width,
                staging_height,
                linewidth=0,
                facecolor='yellow',
                alpha=0.08
            )
            self.ax.add_patch(pre_zone)

            self.ax.text(
                self.env.course_center_x,
                start_y - 60,
                "PRESTART PHASE",
                color='yellow',
                fontsize=9,
                fontweight='bold',
                ha='center',
                alpha=0.9
            )

        # --- Frecce vento ---
        xs, ys, us, vs = self.env.wind_field.get_grid_arrows(n_arrows=8)
        speeds = np.sqrt(us**2 + vs**2)
        self.ax.quiver(xs, ys, us, vs, speeds, cmap='Blues', alpha=0.55,
                    scale=220, width=0.003, headwidth=4, headlength=5)

        colors_map = {'red_boat': 'red', 'blue_boat': 'blue'}
        
        # --- Disegna Boundaries (Confini) ---
        self.ax.plot([self.env.boundaries['x_min'], self.env.boundaries['x_min']], [0, self.env.field_length], 'r--', linewidth=2, alpha=0.5, label='Boundary')
        self.ax.plot([self.env.boundaries['x_max'], self.env.boundaries['x_max']], [0, self.env.field_length], 'r--', linewidth=2, alpha=0.5)

        # --- Disegna Gates (Cancelli) ---
        gate_left = self.env.course_center_x - self.env.gate_width / 2.0
        gate_right = self.env.course_center_x + self.env.gate_width / 2.0
        
        # Top Gate (Bolina)
        self.ax.plot([gate_left, gate_right], [self.env.top_gate_y, self.env.top_gate_y], 'm--', alpha=0.3)
        self.ax.plot(gate_left, self.env.top_gate_y, 'mo', markersize=4, label='Gate Mark')
        self.ax.plot(gate_right, self.env.top_gate_y, 'mo', markersize=4)
        
        # Bottom Gate (Poppa)
        self.ax.plot([gate_left, gate_right], [self.env.bottom_gate_y, self.env.bottom_gate_y], 'g--', alpha=0.3)
        self.ax.plot(gate_left, self.env.bottom_gate_y, 'go', markersize=4)
        self.ax.plot(gate_right, self.env.bottom_gate_y, 'go', markersize=4)

        info_lines = []
        for idx, agent in enumerate(self.env.possible_agents):
            agent_color = colors_map.get(agent, 'black')
            
            # Traiettoria
            if agent in self.env.trajectory and len(self.env.trajectory[agent]) > 1:
                traj = np.array(self.env.trajectory[agent])
                self.ax.plot(traj[:,0], traj[:,1], '-', color=agent_color,
                            alpha=0.5, linewidth=2)

            # Barca
            if agent in self.env.state:
                bx, by = self.env.state[agent]['x'], self.env.state[agent]['y']
                hdg = self.env.state[agent]['heading']
                
                boat_size = 15
                boat_points = np.array([[boat_size,0], [-boat_size/2,boat_size/2], [-boat_size/2,-boat_size/2]])
                rot = np.array([[np.cos(hdg), -np.sin(hdg)],
                                [np.sin(hdg),  np.cos(hdg)]])
                boat_points = boat_points @ rot.T + np.array([bx, by])
                
                # Highlight in foil
                boat_color = 'cyan' if self.env.state[agent].get('is_foiling', False) else agent_color
                boat_edge = agent_color if self.env.state[agent].get('is_foiling', False) else 'darkgray'
                boat = patches.Polygon(boat_points, closed=True,
                                      facecolor=boat_color, edgecolor=boat_edge, linewidth=2)
                self.ax.add_patch(boat)

                # Overlay diagnostico collisioni (raggio contatto e near-collision)
                near_circle = patches.Circle(
                    (bx, by),
                    self.env.near_collision_radius,
                    fill=False,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.18,
                    edgecolor='orange',
                )
                coll_circle = patches.Circle(
                    (bx, by),
                    self.env.collision_radius,
                    fill=False,
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.25,
                    edgecolor='red',
                )
                self.ax.add_patch(near_circle)
                self.ax.add_patch(coll_circle)
                
                # Active foil text indication
                foil_side = "Port" if self.env.state[agent].get('active_foil', 1.0) == 1.0 else "Stbd"
                foil_str = f"FOILING ({foil_side})" if self.env.state[agent].get('is_foiling', False) else f"HULL ({foil_side})"
                self.ax.text(bx, by - 25, foil_str, fontsize=8, color='magenta', fontweight='bold', ha='center')
                
                # Distanza e step infos
                dist = np.linalg.norm(np.array([bx, by]) - self.env.target[agent])
                steps = self.env.state[agent].get('steps_to_target', self.env.step_count)
                info_lines.append(f"{agent}: {dist:.0f}m ({steps}stp)")
                
                rudder_input = self.env.state[agent].get('rudder_angle', 0.0)
                rudder_deg = int(rudder_input * 25)
                rudder_color = 'red' if abs(rudder_input) > 0.5 else 'black'
                self.ax.text(bx + 18, by + 18,
                        f"{rudder_deg:+d}°",
                        fontsize=7, color=rudder_color, fontweight='bold')

                trim_percent = int(self.env.state[agent].get('sail_trim', self.env.default_trim_level) * 100)
                self.ax.text(bx + 18, by + 7,
                    f"Trim:{trim_percent:02d}%",
                    fontsize=7, color='navy', fontweight='bold')
        
        # Info UI
        ref_agent = self.env.possible_agents[0]
        if ref_agent in self.env.state:
            dist = np.linalg.norm(np.array([self.env.state[ref_agent]['x'], self.env.state[ref_agent]['y']]) - self.env.target[ref_agent])

            # Vento locale
            local_wd, local_ws = self.env.wind_field.get_local_wind(
                self.env.state[ref_agent]['x'], self.env.state[ref_agent]['y']
            )
            wind_deg = (90 - np.degrees(local_wd)) % 360

            dist_to_gate = abs(self.env.target[ref_agent][1] - self.env.state[ref_agent]['y'])
            
            winner_text = ""
            finished_agents = {a: self.env.state[a]['steps_to_target'] for a in self.env.possible_agents if 'steps_to_target' in self.env.state[a]}
            if finished_agents:
                winner_agent = min(finished_agents, key=finished_agents.get)
                winner_steps = finished_agents[winner_agent]
                winner_text = f"WIN: {winner_agent} | "
                
            full_title = winner_text + " | ".join(info_lines)
            
            self.ax.set_title(
                f"{full_title}\n"
                f"Leg: {self.env.state[ref_agent].get('current_leg', 1)}/2 | "
                f"Speed: {self.env.state[ref_agent]['speed']:.1f} kts | "
                f"Trim: {self.env.state[ref_agent].get('sail_trim', self.env.default_trim_level) * 100:.0f}% | "
                f"Dist Y to Gate: {dist_to_gate:.0f}m",
                fontsize=10, weight='bold'
            )

            # --- Box info vento (centrato a sinistra) ---
            wind_text = (
                f"Wind (base)\n"
                f"Dir: {(90 - np.degrees(self.env.wind_field.base_direction)) % 360:.0f}\u00b0\n"
                f"Speed: {self.env.wind_field.base_speed:.1f} kts\n"
                f"\nWind (local @ boat)\n"
                f"Dir: {wind_deg:.0f}°\n"
                f"Speed: {local_ws:.1f} kts"
            )
            self.fig.text(
                0.18, 0.5, wind_text,
                fontsize=8, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='gray'),
                family='monospace'
            )

            # --- Rosa dei venti (centrata a destra) ---
            inset_ax = self.fig.add_axes([0.74, 0.42, 0.16, 0.16], polar=True)
            inset_ax.set_theta_zero_location('N')
            inset_ax.set_theta_direction(-1)
            inset_ax.set_rticks([])
            inset_ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            inset_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=6)
            inset_ax.set_title('Wind', fontsize=7, pad=2)
            
            compass_base = np.pi / 2 - self.env.wind_field.base_direction
            compass_local = np.pi / 2 - local_wd
            
            inset_ax.annotate(
                '', xy=(compass_base, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8)
            )
            inset_ax.annotate(
                '', xy=(compass_local, 0.65), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4, linestyle='dashed')
            )
        
        self.fig.canvas.draw()
        try:
            # Need copy=True so the array doesn't reference the mutable buffer
            image = np.array(self.fig.canvas.buffer_rgba(), copy=True)[:, :, :3]
        except AttributeError:
             image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
             image = np.array(image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)), copy=True)
             
        return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None