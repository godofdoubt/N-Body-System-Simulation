import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pygame
import time

class PhysicsSimulation:
    def __init__(self, dimensions=3):
        """Initialize physics simulation with specified number of dimensions"""
        self.dimensions = dimensions
        self.G = 6.67430e-11  # Gravitational constant
        self.objects = []
        self.time = 0
        self.history = []  # For data analysis
        
        # Pygame visualization setup
        self.width, self.height = 1000, 800
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"{dimensions}D Physics Simulation")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = [
            (255, 255, 0),    # Yellow
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (128, 0, 128)     # Purple
        ]
        
        # Controls
        self.paused = False
        self.show_labels = True
        self.show_orbits = True
        self.show_projection_lines = False  # New control for projection lines
        self.simulation_speed = 0.1
        self.view_scale = 100.0

    def add_object(self, mass, position, velocity):
        """Add an object to the simulation"""
        if len(position) != self.dimensions or len(velocity) != self.dimensions:
            raise ValueError(f"Position and velocity must have {self.dimensions} dimensions")
        
        self.objects.append({
            'mass': mass,
            'position': np.array(position, dtype=float),
            'velocity': np.array(velocity, dtype=float),
            'color': self.colors[len(self.objects) % len(self.colors)]
        })
    
    def calculate_acceleration(self, positions):
        """Calculate acceleration for all objects based on gravitational forces"""
        n_objects = len(self.objects)
        positions_reshaped = positions.reshape(n_objects, self.dimensions)
        accelerations = np.zeros_like(positions_reshaped)
        
        for i in range(n_objects):
            for j in range(n_objects):
                if i != j:
                    r_vec = positions_reshaped[j] - positions_reshaped[i]
                    r_mag = np.linalg.norm(r_vec)
                    epsilon = 1e-5
                    acceleration = self.G * self.objects[j]['mass'] * r_vec / (r_mag**3 + epsilon)
                    accelerations[i] += acceleration
        
        return accelerations.flatten()
    
    def derivative(self, t, y):
        """System of differential equations for the N-body problem"""
        n_objects = len(self.objects)
        positions = y[:n_objects * self.dimensions]
        velocities = y[n_objects * self.dimensions:]
        
        accelerations = self.calculate_acceleration(positions)
        
        return np.concatenate([velocities, accelerations])
    
    def run_simulation(self, duration, dt=0.01):
        """Run the simulation for the specified duration"""
        n_objects = len(self.objects)
        
        initial_positions = np.array([obj['position'] for obj in self.objects]).flatten()
        initial_velocities = np.array([obj['velocity'] for obj in self.objects]).flatten()
        initial_state = np.concatenate([initial_positions, initial_velocities])
        
        t_span = (0, duration)
        t_eval = np.arange(0, duration, dt)
        
        solution = solve_ivp(
            self.derivative,
            t_span,
            initial_state,
            method='RK45',
            t_eval=t_eval
        )
        
        self.times = solution.t
        self.states = solution.y.T
        
        self.history = []
        for i, t in enumerate(self.times):
            state = self.states[i]
            positions = state[:n_objects * self.dimensions].reshape(n_objects, self.dimensions)
            velocities = state[n_objects * self.dimensions:].reshape(n_objects, self.dimensions)
            
            frame_data = {
                'time': t,
                'positions': positions.copy(),
                'velocities': velocities.copy(),
                'energies': self.calculate_energies(positions, velocities)
            }
            self.history.append(frame_data)
        
        return self.history
    
    def calculate_energies(self, positions, velocities):
        """Calculate kinetic, potential, and total energy for all objects"""
        n_objects = len(self.objects)
        kinetic_energy = 0
        potential_energy = 0
        
        for i in range(n_objects):
            kinetic_energy += 0.5 * self.objects[i]['mass'] * np.sum(velocities[i]**2)
        
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                potential_energy -= self.G * self.objects[i]['mass'] * self.objects[j]['mass'] / r
        
        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy
        }
    
    def visualize(self):
        """Visualize the simulation using Pygame with added controls and projection lines"""
        n_objects = len(self.objects)
        running = True
        simulation_time = 0.0
        font = pygame.font.SysFont('Arial', 18)
        projection_dims = [0, 1]
        if self.dimensions > 2:
            third_dim = 2
        
        # List of controls to display
        controls = [
            ("Space", "Pause/Play"),
            ("L", "Toggle Labels"),
            ("O", "Toggle Orbits"),
            ("P", "Toggle Projection Lines"),
            ("+", "Zoom In"),
            ("-", "Zoom Out"),
            ("Up", "Increase Speed"),
            ("Down", "Decrease Speed"),
            ("A", "Normal Speed"),
            ("S", "Fast Speed"),
            ("D", "Very Fast Speed"),
            ("F", "Super Fast Speed"),
            ("1", "Cycle Dim 1"),
            ("2", "Cycle Dim 2"),
            ("3", "Cycle Dim 3 (if >2D)"),
        ]
        
        while running:
            dt = self.clock.tick(30) / 1000.0
            if not self.paused:
                simulation_time += self.simulation_speed * dt
            frame_index = np.searchsorted(self.times, simulation_time, side='right') - 1
            if frame_index >= len(self.history):
                frame_index = len(self.history) - 1
                running = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_l:
                        self.show_labels = not self.show_labels
                    elif event.key == pygame.K_o:
                        self.show_orbits = not self.show_orbits
                    elif event.key == pygame.K_p:
                        self.show_projection_lines = not self.show_projection_lines
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.view_scale *= 1.1
                    elif event.key == pygame.K_MINUS:
                        self.view_scale /= 1.1
                    elif event.key == pygame.K_UP:
                        self.simulation_speed *= 2.0
                    elif event.key == pygame.K_DOWN:
                        self.simulation_speed /= 2.0
                    elif event.key == pygame.K_a:
                        self.simulation_speed = 1.0   # Normal speed
                    elif event.key == pygame.K_s:
                        self.simulation_speed = 10.0  # Fast
                    elif event.key == pygame.K_d:
                        self.simulation_speed = 50.0  # Very fast
                    elif event.key == pygame.K_f:
                        self.simulation_speed = 200.0 # Super fast
                    elif event.key == pygame.K_1:
                        projection_dims[0] = (projection_dims[0] + 1) % self.dimensions
                    elif event.key == pygame.K_2:
                        projection_dims[1] = (projection_dims[1] + 1) % self.dimensions
                    if self.dimensions > 2:
                        if event.key == pygame.K_3:
                            third_dim = (third_dim + 1) % self.dimensions
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Mouse wheel up
                        self.view_scale *= 1.1
                    elif event.button == 5:  # Mouse wheel down
                        self.view_scale /= 1.1
            
            self.screen.fill((0, 0, 0))
            
            dim_names = ['x', 'y', 'z', 'w', 'v'][:self.dimensions]
            projection_text = f"Projecting {dim_names[projection_dims[0]]} and {dim_names[projection_dims[1]]}"
            if self.dimensions > 2:
                projection_text += f", {dim_names[third_dim]} as depth"
            time_text = f"Time: {self.history[frame_index]['time']:.2f}s"
            
            text_surface = font.render(projection_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            text_surface = font.render(time_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 40))
            
            positions = self.history[frame_index]['positions']
            
            if self.show_orbits:
                for i in range(n_objects):
                    trajectory = np.array([frame['positions'][i] for frame in self.history[:frame_index+1]])
                    trajectory_proj = trajectory[:, projection_dims]
                    screen_positions = self.width / 2 + trajectory_proj * self.view_scale
                    
                    orbit_offset_y = 150 # You can adjust this value
                    screen_positions[:, 1] += orbit_offset_y 
                    
                    points = [(int(x), int(y)) for x, y in screen_positions]
                    pygame.draw.lines(self.screen, self.objects[i]['color'], False, points, 1)
            
            for i in range(n_objects):
                x = positions[i][projection_dims[0]]
                y = positions[i][projection_dims[1]]
                screen_x = int(self.width / 2 + x * self.view_scale)
                screen_y = int(self.height / 2 + y * self.view_scale)
                
                size = 5
                color = self.objects[i]['color']
                if self.dimensions > 2:
                    z = positions[i][third_dim]
                    size = max(2, int(5 + z * 3))
                    brightness = min(255, max(0, int(128 + z * 50)))
                    color = tuple(min(255, int(c * brightness / 128)) for c in color)
                
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), size)
                
                # Draw projection lines if enabled
                if self.show_projection_lines:
                    offset = 250  # Fixed pixel offset to move lines away
                    projection_screen_y = screen_y + offset
                    pygame.draw.line(self.screen, (100, 100, 100), (screen_x, screen_y), (screen_x, projection_screen_y), 1)
                    pygame.draw.circle(self.screen, (100, 100, 100), (screen_x, projection_screen_y), 2)
                
                trail_length = min(20, frame_index)
                for t in range(1, trail_length + 1):
                    if frame_index - t >= 0:
                        past_pos = self.history[frame_index - t]['positions'][i]
                        past_x = past_pos[projection_dims[0]]
                        past_y = past_pos[projection_dims[1]]
                        past_screen_x = int(self.width / 2 + past_x * self.view_scale)
                        past_screen_y = int(self.height / 2 + past_y * self.view_scale)
                        alpha = 255 * (1 - t / trail_length)
                        trail_color = tuple(min(255, int(c * alpha / 255)) for c in self.objects[i]['color'])
                        pygame.draw.circle(self.screen, trail_color, (past_screen_x, past_screen_y), 1)
                
                if self.show_labels:
                    label = font.render(f"Object {i+1}", True, (255, 255, 255))
                    self.screen.blit(label, (screen_x + 10, screen_y + 10))
            
            # Display key controls panel
            for i, (key, desc) in enumerate(controls):
                text = f"{key}: {desc}"
                text_surface = font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, self.height - 20 - i * 20))
            
            pygame.display.flip()
        
        pygame.quit()
    
    def analyze_energy_conservation(self):
        """Analyze and plot energy conservation during the simulation"""
        times = [frame['time'] for frame in self.history]
        kinetic = [frame['energies']['kinetic'] for frame in self.history]
        potential = [frame['energies']['potential'] for frame in self.history]
        total = [frame['energies']['kinetic'] + frame['energies']['potential'] for frame in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, kinetic, label='Kinetic Energy')
        plt.plot(times, potential, label='Potential Energy')
        plt.plot(times, total, label='Total Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Conservation Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_conservation.png')
        plt.show()
    
    def analyze_trajectories(self):
        """Plot the trajectories of all objects in various phase planes"""
        n_objects = len(self.objects)
        
        if self.dimensions <= 3:
            fig = plt.figure(figsize=(10, 8))
            if self.dimensions == 2:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111, projection='3d')
            
            for i in range(n_objects):
                positions = np.array([frame['positions'][i] for frame in self.history])
                
                if self.dimensions == 2:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           label=f'Object {i+1}', color=tuple(c/255 for c in self.objects[i]['color']))
                else:
                    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                           label=f'Object {i+1}', color=tuple(c/255 for c in self.objects[i]['color']))
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            if self.dimensions == 3:
                ax.set_zlabel('Z Position')
            
            plt.title('Object Trajectories')
            plt.legend()
            plt.savefig('trajectories.png')
            plt.show()
        else:
            dim_names = ['x', 'y', 'z', 'w', 'v', 'u', 't'][:self.dimensions]
            
            fig, axes = plt.subplots(self.dimensions, self.dimensions, figsize=(15, 15))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    if i == j:
                        axes[i, j].text(0.5, 0.5, dim_names[i], 
                                       horizontalalignment='center', verticalalignment='center')
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])
                    else:
                        for obj_idx in range(n_objects):
                            positions = np.array([frame['positions'][obj_idx] for frame in self.history])
                            axes[i, j].plot(positions[:, j], positions[:, i], 
                                           color=tuple(c/255 for c in self.objects[obj_idx]['color']),
                                           alpha=0.7, linewidth=1)
                        
                        if i == self.dimensions-1:
                            axes[i, j].set_xlabel(dim_names[j])
                        else:
                            axes[i, j].set_xticks([])
                        
                        if j == 0:
                            axes[i, j].set_ylabel(dim_names[i])
                        else:
                            axes[i, j].set_yticks([])
            
            plt.suptitle('Higher Dimensional Trajectories - Pairwise Projections')
            plt.savefig('higher_dim_trajectories.png')
            plt.show()

if __name__ == "__main__":
    sim = PhysicsSimulation(dimensions=4)
    sim.G = 1
    
    sim.add_object(mass=100, 
                  position=[0, 0, 0, 0], 
                  velocity=[0, 0, 0, 0])
    
    sim.add_object(mass=1, 
                  position=[1, 0, 0, 0], 
                  velocity=[0, 1, 0, 0])
    
    sim.add_object(mass=1, 
                  position=[0, 0, 1, 0], 
                  velocity=[0, 0, 0, 1])
    
    sim.add_object(mass=1,
                  position=[0, 1, 0, 1], 
                  velocity=[-0.7, 0, 0.7, 0])
    
    sim.run_simulation(duration=10, dt=0.001)
    sim.visualize()
    sim.analyze_energy_conservation()
    sim.analyze_trajectories()
