# N-Body System Simulation

This project provides a flexible N-dimensional simulation of gravitational interactions between multiple bodies, implemented in Python using numpy, scipy, pygame, and matplotlib. It includes a real-time Pygame visualization with interactive controls and tools for analyzing energy conservation and object trajectories.

## Features
**N-Dimensional Simulation:** Simulate gravitational systems in any number of dimensions (configured during initialization).
**Newtonian Gravity:** Accurate calculation of gravitational forces between all pairs of objects.
**Robust Integration:** Uses scipy.integrate.solve_ivp with an adaptive step-size RK45 method for reliable simulation over time.
**Pygame Visualization:** Real-time graphical display of the simulation state.
**Interactive Controls:** Pause/play, zoom, adjust simulation speed, toggle labels and orbits, and cycle through projection dimensions directly within the visualization window.
**Projection Lines:** Option to display projection lines from the N-dimensional position onto the 2D visualization plane.
**Historical Orbits:** Display the past trajectories of objects.
**Energy Analysis:** Plots kinetic, potential, and total energy over time to verify conservation.
**Trajectory Analysis:** Plots object paths, including pairwise projections for higher dimensions.


## Installation

To run this simulation, you need to have Python installed. You can then install the necessary libraries using `pip`:

```bash
pip install numpy matplotlib scipy pygame
```

## Usage

1.  Save the provided Python code as a `.py` file (e.g., `n_system_sim.py`).
2.  Open a terminal or command prompt.
3.  Navigate to the directory where you saved the file.
4.  Run the simulation using the command:

```bash
python n_system_sim.py
```
After the Pygame visualization is closed (either by reaching the end of the simulation duration or pressing Escape), the energy conservation and trajectory plots will be generated and displayed.

## Code Structure
The core functionality is encapsulated within the PhysicsSimulation class:
**__init__(self, dimensions=3):** Initializes the simulation with the specified number of dimensions and Pygame setup.
**add_object(self, mass, position, velocity):** Adds a new object to the simulation with its properties.
**calculate_acceleration(self, positions):** Computes the net gravitational acceleration for all objects based on their current positions.
**derivative(self, t, y):** Defines the system of first-order ODEs for the solver.
**run_simulation(self, duration, dt=0.01):** Executes the simulation using solve_ivp and stores the results in self.history.
**visualize:** Handles the Pygame visualization loop, rendering objects, orbits, and responding to user input.
**analyze_energy_conservation:** Plots the energy components over time.
**analyze_trajectories:** Plots the paths of the objects.
**The main execution block (if __name__ == "__main__":)** demonstrates how to create a PhysicsSimulation instance, add objects, run the simulation, and trigger the visualization and analysis.

## Scientific Accuracy and Math

The simulation is based on **Newton's Law of Universal Gravitation** and the principles of classical mechanics.

**Gravitational Force:** The force between two objects is calculated using the inverse square law, proportional to the product of their masses and inversely proportional to the square of the distance between them.
**N-Body Problem:** The simulation solves the N-body problem, where each object's acceleration is the vector sum of the gravitational forces exerted by all other objects.
**Ordinary Differential Equations (ODEs):** The motion of the system is described by a system of second-order ODEs (Newton's second law, F=ma). This is converted into a system of first-order ODEs by treating position and velocity as separate state variables.
**Numerical Integration:** The solve_ivp function with the RK45 method is used to numerically integrate these ODEs over time. RK45 (Runge-Kutta 4(5)) is a standard method that provides a good balance of accuracy and efficiency by adaptively adjusting the step size (dt) during the simulation to maintain a specified error tolerance.
**Numerical Stability:** A small epsilon value (1e-5) is added to the denominator in the acceleration calculation. This is a common technique to prevent division by zero or extremely large forces when objects get very close, which can cause numerical instability and lead to inaccurate or divergent results in a discrete-time simulation. While it slightly deviates from the pure inverse square law at extremely small distances, it is necessary for practical simulation stability.
**Energy Conservation:** In an ideal gravitational system with no external forces, the total energy (kinetic + potential) should be conserved. The energy analysis plot serves as a check on the numerical accuracy of the integration; small fluctuations in total energy are expected due to the discrete nature of numerical integration, but large drifts would indicate a problem.

## Interacting with the Pygame Visualization:
Space: Pause/Play the simulation.
L: Toggle object labels.
O: Toggle historical orbit lines.
P: Toggle projection lines from the N-dimensional position onto the 2D view.
+ / =: Zoom In.
-: Zoom Out.
Up Arrow: Increase simulation speed.
Down Arrow: Decrease simulation speed.
A: Set simulation speed to Normal (1x).
S: Set simulation speed to Fast (10x).
D: Set simulation speed to Very Fast (50x).
F: Set simulation speed to Super Fast (200x).
1: Cycle the first dimension projected onto the horizontal axis.
2: Cycle the second dimension projected onto the vertical axis.
3: Cycle the third dimension used for size/brightness scaling (only in >2D).
Mouse Wheel Up: Zoom In.
Mouse Wheel Down: Zoom Out.
Escape: Quit the simulation visualization.

Special Thanks to
Dot Physics Youtube Channel , Gemini , Claude and Grok .

Future Updates:
Phase 0: 

Code Refinement: Clean up the existing class. Add comments explaining sections, especially the physics calculations.
