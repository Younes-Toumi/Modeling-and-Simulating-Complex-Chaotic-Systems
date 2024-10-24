# ------- Importing Libraries ------- #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from solver import get_solutions, update, describe_lagrange

# Simulation Parameters
n_simulations = 25  # Number of double pendulum simulations
enable_animation = True  # Toggle animation on/off

# Double pendulum parameters
m1, m2 = 2, 2  # Masses of the pendulums
g = 9.81  # Gravitational acceleration

# Lagrange equation factors (for solving the system)
LEF1, LEF2 = describe_lagrange()

# Time parameters
t_begin, t_end = 0, 10  # Start and end times for simulation
n_time_points = 1001    # Number of time points for each simulation
time_points = np.linspace(t_begin, t_end, n_time_points)

# Initial conditions for each simulation (randomized)
is_perturbed = True  # Toggle perturbation on/off

# Base initial conditions (positions in radians, velocities in radians/sec)
initial_position = [np.pi / 4, np.pi / 4]  # Initial angular positions
initial_velocity = [0.0, 0.0]  # Initial angular velocities

# Replicate initial conditions across simulations
initial_positions = np.array([initial_position] * n_simulations)
initial_velocities = np.array([initial_velocity] * n_simulations)

if is_perturbed:
    # Adding small perturbations to the initial positions
    perturbation = np.pi * (2 * np.random.rand(n_simulations, 1) - 1)  # Perturbations in [-π, π)
    initial_positions += perturbation

# Combine positions and velocities into a single array of initial conditions
initial_conditions = np.hstack((initial_positions, initial_velocities))

# Get the numerical solutions for each simulation
solution_array = get_solutions(n_simulations, time_points, m1, m2, g, LEF1, LEF2, t_begin, t_end, initial_conditions)

# Plot setup
fig, axis = plt.subplots()
axis.set_xlim([-3, 3])
axis.set_ylim([-3, 3])
plt.grid()

# Axis labels and title
axis.set_xlabel('X Axis')
axis.set_ylabel('Y Axis')
axis.set_title(f'Double Pendulum: {n_simulations} Simulations')

# Store plot objects for each simulation
pendulum_plots = []
colors = cm.viridis(np.linspace(0, 1, n_simulations))  # Generate n distinct colors from a colormap

for i in range(n_simulations):
    # Create lines for each pendulum arm and masses
    pendulum1, = axis.plot([], [], lw=2, color=colors[i])  # First pendulum arm
    pendulum2, = axis.plot([], [], lw=2, color=colors[i])  # Second pendulum arm
    mass1, = axis.plot([], [], 'o', markersize=4 * int(m1) + 1, color=colors[i])  # First mass
    mass2, = axis.plot([], [], 'o', markersize=4 * int(m2) + 1, color=colors[i])  # Second mass
    pendulum_plots.append([pendulum1, pendulum2, mass1, mass2])

# Animate the simulation if enabled
if enable_animation:
    animation = FuncAnimation(
        fig, update, frames=range(1, len(time_points), 2),
        interval=10, blit=True,
        fargs=(n_simulations, solution_array, pendulum_plots, axis)
    )

    # Additionally save the animation as a gif
    # animation.save(f"chaos_pendulum.gif", dpi=60)

# Display the plot
plt.show()
