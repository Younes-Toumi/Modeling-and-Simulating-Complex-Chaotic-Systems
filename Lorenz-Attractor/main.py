import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solver import get_solutions, update

# Simulation Parameters
n_simulations = 50  # Number of Lorenz attractor simulations
enable_animation = True  # Toggle animation on/off

# Lorenz system parameters (standard chaotic regime)
sigma = 10  # Prandtl number
beta = 8/3  # Geometric factor
rho = 28    # Rayleigh number

# Time parameters
t_begin, t_end = 0, 20  # Start and end times for simulation
n_time_points = 1501    # Number of time points for each simulation
time_points = np.linspace(t_begin, t_end, n_time_points)

# Initial conditions for each simulation (randomized) (NOTE: Choose one of the following cases)
case = 1

if case == 1:
    # Case 1. Generates initial conditions from random numbers over [-10, 10)
    initial_conditions = 10 * (2 * np.random.rand(n_simulations, 3) - 1)
    
else: # case == 2
    # Case 2. Generates initial conditions very close to each other (Adding small perturbation) 
    initial_condition = 10 * (2*np.random.rand(3) - 1)
    initial_conditions = [initial_condition]*n_simulations
    perturbation = 0.001*(2 * np.random.rand(n_simulations, 1) - 1)
    
    initial_conditions = initial_conditions + perturbation


# Get the numerical solutions for each simulation
solution_array = get_solutions(n_simulations, time_points, sigma, beta, rho, t_begin, t_end, initial_conditions)

# Plot setup (3D plot for the Lorenz attractor)
fig = plt.figure()
axis = fig.add_subplot(projection='3d')

# Axis labels and title
axis.set_xlabel('X Axis')
axis.set_ylabel('Y Axis')
axis.set_zlabel('Z Axis')
axis.set_title(f'Lorenz Attractor: {n_simulations} Simulations')

# Store plot objects for each simulation
lorenz_plots = []

for i in range(n_simulations):
    # Create a plot for each simulation with gradient color effect
    lorenz_plt, = axis.plot(solution_array[i][1], solution_array[i][2], solution_array[i][3], 
                            lw=2, alpha=0.8)  # Slight transparency
    lorenz_plots.append(lorenz_plt)

# Adjust the viewing angle for a better perspective
axis.view_init(elev=30., azim=120)

# Animate the simulation if enabled
if enable_animation:
    animation = FuncAnimation(
        fig, update, frames=len(time_points),
        interval=25, blit=False,
        fargs=(n_simulations, solution_array, lorenz_plots, axis)
    )

    # Additionally saving the simulation as a gif
    print("Saving the simulation as a gif...")
    animation.save(f"lorenz_attractor.gif")
    print(".gif generated successfully!")
    
# Display the plot
plt.show()
