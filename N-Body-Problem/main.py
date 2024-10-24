import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solver import solve_n_body, update_animation

# Number of planets (bodies) in the simulation
n_planets = 25  

# Masses of the planets (all equal for simplicity)
masses = np.ones(n_planets)

# Initial positions and velocities of the planets
# Randomized positions within a 3D box of size [-2, 2) for each axis
initial_positions = 2 * (2 * np.random.rand(n_planets, 3) - 1)

# Randomized velocities, small perturbations in the range [-0.1, 0.1)
initial_velocities = 0.1 * (2 * np.random.rand(n_planets, 3) - 1)

# Combine initial positions and velocities into a single array for solver
initial_conditions = np.array([initial_positions, initial_velocities]).ravel()



# Time parameters for the simulation
time_start, time_end = 0, 20  # Start and end times
n_time_points = 2001          # Number of time points for the simulation
time_points = np.linspace(time_start, time_end, n_time_points)

# Solve the system of differential equations
epsilon = 0.005
solution = solve_n_body(time_points, initial_conditions, masses, n_planets, epsilon)

# Extract the time solutions
t_sol = solution.t

# Prepare arrays to store the x, y, z positions of each planet
px_sol = []
py_sol = []
pz_sol = []

# Fill the arrays with solutions for each planet's position over time
for i in range(n_planets):
    px_sol.append(solution.y[3 * i])
    py_sol.append(solution.y[3 * i + 1])
    pz_sol.append(solution.y[3 * i + 2])



# Plotting setup for the 3D trajectory of the planets
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
planet_traces = []
planet_dots = []

# Initialize the plot lines and dots for each planet
for i in range(n_planets):
    # Trace line for each planet's trajectory
    planet_trace, = ax.plot(px_sol[i], py_sol[i], pz_sol[i], color='black', label=f"planet {i}")
    # Dot representing the current position of each planet
    planet_dot, = ax.plot([px_sol[i][-1]], [py_sol[i][-1]], [pz_sol[i][-1]], 'o', color='black')
    
    planet_traces.append(planet_trace)
    planet_dots.append(planet_dot)

# Set plot labels and title
ax.set_title(f"{n_planets}-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Set axis limits for visualization
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])

# Adjust the viewing angle for better perspective
ax.view_init(elev=30., azim=120)


# Animate the planetary motion

# Create the animation function
animation = FuncAnimation(
    fig, update_animation, frames=range(0, len(time_points), 4), interval=25, blit=False,
    fargs=(n_planets, px_sol, py_sol, pz_sol, planet_traces, planet_dots, ax)
)

# Optionally, save the animation as a GIF (uncomment if needed)
print("Saving the simulation as a gif...")

animation.save(f"n_body_simulation.gif")

print(".gif generated successfully!")

# Display the plot
plt.show()
