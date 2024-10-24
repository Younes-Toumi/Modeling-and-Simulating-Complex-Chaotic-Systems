import numpy as np
from scipy.integrate import solve_ivp

def system_odes(t: float, S: np.ndarray, masses: np.ndarray, n_planets: int, epsilon: float = 0.01) -> np.ndarray:
    """
    Defines the system of differential equations for the N-body problem, with gravitational softening
    to handle very small distances between planets.

    Input:
        - t (float): Current time (unused, system is time-independent)
        - S (ndarray): State vector containing positions and velocities of all planets
        - masses (ndarray): Array of masses for the planets
        - n_planets (int): Number of planets in the system
        - epsilon (float): Softening parameter to prevent singularities at close distances
    
    Output:
        - ndarray: Derivatives of positions and velocities (i.e., velocities and accelerations)
    """

    print(f"Currently at t = {t:.3f} [s]", end='\r')
    
    # Extract positions and velocities from the state vector
    positions = np.array([S[3 * i:3 * i + 3] for i in range(n_planets)])
    velocities = np.array([S[3 * (i + n_planets):3 * (i + n_planets) + 3] for i in range(n_planets)])
    
    # Array to store accelerations
    accelerations = np.zeros_like(positions)

    # Compute accelerations based on gravitational interactions
    for j in range(n_planets):
        acceleration_sum = np.zeros(3)
        for i in range(n_planets):
            if i != j:
                # Gravitational force between planet i and planet j with softening factor
                r_ij = positions[i] - positions[j]
                distance_squared = np.dot(r_ij, r_ij) + epsilon**2  # Add softening factor to prevent zero division
                acceleration_sum += masses[i] * r_ij / distance_squared**(3/2)
        
        # Update acceleration for planet j
        accelerations[j] = acceleration_sum

    # Construct the derivative of the state vector (velocities and accelerations)
    dS_dt = np.concatenate([velocities, accelerations]).ravel()

    return dS_dt


def solve_n_body(time_points: np.ndarray, initial_conditions: np.ndarray, masses: np.ndarray, n_planets: int, epsilon:float):
    """
    Solves the N-body problem using scipy's solve_ivp.

    Input:
        - time_points (ndarray): Array of time points for the simulation
        - initial_conditions (ndarray): Initial positions and velocities of all planets
        - masses (ndarray): Array of masses for the planets
        - n_planets (int): Number of planets in the system

    Output:
        - solution (OdeSolution): Object containing the numerical solution for the system
    """

    # Solve the system of differential equations
    solution = solve_ivp(
        fun=system_odes, t_span=(time_points[0], time_points[-1]), y0=initial_conditions,
        t_eval=time_points, args=(masses, n_planets, epsilon)
    )
    
    return solution

def update_animation(frame: int, n_planets: int, px_sol: np.ndarray, py_sol: np.ndarray, pz_sol: np.ndarray,
                     planet_traces: list, planet_dots: list, ax) -> list:
    """
    Updates the positions of the planets for each frame in the animation.

    Input:
        - frame (int): Current frame number
        - n_planets (int): Number of planets
        - px_sol, py_sol, pz_sol (ndarrays): Solutions for x, y, z positions of all planets
        - planet_traces (list): List of Line3D objects representing the trajectory traces
        - planet_dots (list): List of Line3D objects representing the current positions (dots)
        - ax (Axes3D): 3D plot axis
    
    Output:
        - list: Updated plot objects (traces and dots)
    """
    # Number of time steps to display (limits trailing length of the trajectory)
    trail_length = 500
    lower_limit = max(0, frame - trail_length)

    # Update the position of each planet's trace and dot
    for i in range(n_planets):
        # Update the trace of the planet's trajectory
        planet_traces[i].set_data(px_sol[i][lower_limit:frame], py_sol[i][lower_limit:frame])
        planet_traces[i].set_3d_properties(pz_sol[i][lower_limit:frame])

        # Update the current position (dot) of the planet
        planet_dots[i].set_data([px_sol[i][frame]], [py_sol[i][frame]])
        planet_dots[i].set_3d_properties([pz_sol[i][frame]])

    # Optionally update the viewing angle for a dynamic effect
    ax.view_init(elev=10., azim=0.1 * frame)

    return planet_traces + planet_dots
