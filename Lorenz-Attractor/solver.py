from scipy.integrate import solve_ivp
import numpy as np

def system_of_odes(t: float, S: list[float], sigma: float, beta: float, rho: float) -> list[float]:
    """
    Defines the system of ODEs for the Lorenz attractor.

    Input:
        - t (float): Current time
        - S (list of float): Current state [x, y, z] at time t
        - sigma (float): Prandtl number
        - beta (float): Geometric factor
        - rho (float): Rayleigh number
    
    Output:
        - list of float: Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = S  # Unpack current state

    # Lorenz equations
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return [dx_dt, dy_dt, dz_dt]

def get_solutions(n_simulations: int, time_points: np.ndarray, sigma: float, beta: float, rho: float, 
                  t_begin: float, t_end: float, initial_conditions: np.ndarray) -> np.ndarray:
    """
    Solves the Lorenz system for multiple initial conditions using scipy's solve_ivp.

    Input:
        - n_simulations (int): Number of simulations to run
        - time_points (ndarray): Array of time points for evaluation
        - sigma (float), beta (float), rho (float): Lorenz system parameters
        - t_begin (float), t_end (float): Time range for the simulation
        - initial_conditions (ndarray): Initial states for all simulations
    
    Output:
        - ndarray: Array of solutions [t, x, y, z] for each simulation
    """
    # Initialize the solution array
    solutions = [None] * n_simulations

    for i in range(n_simulations):
        # Solve the Lorenz system for the i-th set of initial conditions
        solution = solve_ivp(
            fun=system_of_odes, t_span=(t_begin, t_end),
            y0=initial_conditions[i], t_eval=time_points,
            args=(sigma, beta, rho)
        )

        # Extract time and state (x, y, z) solutions
        t_points = solution.t
        x_sol = solution.y[0]
        y_sol = solution.y[1]
        z_sol = solution.y[2]

        # Store the solutions [t, x, y, z] for this simulation
        solutions[i] = np.array([t_points, x_sol, y_sol, z_sol])

    return np.array(solutions, dtype=np.float32)

def update(frame: int, n_simulations: int, solution_array: np.ndarray, lorenz_plots: list, ax) -> list:
    """
    Updates the plot for each frame in the animation.

    Input:
        - frame (int): Current frame number
        - n_simulations (int): Number of simulations
        - solution_array (ndarray): Array of solution data for all simulations
        - lorenz_plots (list): List of plot objects for each simulation
        - ax: Plot axis (3D)
    
    Output:
        - list: Updated plot objects
    """
    lower_lim = max(0, frame - 200)  # Only display the last 200 points


    for i in range(n_simulations):
        # Update the plot data for each simulation
        x_current = solution_array[i, 1][lower_lim:frame+1]
        y_current = solution_array[i, 2][lower_lim:frame+1]
        z_current = solution_array[i, 3][lower_lim:frame+1]

        lorenz_plots[i].set_data(x_current, y_current)
        lorenz_plots[i].set_3d_properties(z_current)

    # Adjust the viewpoint slightly with each frame
    ax.view_init(elev=10., azim=0.25 * frame)
    ax.set_title(f"Lorenz Attractor: {n_simulations} Simulations | Progress: {(frame+1)/len(solution_array[0, 1]):.1%}")
    
    return lorenz_plots
