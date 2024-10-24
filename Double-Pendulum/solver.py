# ------- Importing Libraries ------- #
from scipy.integrate import solve_ivp
import sympy as sm
import numpy as np


def describe_lagrange():
    """
    Returns the lambdified symbolic expressions for the second-order derivatives of the angles
    (d²θ₁/dt² and d²θ₂/dt²) based on Lagrange's equations.
    
    Output:
        - LEF1 (function): Expression for d²θ₁/dt² (angular acceleration of the first pendulum)
        - LEF2 (function): Expression for d²θ₂/dt² (angular acceleration of the second pendulum)
    """
    # Define time as a symbol
    t = sm.symbols('t')
    # Define masses and gravitational acceleration
    m_1, m_2, g = sm.symbols('m_1 m_2 g', positive=True)
    
    # Define angular positions (θ₁, θ₂) as functions of time
    the1, the2 = sm.symbols(r'\theta_1, \theta_2', cls=sm.Function)
    the1 = the1(t)
    the2 = the2(t)

    # Cartesian coordinates of the masses based on the angular positions
    x1 = sm.sin(the1)
    y1 = -sm.cos(the1)
    x2 = x1 + sm.sin(the2)
    y2 = y1 + -sm.cos(the2)

    # First and second derivatives (angular velocity and angular acceleration)
    the1_d = sm.diff(the1, t)  # Angular velocity θ₁
    the1_dd = sm.diff(the1_d, t)  # Angular acceleration θ₁
    x1_d = sm.diff(x1, t)
    y1_d = sm.diff(y1, t)

    the2_d = sm.diff(the2, t)  # Angular velocity θ₂
    the2_dd = sm.diff(the2_d, t)  # Angular acceleration θ₂
    x2_d = sm.diff(x2, t)
    y2_d = sm.diff(y2, t)

    # Kinetic energy (T) and potential energy (V) for both pendulums
    T_1 = 1/2 * m_1 * (x1_d**2 + y1_d**2)
    T_2 = 1/2 * m_2 * (x2_d**2 + y2_d**2)
    V_1 = m_1 * g * y1
    V_2 = m_2 * g * y2

    # Lagrangian (L = T - V)
    L = T_1 + T_2 - (V_1 + V_2)

    # Lagrange's equations
    LE1 = sm.diff(sm.diff(L, the1_d), t) - sm.diff(L, the1)
    LE2 = sm.diff(sm.diff(L, the2_d), t) - sm.diff(L, the2)

    # Simplify equations
    LE1 = LE1.simplify()
    LE2 = LE2.simplify()

    # Solve for angular accelerations (d²θ₁/dt² and d²θ₂/dt²)
    solutions = sm.solve([LE1, LE2], the1_dd, the2_dd)

    # Lambdify the solutions for numerical evaluation
    LEF1 = sm.lambdify((the1, the2, the1_d, the2_d, t, m_1, m_2, g), solutions[the1_dd])
    LEF2 = sm.lambdify((the1, the2, the1_d, the2_d, t, m_1, m_2, g), solutions[the2_dd])

    return LEF1, LEF2


def system_of_odes(t, y, m_1, m_2, g, LEF1, LEF2):
    """
    Defines the system of ODEs for the double pendulum using the lambdified Lagrangian equations.
    
    Input:
        - t (float): Current time
        - y (list of float): Current state [θ₁, θ₂, dθ₁/dt, dθ₂/dt]
        - m_1 (float), m_2 (float): Masses of the pendulums
        - g (float): Gravitational acceleration
        - LEF1, LEF2: Lambdified functions for d²θ₁/dt² and d²θ₂/dt²
    
    Output:
        - list of float: Derivatives [dθ₁/dt, dθ₂/dt, d²θ₁/dt², d²θ₂/dt²]
    """
    the1, the2, the1_d, the2_d = y

    # Compute angular accelerations
    the1_dd = LEF1(the1, the2, the1_d, the2_d, t, m_1, m_2, g)
    the2_dd = LEF2(the1, the2, the1_d, the2_d, t, m_1, m_2, g)

    return [the1_d, the2_d, the1_dd, the2_dd]


def get_solutions(n_simulations: int, time_points: np.ndarray, m1: float, m2: float, g: float, LEF1, LEF2,
                  t_begin: float, t_end: float, initial_conditions: np.ndarray) -> np.ndarray:
    """
    Solves the double pendulum system for multiple initial conditions using scipy's solve_ivp.

    Input:
        - n_simulations (int): Number of simulations to run
        - time_points (ndarray): Array of time points for evaluation
        - m1 (float), m2 (float): Masses of the pendulums
        - g (float): Gravitational acceleration
        - LEF1, LEF2: Lambdified functions for angular accelerations
        - t_begin (float), t_end (float): Time range for the simulation
        - initial_conditions (ndarray): Initial states for all simulations
    
    Output:
        - ndarray: Array of solutions [t, θ₁, θ₂] for each simulation
    """
    # Initialize the solution array
    solutions = [None] * n_simulations

    for i in range(n_simulations):
        # Solve the system for the i-th set of initial conditions
        solution = solve_ivp(
            fun=system_of_odes, t_span=(t_begin, t_end),
            y0=initial_conditions[i], t_eval=time_points,
            args=(m1, m2, g, LEF1, LEF2)
        )

        # Extract time and angular solutions
        t_points = solution.t
        theta1_sol = solution.y[0]
        theta2_sol = solution.y[1]

        # Store the solutions [t, θ₁, θ₂] for this simulation
        solutions[i] = np.array([t_points, theta1_sol, theta2_sol])

    return np.array(solutions, dtype=np.float32)


def update(frame: int, n_simulations: int, solution_array: np.ndarray, pendulum_plots: list, ax) -> list:
    """
    Updates the plot for each frame in the animation.

    Input:
        - frame (int): Current frame number
        - n_simulations (int): Number of simulations
        - solution_array (ndarray): Array of solution data for all simulations
        - pendulum_plots (list): List of plot objects for each simulation
        - ax: Plot axis
    
    Output:
        - list: Updated plot objects
    """
    updated_artists = []  # Collect updated plot objects

    for i in range(n_simulations):
        theta1_sol = solution_array[i][1]
        theta2_sol = solution_array[i][2]

        # Calculate pendulum positions based on angular solutions
        x1_pendulum = np.sin(theta1_sol)
        y1_pendulum = -np.cos(theta1_sol)
        x2_pendulum = x1_pendulum + np.sin(theta2_sol)
        y2_pendulum = y1_pendulum + -np.cos(theta2_sol)

        # Update lines and masses for each pendulum
        pendulum_plots[i][0].set_data([0, x1_pendulum[frame]], [0, y1_pendulum[frame]])  # First pendulum arm
        pendulum_plots[i][1].set_data([x1_pendulum[frame], x2_pendulum[frame]], [y1_pendulum[frame], y2_pendulum[frame]])  # Second pendulum arm
        pendulum_plots[i][2].set_data([x1_pendulum[frame]], [y1_pendulum[frame]])  # First mass
        pendulum_plots[i][3].set_data([x2_pendulum[frame]], [y2_pendulum[frame]])  # Second mass

        # Append updated objects to the list
        updated_artists.extend([pendulum_plots[i][0], pendulum_plots[i][1], pendulum_plots[i][2], pendulum_plots[i][3]])

    # Update the title for each frame
    ax.set_title(f"Double Pendulum: {n_simulations} Simulations | Progress: {(frame+1)/len(solution_array[0, 1]):.1%}")

    return updated_artists
