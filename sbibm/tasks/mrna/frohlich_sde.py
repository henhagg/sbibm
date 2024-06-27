# The following code is taken and from https://github.com/arrjon/Amortized-NLME-Models and slightly modified

# load necessary packages
import numpy as np
from numba import jit

@jit(nopython=True)
def drift_term(x: np.ndarray, delta: float, gamma: float, k: float) -> np.ndarray:
    """
    Computes the drift term of the SDE.

    Parameters:
    x (np.ndarray): 2-dimensional state of the system at time t.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.

    Returns:
    np.ndarray: Drift term of the SDE at time t.
    """
    m = -delta * x[0]
    p = k * x[0] - gamma * x[1]
    return np.array([m, p])


@jit(nopython=True)
def diffusion_term(x: np.ndarray, delta: float, gamma: float, k: float) -> np.ndarray:
    """
    Computes the diffusion term of the SDE.

    Parameters:
    x (np.ndarray): 2-dimensional state of the system at time t.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.

    Returns:
    np.ndarray: Diffusion term of the SDE at time t.
    """
    m = np.sqrt(delta * x[0])
    p = np.sqrt(k * x[0] + gamma * x[1])
    return np.array([m, p])


@jit(nopython=True)
def measurement(x: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """
    Applies a measurement function to a given variable.

    Parameters:
    x (np.ndarray): Measurements of the variable to which the measurement function should be applied.
    scale (float): Scale for the measurement function.
    offset (float): Offset for the measurement function.

    Returns:
    np.ndarray: The result of applying the measurement function to the input variable.
    """
    return np.log(scale * x[:, 1] + offset)


@jit(nopython=True)
def euler_maruyama(t0: float, m0: float, delta: float, gamma: float, k: float,
                   step_size: float = 0.01) -> np.ndarray:
    """
    Simulates the SDE using the Euler-Maruyama method.

    Parameters:
    t0 (float): Initial time point.
    m0 (float): Initial value of the system.
    delta (float): Parameter of the SDE.
    gamma (float): Parameter of the SDE.
    k (float): Parameter of the SDE.
    step_size (float, optional): Step size for the Euler-Maruyama method.

    Returns:
    tuple(np.ndarray, np.ndarray): The simulated trajectory and the corresponding time points.
    """
    # if t0 > 30, return only the measurement at t0
    if t0 > 30:
        t_points = np.array([t0])
        x0 = np.array([[m0, 0]])
        return np.column_stack((t_points, x0))

    # precompute time points (including t0 and t_end=30)
    t_points = np.arange(t0, 30 + step_size, step_size)
    n_points = t_points.size

    # initialize array
    x = np.zeros((n_points, 2))
    x[0] = [m0, 0]

    # precompute random numbers for the diffusion term (Brownian motion)
    bm = np.random.normal(loc=0, scale=np.sqrt(step_size), size=(n_points - 1, 2))

    # simulate one step at a time (first step already done)
    for t_idx in range(n_points - 1):
        drift = step_size * drift_term(x[t_idx], delta, gamma, k)
        diffusion = bm[t_idx] * diffusion_term(x[t_idx], delta, gamma, k)
        # add up and set negative values to zero
        x[t_idx + 1] = np.maximum(x[t_idx] + drift + diffusion, 0)
    return np.column_stack((t_points, x))


@jit(nopython=True)
def add_noise_for_one_cell(y: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """
    Adds Gaussian noise to a given trajectory.

    Parameters:
    y (np.ndarray): The trajectory to which noise should be added.
    sigma (float): Standard deviation of the Gaussian noise.
    seed (int, optional): Seed for the random number generator.
            If provided, the function will always return the same noise for the same seed.

    Returns:
    np.ndarray: The noisy trajectory.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=sigma, size=y.size)
    y_noisy = y + noise
    return y_noisy


def batch_simulator(param_samples: np.ndarray, n_obs: int, with_noise: bool = True) -> np.ndarray:
    """
    Simulate ODE model

    param_samples: np.ndarray - (#simulations, #parameters) or (#parameters)
    n_obs: int - number of observations to generate
    with_noise: bool - if noise should be added to the simulation (must be true during training)

    Return: sim_data: np.ndarray - simulated data (#simulations, #observations, 1) or (#observations, 1)
    """

    n_sim = param_samples.shape[0]
    sim_data = np.zeros((n_sim, n_obs, 1), dtype=np.float32)
    t_points = np.linspace(start=1 / 6, stop=30, num=n_obs, endpoint=True)

    # iterate over batch
    for i, p_sample in enumerate(param_samples):
        delta, gamma, k, m0, scale, t0, offset, sigma = np.exp(p_sample)
        # simulate all observations together
        sol_euler = euler_maruyama(t0, m0, delta, gamma, k)
        sol_approx = measurement(sol_euler[:, 1:], scale, offset)
        # get the closest time points to t_points via interpolation
        sol = np.interp(t_points, sol_euler[:, 0], sol_approx, left=np.log(offset))
        # add noise
        if with_noise:
            sim_data[i, :, 0] = add_noise_for_one_cell(sol, sigma)
        else:
            sim_data[i, :, 0] = sol
    return sim_data