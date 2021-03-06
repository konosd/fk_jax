import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import functools
import math
from fk import params
from fk import convert
from fk import plot


def forward(shape,
            checkpoints,
            cell_parameters,
            diffusivity,
            stimuli,
            dt, dx,
            probes_list):
    """
    Solves the Fenton-Karma model using second order finite difference and explicit euler update scheme.
    Neumann conditions are considered at the boundaries.
    Units are adimensional.
    Args:
        shape (Tuple[int, int]): The shape of the finite difference grid
        checkpoints (iter): An iterable that contains time steps in simulation units, at which pause, and display the state of the system
        cell_parameters (Dict[string, float]): Dictionary of physiological parameters as illustrated in Fenton, Cherry, 2002.
        diffusivity (float): Diffusivity of the cardiac tissue
        stimuli (List[Dict[string, object]]): A list of stimuli to provide energy to the tissue
        dt (float): time infinitesimal to use in the euler stepping scheme
        dx (float): space infinitesimal to use in the spatial gradient calculation
        probes_list (list of Tuples[int,int]): locations of probes to measure voltage
    Returns:
        (List[jax.numpy.ndarray]): The list of states at each checkpoint
    """
    state = init(shape)
    probes = [np.ones(len(checkpoints-1))]*len(probes_list)
    states = []
    for i in range(len(checkpoints) - 1):
        print("Solving at: %dms/%dms\t\t" % (checkpoints[i + 1], checkpoints[-1]), end="\r")
        state = _forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, np.ones(shape) * diffusivity, stimuli, dt, dx)
        for p in range(len(probes)):
            x_probe,y_probe = probes_list[p]
            val = state[2][x_probe,y_probe]
            probes[p]=jax.ops.index_update(probes[p], i, val)
        plot.show(state)
        states.append(state[2])
    return states, probes


@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    v = np.ones(shape)
    w = np.ones(shape)
    u = np.zeros(shape)

    # at = np.zeros(shape) # activation time
    # max_du = np.zeros(shape) # keep track of maximum du/dt

    state = (v, w, u)
    state = np.asarray(state)
    print("Updated!") #Kostas
    return state


@jax.jit
def step(state, t, params, D, stimuli, dt, dx):
    # v, w, u, at, max_du = state
    v, w, u = state

    # apply stimulus
    u = np.where(params["current_stimulus"], u, stimulate(t, u, stimuli))

    # apply boundary conditions
    v = neumann(v)
    w = neumann(w)
    u = neumann(u)

    # gate variables
    p = np.greater_equal(u, params["V_c"])
    q = np.greater_equal(u, params["V_v"])
    tau_v_minus = (1 - q) * params["tau_v1_minus"] + q * params["tau_v2_minus"]

    d_v = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / params["tau_v_plus"])
    d_w = ((1 - p) * (1 - w) / params["tau_w_minus"]) - ((p * w) / params["tau_w_plus"])

    # currents
    J_fi = - v * p * (u - params["V_c"]) * (1 - u) / params["tau_d"]
    J_so = (u * (1 - p) / params["tau_0"]) + (p / params["tau_r"])
    J_si = - (w * (1 + np.tanh(params["k"] * (u - params["V_csi"])))) / (2 * params["tau_si"])

    I_ion = -(J_fi + J_so + J_si) / params["Cm"]

    # voltage01
#     u_x, u_y = np.gradient(u)
    u_x, u_y = gradient(u, 0), gradient(u, 1)
    u_x /= dx
    u_y /= dx
#     u_xx = np.gradient(u_x, axis=0)
#     u_yy = np.gradient(u_y, axis=1)
    u_xx = gradient(u_x, 0)
    u_yy = gradient(u_y, 1)
    u_xx /= dx
    u_yy /= dx
#     D_x, D_y = np.gradient(D)
#     D_x /= dx
#     D_y /= dx

# Kostas ---------
    D_x, D_y = gradient(D,0), gradient(D,1)
    D_x /= dx
    D_y /= dx
    extra_term = D_x*u_x + D_y*u_y


    current_stimuli = np.zeros(u.shape)
    current_stimuli = np.where(params["current_stimulus"], stimulate(t, current_stimuli, stimuli), current_stimuli)

# Kostas ---------


    d_u = D * (u_xx + u_yy) + extra_term + I_ion + current_stimuli
    
    # checking du for activation time update
    # at = np.where(np.greater_equal(d_u,max_du), t, at)
    # max_du = np.where(np.greater_equal(d_u,max_du), d_u, max_du)

    # euler update
    v += d_v * dt
    w += d_w * dt
    u += d_u * dt

    
    return np.asarray((v, w, u))


@functools.partial(jax.jit, static_argnums=1)
def gradient(a, axis):
    sliced = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
    a_grad = jax.numpy.concatenate((
        # 3th order edge
        ((-11/6) * sliced(0, 2) + 3 * sliced(1, 3) - (3/2) * sliced(2, 4) + (1/3) * sliced(3, 5)),
        # ((-25/12) * sliced(0, 2) + 4 * sliced(1, 3) - 3 * sliced(2, 4) + (4/3) * sliced(3, 5) - (1/4) * sliced(4,6)),
        # 4th order inner
        ((1/12) * sliced(None, -4) - (2/3) * sliced(1, -3) + (2/3) * sliced(3, -1) - (1/12) * sliced(4, None)),
        # 3th order edge
        ((-1/3) * sliced(-5, -3) + (3/2) * sliced(-4, -2) - 3 * sliced(-3, -1) + (11/6) * sliced(-2, None))
        # ((1/4)*sliced(-6,-4) -(4/3) * sliced(-5, -3) + 3 * sliced(-4, -2) - 4 * sliced(-3, -1) + (25/12) * sliced(-2, None))
    ), axis)
    return a_grad

@jax.jit
def neumann(X):
    # X = jax.ops.index_update(X, jax.ops.index[0], X[1])
    # X = jax.ops.index_update(X, jax.ops.index[-1], X[-2])
    # X = jax.ops.index_update(X, jax.ops.index[..., 0], X[..., 1])
    # X = jax.ops.index_update(X, jax.ops.index[..., -1], X[..., -2])

    X = jax.ops.index_update(X, jax.ops.index[0], (18*X[1]/11 - 18*X[2]/22 + 6*X[3]/33))
    X = jax.ops.index_update(X, jax.ops.index[-1], (18*X[-2]/11 - 18*X[-3]/22 + 6*X[-4]/33))
    X = jax.ops.index_update(X, jax.ops.index[..., 0], (18*X[...,1]/11 - 18*X[...,2]/22 + 6*X[...,3]/33))
    X = jax.ops.index_update(X, jax.ops.index[..., -1], (18*X[...,-2]/11 - 18*X[...,-3]/22 + 6*X[...,-4]/33))

    # X = jax.ops.index_update(X, jax.ops.index[0], (48*X[1]/25 - 36*X[2]/25 + 48*X[3]/75 - 12*X[4]/100))
    # X = jax.ops.index_update(X, jax.ops.index[-1], (48*X[-2]/25 - 36*X[-3]/25 + 48*X[-4]/75 -12*X[-5]/100))
    # X = jax.ops.index_update(X, jax.ops.index[..., 0], (48*X[...,1]/25 - 36*X[...,2]/25 + 48*X[...,3]/75 -12*X[...,4]/100))
    # X = jax.ops.index_update(X, jax.ops.index[..., -1], (48*X[...,-2]/25 - 36*X[...,-3]/25 + 48*X[...,-4]/75 - 12*X[...,-5]/100))
    return X


@jax.jit
def stimulate(t, X, stimuli):
    stimulated = np.zeros_like(X)
    for stimulus in stimuli:
        # active = np.greater_equal(t, stimulus["start"])
        # active &= (np.mod(stimulus["start"] - t + 1, stimulus["period"]) < stimulus["duration"])
        active = np.greater_equal(t ,stimulus["start"])
        # active &= np.greater_equal(stimulus["start"] + stimulus["duration"],t)
        active = (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"]) # this works for cyclics
        stimulated = np.where(stimulus["field"] * (active), stimulus["field"], stimulated)
    return np.where(stimulated != 0, stimulated, X)


@jax.jit
def current_stimulate(t, X, stimuli):
    stimulated = np.zeros_like(X)
    for stimulus in stimuli:
        # active = np.greater_equal(t, stimulus["start"])
        # active &= (np.mod(stimulus["start"] - t + 1, stimulus["period"]) < stimulus["duration"])
        active = np.greater_equal(t ,stimulus["start"])
        # active &= np.greater_equal(stimulus["start"] + stimulus["duration"],t)
        active &= (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"]) # this works for cyclics
        stimulated = np.where(stimulus["field"] * (active), stimulus["field"], stimulated)
    return np.where(stimulated != 0, stimulated, X)


@jax.jit
def _forward(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
    state = jax.lax.fori_loop(t, t_end, lambda i, state: step(state, i, params, diffusion, stimuli, dt, dx), state)
    return state


@functools.partial(jax.jit, static_argnums=(1, 2))
def _forward_stack(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
    def _step(state, i):
        new_state = step(state, i, params, diffusion, stimuli, dt, dx)
        return (new_state, new_state)
    xs = np.arange(t, t_end)
    last_state, states = jax.lax.scan(_step, state, xs)
    return states
