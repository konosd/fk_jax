import jax
import jax.numpy as np
from scipy.ndimage.interpolation import rotate
import random
from fk import convert
import numpy as onp

def rectangular(shape, centre, size, diffusion, scar):
    """
    Generates a rectangular scar given the center and the dimension of the rectangle.
    Args:
        shape (Tuple[int, int]): the shape of the scar array in simulation units (not cm)
        centre (Tuple[int, int]): the carthesian coordinates of the centre of the rectangle in simulation units
        size (Tuple[int, int]): the width and height dimensions of the rectangle in simulation units
        diffusion (float): the conductivity of the regular tissue
        scar (float): the conductivity of the scar tissue as a multiple of the normal diffusivity
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        The np.array that will be used multiplied by diffusivity units, in the simulations.
    """
    mask = np.ones(shape, dtype="float32")*diffusion
    x1 = (int(centre[0] - size[0] / 2))
    x2 = (int(centre[0] + size[0] / 2))
    y1 = (int(centre[1] - size[1] / 2))
    y2 = (int(centre[1] + size[1] / 2))
    mask = jax.ops.index_update(mask, jax.ops.index[x1:x2, y1:y2], scar*diffusion)
    return mask


def linear(shape, direction, coverage, diffusion, scar):
    """
    Generates a linear wave stimulus.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        direction (str): Direction of the wave as a string. Can be either:
                         'left', 'right', 'up', or 'down'
        coverage (float): percentage of the field that the wave will cover.
                        It must be between 0 and 1
        modulus (float): the amplitude of the stimulus in mV
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        (Dict[str, object]): The stimulus as a dictionary containing:
                             "field": (np.ndarray),
                             "start": (int),
                             "duration": (int),
                             "period": (int)
    """
    direction = direction.lower()
    stripe_size = int(shape[0] * coverage)
    stripe = None
    if direction == "left":
        stripe = jax.ops.index[:, :stripe_size]
    elif direction == "right":
        stripe = jax.ops.index[:, -stripe_size:]
    elif direction == "up":
        stripe = jax.ops.index[:stripe_size, :]
    elif direction == "down":
        stripe = jax.ops.index[-stripe_size:, :]
    else:
        raise ValueError("direction mus be either 'left', 'right', 'up', or 'down' not %s" % direction)
        
    mask = np.ones(shape, dtype="float32")*diffusion
    mask = jax.ops.index_update(mask, stripe, scar*diffusion)
    return mask


def triangular(shape, direction, angle, coverage, diffusion, scar):
    """
    Generates a linear wave at a custom angle.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        direction (str): Direction of the wave as a string. Can be either:
                         'left', 'right', 'up', or 'down'        
        angle (str): Incidence angle of the wave in degrees.
        coverage (float): percentage of the field that the wave will cover.
                        It must be between 0 and 1
        modulus (float): the amplitude of the stimulus in mV
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        (Dict[str, object]): The stimulus as a dictionary containing:
                             "field": (np.ndarray),
                             "start": (int),
                             "duration": (int),
                             "period": (int)
    """
    stim = linear(shape, direction, coverage, diffusion, scar)
    stim = rotate(stim, angle=angle, mode="nearest", prefilter=False, reshape=False)
    return stim
    

def random_triangular(shape, diffusion, scar):
    @property
    def rand(a=None):
        angle = random.random() * 360
        stim = triangular(shape, "up", angle, 0.2, diffusion, scar)
        return stim
    return rand.fget()


def random_rectangular(shape, dx, diffusion, scar):
    @property
    def rand(key):
        x1 = random.random() * (shape[0] - 1)
        x2 = random.random() * (shape[1] - 1)
        centre = (x1, x2)
        size = convert.realsize_to_shape((1, 1), dx)
        stim = rectangular(shape, centre, size, diffusion, scar)
        return stim
    return rand.fget()


def circular(shape, centre, radius, diffusion, scar):
    """
    Generates a rectangular scar given the center and the dimension of the rectangle.
    Args:
        shape (Tuple[int, int]): the shape of the scar array in simulation units (not cm)
        centre (Tuple[int, int]): the carthesian coordinates of the centre of the rectangle in simulation units
        radius (int): radius of circle in simulation units
        diffusion (float): the conductivity of the regular tissue
        scar (float): the conductivity of the scar tissue as a multiple of the normal diffusivity
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        The np.array that will be used multiplied by diffusivity units, in the simulations.
    """
    difmap = np.ones(shape, dtype="float32")*diffusion
    circle = np.array([[x,y] for x in range(-radius,radius+1) for y in range(-int((radius*radius-x*x)**0.5), int((radius*radius-x*x)**0.5)+1)])
    circ_mask = circle + np.array(centre)
    circ_mask = circ_mask[ (circ_mask[:,0]<shape[0]) & (circ_mask[:,1]<shape[0]) ]
    circ_mask = circ_mask[ (circ_mask[:,0]>0) & (circ_mask[:,1]>0) ]
    
    ii = circ_mask[:,0]
    jj = circ_mask[:,1]
    
    difmap = jax.ops.index_update(difmap, jax.ops.index[ii, jj], scar*diffusion)
    return difmap


def random_circular(shape, num_circles, min_radius, max_radius, diffusion, scar):
    """
    Generates a rectangular scar given the center and the dimension of the rectangle.
    Args:
        shape (Tuple[int, int]): the shape of the scar array in simulation units (not cm)
        num_circles (int): the number of circles on the field
        min_radius (int): minimum radius of circles in simulation units
        diffusion (float): the conductivity of the regular tissue
        scar (float): the conductivity of the scar tissue as a multiple of the normal diffusivity
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        The np.array that will be used multiplied by diffusivity units, in the simulations.
    """
    difmap = np.ones(shape, dtype="float32")*diffusion
    ii = []
    jj = []
    for c in range(num_circles):
        radius = onp.random.randint(min_radius, max_radius)
        center = (onp.random.randint(int(0.1*shape[0]), int(0.9*shape[0])), onp.random.randint(int(0.1*shape[1]), int(0.9*shape[1])))
        circle = np.array([[x,y] for x in range(-radius,radius+1) for y in range(-int((radius*radius-x*x)**0.5), int((radius*radius-x*x)**0.5)+1)])
        circ_mask = circle + np.array(center)
        circ_mask = circ_mask[ (circ_mask[:,0]<shape[0]) & (circ_mask[:,1]<shape[0]) ]
        circ_mask = circ_mask[ (circ_mask[:,0]>0) & (circ_mask[:,1]>0) ]

        ii = circ_mask[:,0]
        jj = circ_mask[:,1]
        difmap = jax.ops.index_update(difmap, jax.ops.index[ii, jj], scar*diffusion)
    return difmap


