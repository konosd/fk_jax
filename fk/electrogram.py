from fk.model import gradient
import jax 
import jax.numpy as np
import h5py
import functools
import math

def hd_grid_probes(bottom_left_corner):
    x = (np.arange(0,12,3)*0.1+bottom_left_corner)
    y = (np.arange(0,12,3)*0.1+bottom_left_corner)
    points = [(i,j) for i in x for j in y]
    return points

def steady_probes(filepath, dt=1, dx=0.01, real_size = 12, scaled_size = 256):
    '''
    Calculates the contact electrogram at a 4x4 grid, that is distributed along the field.
    - points: list of tuples, each tuple contains (x,y) of location (in cm) of unipolar
    - dt: timestep between states set to 1 ms
    - dx: grid discretization set to 0.01 cm
    - real_size: in cm
    - scaled_size: in pixels - matrix dimensions

    Returns the 4x4 field, of 16 pixels, where each pixel is the value of the contact electrogram
    at each of the positions.
    '''
    p = [2.4, 4.8, 7.2, 9.6]
    points = np.meshgrid(p,p, indexing = 'ij')
    filename = filepath[:-5]+'_ecg.hdf5'
    states_file = h5py.File(filepath, 'r')
    states = states_file['states']
    conductivity_field = states_file["params/D"][:]
    shape = states.shape
    phi = np.zeros((shape[0], 4, 4))
    x = np.linspace(0, real_size, states.shape[-1])
    y = np.linspace(0, real_size, states.shape[-1])

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    egm_x = points[0].reshape((1,16))
    egm_y = points[1].reshape((1,16))

    for i in range(states.shape[0]):
        u = states[i,2]
        u_x = fk.model.gradient(u,0)/dx
        u_y =  fk.model.gradient(u,1)/dx
        val = -np.sum((u_y[:,:,np.newaxis]*(egm_y-yv[:,:,np.newaxis])+u_x[:,:,np.newaxis]*(egm_x-xv[:,:,np.newaxis]))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x-xv[:,:,np.newaxis])**2+(egm_y-yv[:,:,np.newaxis])**2+10**(-2))**3), axis=(0,1)).reshape((4,4))   
        phi = jax.ops.index_update(phi, i, val)
        
    hdf5 = h5py.File(filename, "w")
    ecg_dset = hdf5.create_dataset('electrogram', shape = phi.shape, dtype = "float32")
    ecg_dset[:] = phi
    conductivity = hdf5.create_dataset('conductivity', shape = states.shape[-2:], dtype = 'float32')
    conductivity[:] = conductivity_field
    return True

def calc_egm(filepath, points, dt=1, dx=0.01, real_size = 12, scaled_size = 256):
    '''
    Calculates the contact electrogram at the given points.
    - points: list of tuples, each tuple contains (x,y) of location (in cm) of unipolar
    - dt: timestep between states set to 1 ms
    - dx: grid discretization set to 0.01 cm
    - real_size: in cm
    - scaled_size: in pixels - matrix dimensions
    '''
    filename = filepath[:-5]+'_ecg.hdf5'
    states_file = h5py.File(filepath, 'r')
    states = states_file['states']
    shape = states.shape
    phi = np.zeros((len(points),shape[0]))
    x = np.linspace(0, real_size, states.shape[-1])
    y = np.linspace(0, real_size, states.shape[-1])
    
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    
    for i in range(states.shape[0]):
        u = states[i,2,:,:]
        u_x = gradient(u,0)/dx
        u_y =  gradient(u,1)/dx
        for j in range(len(points)):
            egm_x = points[j][0]
            egm_y = points[j][1]
#             val = np.sum((egm_dy*(yv - egm_y)+egm_dx*(xv - egm_x))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv)**2+(egm_y-yv)**2+10**(-6))**3))
            val = -np.sum((u_y*(egm_y-yv)+u_x*(+ egm_x - xv))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv)**2+(egm_y-yv)**2+10**(-2))**3))  
#             print(val)
            phi = jax.ops.index_update(phi, jax.ops.index[j,i], val)
    hdf5 = h5py.File(filename, "w")
    ecg_dset = hdf5.create_dataset('electrogram', shape = phi.shape, dtype = "float32")
    ecg_dset[:] = phi
    return True

def calc_local_egm(params, points, dt=1, dx=0.01, real_size = 12, scaled_size = 256, radius=3):
    '''
    Calculates the contact electrogram at the given points.
    - points: list of tuples, each tuple contains (x,y) of location (in cm) of unipolar
    - dt: timestep between states set to 1 ms
    - dx: grid discretization set to 0.01 cm
    - real_size: in cm
    - scaled_size: in pixels - matrix dimensions
    '''
    states_file = h5py.File(params['file'], 'r')
    states = states_file['states']
    shape = states.shape
    phi = np.zeros((len(points),shape[0]))
    x = np.linspace(0, real_size, states.shape[-1])
    y = np.linspace(0, real_size, states.shape[-1])
    
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    
    for i in range(states.shape[0]):
        u = states[i,2,:,:]
        u_x = gradient(u,0)/dx
        u_y = gradient(u,1)/dx
        for j in range(len(points)):
            egm_x = points[j][0]
            egm_y = points[j][1]
            egm_dx = u_x[int(scaled_size*egm_x/real_size),int(scaled_size*egm_y/real_size)]
            egm_dy = u_y[int(scaled_size*egm_x/real_size),int(scaled_size*egm_y/real_size)]
#             val = np.sum((egm_dy*(yv - egm_y)+egm_dx*(xv - egm_x))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv)**2+(egm_y-yv)**2+10**(-6))**3))
#             val = -np.sum((u_y*(egm_y-yv)+u_x*(egm_x - xv))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv)**2+(egm_y-yv)**2+10**(-6))**3))  
#             print(val)
            val = (u_y*(egm_y-yv)+u_x*(egm_x - xv))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv)**2+(egm_y-yv)**2+10**(-2))**3)
            x1 = int( np.max( (scaled_size*(egm_x-radius)/real_size, 0) ))
            x2 = int( np.min( (scaled_size*(egm_x+radius)/real_size, scaled_size) ))
            y1 = int( np.max( (scaled_size*(egm_y-radius)/real_size, 0) ))
            y2 = int( np.min( (scaled_size*(egm_y+radius)/real_size, scaled_size) ))
            val = jax.ops.index_update(val, jax.ops.index[:x1, :], 0)
            val = jax.ops.index_update(val, jax.ops.index[x2:, :], 0)
            val = jax.ops.index_update(val, jax.ops.index[:, :y1], 0)
            val = jax.ops.index_update(val, jax.ops.index[:, y2:], 0)
#             plt.imshow(val)


            val = -np.sum(val)
            phi = jax.ops.index_update(phi, jax.ops.index[j,i], val)
    return phi



def calc_round_egm(params, points, dt=1, dx=0.01, real_size = 12, scaled_size = 256, radius=3):
    '''
    Calculates the contact electrogram at the given points.
    - points: list of tuples, each tuple contains (x,y) of location (in cm) of unipolar
    - dt: timestep between states set to 1 ms
    - dx: grid discretization set to 0.01 cm
    - real_size: in cm
    - scaled_size: in pixels - matrix dimensions
    '''
    states_file = h5py.File(params['file'], 'r')
    states = states_file['states']
    shape = states.shape
    phi = np.zeros((len(points),shape[0]))
    x = np.linspace(0, real_size, states.shape[-1])
    y = np.linspace(0, real_size, states.shape[-1])
    
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    radi = radius*scaled_size/real_size
    X = int(radi)
    circle = np.array([[x,y] for x in range(-X,X+1) for y in range(-int((radi*radi-x*x)**0.5), int((radi*radi-x*x)**0.5)+1)])

    
    for i in range(states.shape[0]):
        u = states[i,2,:,:]
        u_x = gradient(u,0)/dx
        u_y = gradient(u,1)/dx
        for j in range(len(points)):
            egm_x = points[j][0]
            egm_y = points[j][1]
            
            circ_mask = circle + np.array([int(scaled_size*egm_x/real_size), int(scaled_size*egm_y/real_size)])
            circ_mask = circ_mask[ (circ_mask[:,0]<scaled_size) & (circ_mask[:,1]<scaled_size) ]
            ii = circ_mask[:,0]
            jj = circ_mask[:,1]
            
            val = (u_y[ii,jj]*(egm_y-yv[ii,jj])+u_x[ii,jj]*(egm_x - xv[ii,jj]))*0.0001/(np.pi*4*2.36*1*np.sqrt((egm_x - xv[ii,jj])**2+(egm_y-yv[ii,jj])**2+10**(-2))**3)
            
#             val = jax.ops.index_update(val, jax.ops.index[circ_mask[:,0], circ_mask[:,1]], 0)

#             plt.imshow(val)
            val = -np.sum(val)
            phi = jax.ops.index_update(phi, jax.ops.index[j,i], val)
#         print(i) 
    return phi