import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def init_sys(l):
    sys = np.zeros((l,l,l))

    return sys

def elec_field(phi,dx):
    '''
    Approximation for electric field vector
    '''
    x_comp = 1./(2.*dx) * (np.roll(phi,+1,axis=0) - np.roll(phi,-1,axis=0))
    y_comp = 1./(2.*dx) * (np.roll(phi,+1,axis=1) - np.roll(phi,-1,axis=1))
    z_comp = 1./(2.*dx) * (np.roll(phi,+1,axis=2) - np.roll(phi,-1,axis=2))

    return x_comp, y_comp, z_comp

def jacobi(phi,rho,dx):
    '''
    update using 3D vectorised jacobi algorithm
    '''
    phi = np.copy(phi)

    new_phi = (1.0/6.0) * (np.roll(phi,+1,axis=0) + np.roll(phi,-1,axis=0) + np.roll(phi,+1,axis=1) + np.roll(phi,-1,axis=1) + np.roll(phi,+1,axis=2) + np.roll(phi,-1,axis=2) + (dx**2.0)*rho )

    #boundary conditions
    new_phi[0,:,:]=0
    new_phi[:,0,:]=0
    new_phi[:,:,0]=0
    new_phi[-1,:,:]=0
    new_phi[:,-1,:]=0
    new_phi[:,:,-1]=0

    return new_phi

def jac_state_update(phi,rho,dx):
    '''
    Updates state and computes difference between the 2 times
    '''
    new_phi = jacobi(phi,rho,dx)
    delta = abs(np.sum(new_phi - phi))

    return new_phi, delta

def gs_alg(phi,l):
    '''
    Updates phi using Gauss-Seidel algorithm
    '''
    for i in range(l):
        for j in range(l):
            for k in range(l):
                phi[i,j,k] = (1./6.) * (phi[(i+1)%l,j,k] + phi[(i-1)%l,j,k] + phi[i,(j+1)%l,k] + phi[i,(j-1)%l,k] + phi[i,j,(k+1)%l] + phi[i,j,(k-1)%l] + (dx**2.)*rho[i,j,k])

    phi[0,:,:]=0
    phi[:,0,:]=0
    phi[:,:,0]=0
    phi[-1,:,:]=0
    phi[:,-1,:]=0
    phi[:,:,-1]=0

    return phi

def gs_state_update(phi,l):
    new_phi = gs_alg(phi,l)
    delta = abs(np.sum(new_phi - phi))

    return new_phi, delta

def dist_to_centre(x,y,z,c):
    r = np.sqrt((c-x)**2. + (c-y)**2. + (c-z)**2.)

    return r


l=100
dx=1.0
rho=init_sys(l)
rho[50,50,50]=1.0

phi = init_sys(l)

f1=open('potvsR.dat','w')
f2=open('electric_field.dat','w')
for i in range(1000000):
    phi, delta = jac_state_update(phi,rho,dx)
    #phi, delta = gs_state_update(phi,l)

    if i%100 and delta < float(1/2**10):
        print(i)
        break

ex, ey, ez = elec_field(phi,dx)

z=50    #cut through the middle
for x in range(l):
    for y in range(l):

        dist = dist_to_centre(x,y,z,50)
        efield_mag = np.sqrt(ex[x,y,z]**2. + ey[x,y,z]**2. + ez[x,y,z]**2.)

        f1.write('%lf %lf %lf\n'%(dist,phi[x,y,z],efield_mag))
        f2.write('%lf %lf %lf %lf %lf %lf %lf\n'%(x,y,z,phi[x,y,z],ex[x,y,z],ey[x,y,z],ez[x,y,z]))
        f2.write('\n')

        # f.write('%lf %lf %lf %lf %lf %lf %lf %lf\n'%(x, y, z, dist, phi[x,y,z], ex[x,y,z], ey[x,y,z], ez[x,y,z]))
        # f.write('\n')

f1.close()
f2.close()
