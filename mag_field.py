import numpy as np
import math

def init_sys(l):
    sys = np.zeros((l,l))

    return sys

def jacobi2d(phi,j,dx):
    '''
    update using 2D vectorised jacobi algorithm
    '''
    phi = np.copy(phi)

    new_phi = (1.0/4.0) * (np.roll(phi,+1,axis=0) + np.roll(phi,-1,axis=0) + np.roll(phi,+1,axis=1) + np.roll(phi,-1,axis=1) + (dx**2.0)*j )

    #boundary conditions
    new_phi[0,:]=0
    new_phi[:,0]=0
    new_phi[-1,:]=0
    new_phi[:,-1]=0

    return new_phi

def jac2d_state_update(phi,j,dx):
    '''
    Updates state and computes difference between the 2 times
    '''
    new_phi = jacobi2d(phi,j,dx)
    delta = abs(np.sum(new_phi - phi))

    return new_phi, delta

def dist_to_centre(x,y,c):
    '''
    Calculates distance from point centre
    '''
    r = np.sqrt((c-x)**2. + (c-y)**2.)

    return r

def mag_field(phi,dx):
    '''
    Approximation for electric field vector
    '''
    x_comp = -1./(2.*dx) * (np.roll(phi,+1,axis=1) - np.roll(phi,-1,axis=1))
    y_comp = 1./(2.*dx) * (np.roll(phi,+1,axis=0) - np.roll(phi,-1,axis=0))

    return x_comp, y_comp

l=100
dx=1.0
c=50

phi = init_sys(l)
j = init_sys(l)
j[c,c]=1.0

f1=open('potvsR_mag.dat','w')
f2=open('mag_field.dat','w')
for i in range(1000000):
    phi, delta = jac2d_state_update(phi,j,dx)

    if i%100 and delta < float(1/2**10):
        print(i)
        break

bx, by = mag_field(phi,dx)

for x in range(l):
    for y in range(l):
        dist = dist_to_centre(x,y,c)
        bfield_mag = np.sqrt(bx[x,y]**2. + by[x,y]**2.)

        f1.write('%lf %lf %lf\n'%(dist,phi[x,y],bfield_mag))
        f2.write('%lf %lf %lf %lf %lf\n'%(x,y,phi[x,y],bx[x,y]/bfield_mag,by[x,y]/bfield_mag))
        f2.write('\n')

f1.close()
f2.close()
