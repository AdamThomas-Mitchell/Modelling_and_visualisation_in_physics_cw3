import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def init_sys(phi0,l):
    '''
    Initialises array with small amount of noise
    '''
    sys = np.full((l,l), phi0) + np.random.uniform(-0.1,0.1,(l,l))
    return sys

def state_update(phi):
    '''
    Vectorised state update     #maybe make this nicer at some point
    '''
    phi = np.copy(phi)

    part1 = -a*(np.roll(phi,+1,axis=0) + np.roll(phi,-1,axis=0) + np.roll(phi,+1,axis=1) + np.roll(phi,-1,axis=1) - 4.0*phi)

    part2 = a*(np.roll(phi,+1,axis=0)**3.0 + np.roll(phi,-1,axis=0)**3.0 + np.roll(phi,+1,axis=1)**3.0 + np.roll(phi,-1,axis=1)**3.0 - 4.0*phi**3.0)

    part3 = -(k/del_x**2.0)*(20.0*phi - 8.0*np.roll(phi,1,axis=1) - 8.0*np.roll(phi,-1,axis=1) - 8.0*np.roll(phi,1,axis=0) - 8.0*np.roll(phi,-1,axis=0) + np.roll(phi,2,axis=1) + np.roll(phi,-2,axis=1) + np.roll(phi,2,axis=0) + np.roll(phi,-2,axis=0) + 2.0*np.roll(np.roll(phi,-1,axis=0),-1,axis=1) + 2.0*np.roll(np.roll(phi,1,axis=0),-1,axis=1) + 2.0*np.roll(np.roll(phi,-1,axis=0),1,axis=1) + 2.0*np.roll(np.roll(phi,1,axis=0),1,axis=1))

    new_phi = phi + (M*del_t/del_x**2.0)*(part1 + part2 + part3)

    return new_phi

def free_energy_density(phi):
    '''
    Calculates free energy density for state
    '''
    bracket = (np.roll(phi,+1,axis=1) - np.roll(phi,-1,axis=1))**2.0 + (np.roll(phi,+1,axis=0) - np.roll(phi,-1,axis=0))**2.0

    f = -(a/2.0)*phi**2.0 + (a/4.0)*phi**4.0 + (k/8.0*del_x**2.0)*bracket

    return np.sum(f)

### INPUT ###
if(len(sys.argv) != 8):
    print("Usage python ch_animation.py N phi0 dx dt a k M")
    sys.exit()
l=int(sys.argv[1])
phi0=float(sys.argv[2])
del_x=float(sys.argv[3])
del_t=float(sys.argv[4])
a=float(sys.argv[5])
k=float(sys.argv[6])
M=float(sys.argv[7])

#ask user for animation or data analysis
choice = input("Type (1) for animation or (2) for data collection.     ")
method=None
if choice == "1":
    method=1
elif choice == "2":
    method=2
else:
    print('Select (1) or (2)')

###MAIN LOOP###
phi = init_sys(phi0, l)

if method==1:
    fig=plt.figure()
    im=plt.imshow(phi, animated=True, cmap='RdBu')
    plt.colorbar(extend='both')
    plt.clim(-1, 1);

if method==2:
    g=open('ch_data.dat', 'w')

for i in range(100000):
    phi = state_update(phi)

    if method==1:
        if i%100==0:
            print(i)
            plt.cla()
            im=plt.imshow(phi, interpolation='mitchell', animated=True, cmap='RdBu')
            plt.draw()
            plt.pause(0.0001)

    if method==2:
        if i%1000==0:
            f = free_energy_density(phi)
            g.write('%lf %lf\n'%(i, f))

g.close()
