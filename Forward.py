'''
Forward method to solve 1D reaction-diffusion equation:
    u_t = D * u_xx + alpha * u
    
with Dirichlet boundary conditions u(x0,t) = 0, u(xL,t) = 0
and initial condition u(x,0) = 4*x - 4*x**2
'''


import numpy as np
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import pi, exp, sin

def getModelPrice(params):
    alpha =params[0]
    beta=params[1]
    chi =params[2]
    delta=params[3]
    
    M = 9 # GRID POINTS on space interval
    N = 98 # GRID POINTS on time interval
    
    x0 = 170
    xL = 210
    
    # ----- Spatial discretization step -----
    dx = (xL - x0)/(M - 1)
    
    t0 = 0
    tF = 96
    K =  190
    r = 0.0245
    q = 0.005
    
    # ----- Time step -----
    dt = (tF - t0)/(N - 1)
    
    D = 0.1  # Diffusion coefficient
    alpha = -3 # Reaction rate
    
    r = dt*D/dx**2
    s = dt*alpha;
    
    
    # ----- Creates grids -----
    xspan = np.linspace(x0, xL, M)
    tspan = np.linspace(t0, tF, N)
    
    # ----- Initializes matrix solution U -----
    U = np.zeros((M, N))
    sigsq = np.ones((M,N))
    for i in range(0,M):
        for j in range(0,N):
            sigsq[i,j] = alpha*(exp(pi*i*beta)-exp(-pi*i)*chi)*sin(pi*j*delta)
    
    # ----- Initial condition -----
    U[:,0] = 4*xspan - 4*xspan**2
    
    # ----- Dirichlet Boundary Conditions -----
    U[0,:] = 1.0
    U[-1,:] = 1.0
    
    # ----- Dupire PDE -----
    for k in range(0, N-1):
        for i in range(1, M-1):
            U[i, k+1] = dt*(U[i,k]/dt+0.5*sigsq[i,k]*K**2/dx*(U[i-1,k]-2*U[i,k]+U[i+1,k])-(r-q)*K/dx*(U[i+1,k]-U[i,k])-q*U[i,k])
            #r*U[i-1, k] + (1-2*r+s)*U[i,k] + r*U[i+1,k] 
#    
#    X, T = np.meshgrid(tspan, xspan)
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    surf = ax.plot_surface(X, T, U, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    
#    #ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
#    
#    ax.set_xlabel('Space')
#    ax.set_ylabel('Time')
#    ax.set_zlabel('U')
#    plt.tight_layout()
#    plt.show()
    return np.log(U.transpose())/17.5