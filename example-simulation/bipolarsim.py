import os
import sys
import time
import subprocess
from datetime import timedelta

import numpy as np
from numba import njit

# Plotting import and settinngs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator

from matplotlib.animation import FuncAnimation, PillowWriter


linewidth = 5.92765763889 # inch
plt.rcParams["figure.figsize"] = (1.61*linewidth, linewidth)
plt.rcParams['figure.dpi'] = 256
plt.rcParams['font.size'] = 16
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Constants and grid parameters (these should be defined based on your problem)
M = 100  # number of spatial grid points
N = 100  # number of time steps
h0 = 0.01  # spatial step size
tau = 0.0001  # time step size

# Assuming you have the spatial points array for N(x)
x = np.linspace(0, 1, M)  # Spatial grid points
t = tau*np.arange(0,N)

vn, vp = 10.0, 10.0  # velocity of current carriers for electrons and holes
b_n, b_p = 1.0, 1.0
m_n, m_p = 1.0, 1.0

V, epsilion, q = 10, 8.85418782e-12, 1.6e-19
# Assuming hi is a uniform grid step size equal to h
h = h0*np.ones(M)  # spatial step size for each point


# Function N(x), here we assume a constant or you can define it as needed
N_a, N_d = 2e17, 4e17
x1,x2,x3 = 0,0.5,1

def Ni(x):
    return np.where(x < x2,-N_a,N_d)


# Initialize concentrations (example, these should be defined for your problem)
n = np.zeros([N,M])  # electron concentration
n[0,:int(N*0.4)]  = 1e6*np.ones(int(N*0.4))

p = np.zeros([N,M])  # hole concentration
p[0,-int(N*0.4):] = 1e6*np.ones(int(N*0.4))

# Initial conditions and arrays
phi = np.zeros([N,M])  # potential
phi[0] = 0.5*(1+np.tanh(np.pi*x))

E = np.zeros([N,M])  # electric field
E[0] = np.array([-xi if xi <x2 else xi-2*x2 for xi in x ]) # np.linspace(0,1,M) #

Jn = np.zeros([N,M])  # electron current density
Jp = np.zeros([N,M])  # hole current density
J_bias = np.zeros([N,M])  # bias current
J_total = np.zeros([N,M])  # total current

# Coefficients alpha_n and alpha_p
alpha_n = np.zeros([N,M])
alpha_p = np.zeros([N,M])

# Coefficients from the equations
A_phi = 1 / np.copy(h)**2  # A_phi = 1 / h_i^2
B_phi = np.copy(A_phi)  # B_phi is the same as A_phi
C_phi = 2 * np.copy(A_phi)  # C_phi is also the same as A_phi

A_n = 1 - (h/ (vn * tau))  # alpha_n = 1 - h_i / (vn * tau)
A_p = 1 + (h/ (vp * tau))  # alpha_p = 1 + h_i / (vp * tau)

# F_j calculation: F_j = N(x) + p^j_i - n^j_i

printProgressBar(0, N-1, prefix = 'Progress:', suffix = 'Complete', length = 50)
t0 = time.time()

# Update loop over time steps
for j in range(N-1):
    A = np.diag(np.ones(M))-2*np.diag(np.ones(M-1),1)+np.diag(np.ones(M-2),2)

    # Update F_j at each time step
    F_ij = Ni(x) + p[j] - n[j]
    # phi[j+1] = np.roll( 2*phi[j+1] - np.roll(phi[j+1],1) - tau*(q/epsilion)*F_ij, 1)

    phi[j+1] =  tau*(q/epsilion) * np.linalg.inv(A)@F_ij
    # phi[j+1,0], phi[j+1,-1] = V, 0
    # phi[j+1,int(0.33*N)], phi[j+1,int(0.66*N)] = V, 0

    # Update electric field E
    E[j+1] = (np.roll(phi[j+1],-1) - phi[j+1]) / h
    E[j+1,0],E[j+1,-1] = 0, 0
    # E[j+1,int(0.33*N)],E[j+1,int(0.66*N)] = 0, 0

    # Coefficients alpha_n and alpha_p
    alpha_n[j+1] = A_n*np.exp(-(b_n/E[j+1])**m_n)
    alpha_p[j+1] = A_p*np.exp(-(b_p/E[j+1])**m_p)

    # Update Jn (electron current) and Jp (hole current)
    # Jn[j+1] = Jn[j] + tau*vn*( ( np.roll(Jn[j],-1) - Jn[j] )/h + q*(alpha_n[j] * Jn[j] + alpha_p[j+1] * Jp[j+1]))
    # Jp[j+1] = Jp[j] + tau*vp*( ( np.roll(Jp[j],-1) - Jp[j] )/h + q*(alpha_n[j] * Jn[j] + alpha_p[j+1] * Jp[j+1]))
    # n[j+1]  = Jn[j+1] /(vn)
    # p[j+1]  = Jp[j+1] /(vp)

    a1 = np.array([[h[i]+vn*tau*(1-q*h[i]*alpha_n[j+1][i]), h[i]-vn*tau*(1+q*h[i]*alpha_p[j+1][i])] for i in range(M) ]).flatten()
    b1 = np.array([[-q*tau*h[i]*vp*alpha_p[j+1][i], 0] for i in range(M) ]).flatten()[:-1]
    b2 = np.array([[-q*tau*h[i]*vn*alpha_n[j+1][i], 0] for i in range(M) ]).flatten()[:-1]
    a2 = np.array([[[vn*tau,vp*tau]] for i in range(M)]).flatten()[:-2]

    Mi = np.diag(a1) + np.diag(b1,-1) + np.diag(b2,1) + np.diag(a2,2)
     
    c_0 = np.array([[n[j,i], p[j,i]] for i in range(M) ]).flatten()
    c_1 = h[0]*(np.linalg.inv(Mi)@c_0)

    n[j+1] = c_1[::2]
    p[j+1] = c_1[1::2]

    Jn[j+1] = q*(vn)*n[j+1]
    Jp[j+1] = q*(vp)*p[j+1]

    # Update bias current
    J_bias[j+1] = epsilion*(E[j+1] - E[j]) / tau

    # Total current
    J_total[j+1] = Jn[j+1] + Jp[j+1] + J_bias[j+1]


    if j%int(0.1*N) == 0 :
        t1 = time.time() 
        deltat = (t1-t0)/(j+1)

    tprogress = timedelta(seconds=(N-1-j)*deltat) 

    printProgressBar(j+1, N-1, prefix = 'Progress', suffix = 'Complete | Time left- '+str(tprogress).split('.')[0], length = 50)       




# Now, you can plot the results or analyze the arrays for each time step
#-------------------------------------------Plot Data------------------------------------------
fig, ax = plt.subplots(3,2, figsize = (1.61*linewidth/2, 2.2*linewidth/2))

im = ax[0,0].imshow(n)
ax[0,0].set_ylabel(r't ($\mu$s)', rotation = 90)
ax[0,0].set_xlabel(r'$n$ ')

ax[0,1].imshow(p)
ax[0,1].set_ylabel(r't ($\mu$s)', rotation = 90)
ax[0,1].set_xlabel(r'$p$ ')

ax[1,0].imshow(J_total)
ax[1,0].set_ylabel(r't ($\mu$s)', rotation = 90)
ax[1,0].set_xlabel(r'$J$ ')


ax[1,1].imshow(E)
ax[1,1].set_ylabel(r't ($\mu$s)', rotation = 90)
ax[1,1].set_xlabel(r'$E$ ')


ax[2,0].imshow(phi)
ax[2,0].set_ylabel(r't ($\mu$s)', rotation = 90)
ax[2,0].set_xlabel(r'$\phi$ ')

cbar= fig.colorbar(im, ax= ax , orientation = 'vertical', fraction=0.01675, pad=0.025)
# cbar.set_ticks(np.array([0, 0.01, 0.03]))
# cbar.set_label(r'$\frac{F}{E*R^2}$', rotation=0)
# cbar.ax.yaxis.set_label_coords(7.5, 0.5)
# cbar.ax.set_ylim(0, 0.03)
# cbar.minorticks_on() 

# fig.tight_layout()
fig.savefig('Figures'+os.sep+'Bipolar-heatmap.pdf')#, bbox_inches = 'tight')


def animate(i, ax, x, f, strf):
    ax.clear()
    wdth = abs( np.max(f) - np.min(f) )
    ax.set_ylim(np.min(f)-0.3*wdth, np.max(f)+0.3*wdth)
    ax.set_xlabel(r'x ($\mu$s)')
    ax.set_ylabel(r'${0}$ '.format(strf), rotation = 0)

    line = ax.plot(x,f[i])
    ax.plot(x,p[i])
    return line
 
fig,ax = plt.subplots()
       
ani = FuncAnimation(fig, animate, fargs=(ax, x, n, 'n'), interval=N, blit=True, repeat=True, frames=N)    
ani.save('Figures'+os.sep+"Bipolar-n.gif", dpi=300, writer=PillowWriter(fps=25))

ani = FuncAnimation(fig, animate, fargs=(ax, x, p, 'p'), interval=N, blit=True, repeat=True, frames=N)    
ani.save('Figures'+os.sep+"Bipolar-p.gif", dpi=300, writer=PillowWriter(fps=25))

ani = FuncAnimation(fig, animate, fargs=(ax, x, E, 'E'), interval=N, blit=True, repeat=True, frames=N)    
ani.save('Figures'+os.sep+"Bipolar-E.gif", dpi=300, writer=PillowWriter(fps=25))

ani = FuncAnimation(fig, animate, fargs=(ax, x, phi, 'phi'), interval=N, blit=True, repeat=True, frames=N)    
ani.save('Figures'+os.sep+"Bipolar-phi.gif", dpi=300, writer=PillowWriter(fps=25))

ani = FuncAnimation(fig, animate, fargs=(ax, x, J_total, 'J'), interval=N, blit=True, repeat=True, frames=N)    
ani.save('Figures'+os.sep+"Bipolar-J.gif", dpi=300, writer=PillowWriter(fps=25))






# #-------------------------------------------Plot Data------------------------------------------
# fig, ax = plt.subplots(4,1, figsize = (1.61*linewidth/2, 2.2*linewidth/2))
# for i in range(M)[::10]:
#     ax[0].plot(t,n[:,i])
#     # ax[0].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[0].set_xlabel(r't ($\mu$s)')
#     ax[0].set_ylabel(r'$n$ ', rotation = 0)
#     # # ax[0,0].legend()

#     ax[1].plot(t,p[:,i])
#     # ax[1].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[1].set_xlabel(r't ($\mu$s)')
#     ax[1].set_ylabel(r'$p$ ', rotation = 0)
#     # # ax[1].legend()

#     ax[2].plot(t,phi[:,i])
#     # ax[2].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[2].set_xlabel(r't ($\mu$s)')
#     ax[2].set_ylabel(r'$\phi$ ', rotation = 0)
#     # # ax[2].legend()

#     ax[3].plot(t,E[:,i])
#     # ax[3].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[3].set_xlabel(r't ($\mu$s)')
#     ax[3].set_ylabel(r'$E$ ', rotation = 0)
#     # # ax[3].legend()

# fig.tight_layout()
# fig.savefig('Figures'+os.sep+'Bipolar-timePlots.pdf', bbox_inches = 'tight')



# #-------------------------------------------Plot Data------------------------------------------
# fig, ax = plt.subplots(4,1, figsize = (1.61*linewidth/2, 2.2*linewidth/2))
# for i in range(N)[::10]:
#     ax[0].plot(x,n[i])
#     # ax[0].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[0].set_xlabel(r'x ($\mu$s)')
#     ax[0].set_ylabel(r'$n$ ', rotation = 0)
#     # # ax[0,0].legend()

#     ax[1].plot(x,p[i])
#     # ax[1].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[1].set_xlabel(r'x ($\mu$s)')
#     ax[1].set_ylabel(r'$p$ ', rotation = 0)
#     # # ax[1].legend()

#     ax[2].plot(x,phi[i])
#     # ax[2].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[2].set_xlabel(r'x ($\mu$s)')
#     ax[2].set_ylabel(r'$\phi$ ', rotation = 0)
#     # # ax[2].legend()

#     ax[3].plot(x,E[i])
#     # ax[3].set_ylim(ax[0,0].get_ylim()[0], ax[0].get_ylim()[1])
#     ax[3].set_xlabel(r'x ($\mu$s)')
#     ax[3].set_ylabel(r'$E$ ', rotation = 0)
#     # # ax[3].legend()

# fig.tight_layout()
# fig.savefig('Figures'+os.sep+'Bipolar-spacialPlots.pdf', bbox_inches = 'tight')






    # # Update phi using the finite difference equation
    # for i in range(1, M-1):
    #     # Update F_j at each time step
    #     F_ij = Ni(x[i]) + p[j,i] - n[j,i]
    #     phi[j+1,i+1] = -( A_phi[i] * phi[j+1,i-1] - C_phi[i] * phi[j+1,i] + F_ij ) /B_phi[i]

    #     # Update electric field E
    #     E[j+1,i] = (phi[j+1, i-1] - phi[j+1,i]) / h[i]

    #     # Coefficients alpha_n and alpha_p
    #     alpha_n[j+1,i] = A_n[i]*np.exp(-(b_n/E[j+1,i])**m_n)
    #     alpha_p[j+1,i] = A_p[i]*np.exp(-(b_p/E[j+1,i])**m_p)

    #     # Update Jn (electron current) and Jp (hole current)
    #     Jn[j+1,i+1] = (A_n[i] * Jn[j,i] + (h[i] / (vn * tau)) * Jn[j,i] + h[i] * (alpha_n[j+1,i] * Jn[j+1,i] + alpha_p[j+1,i] * Jp[j+1,i]))
    #     Jp[j+1,i+1] = (A_p[i] * Jp[j,i] + (h[i] / (vp * tau)) * Jp[j,i] + h[i] * (alpha_n[j+1,i] * Jn[j+1,i] + alpha_p[j+1,i] * Jp[j+1,i]))

    #     # Update bias current
    #     J_bias[j+1, i+1] = (E[j+1,i] - E[j,i]) / tau

    #     n[j+1] = Jn[j+1] /(vn)
    #     p[j+1] = Jp[j+1] /(vp)

    # # Total current
    # J_total[j+1] = Jn[j+1] + Jp[j+1] + J_bias[j+1]



    # # Update F_j at each time step
    # F_ij = Ni(x) + p[j] - n[j]
    # phi[j] = np.roll( 2*phi[j] - np.roll(phi[j],1) - dt*(q/epsilon)*F_ij, 1)

    # # Update electric field E
    # E[j] = (np.roll(phi[j],-1) - phi[j]) / h

    # # Coefficients alpha_n and alpha_p
    # alpha_n[j] = A_n * np.exp( -( b_n/E[j])**m_n )
    # alpha_p[j] = A_p * np.exp( -( b_p/E[j])**m_p )

    # # Update Jn (electron current) and Jp (hole current)
    # (1+dt*vn/h) * Jn[j+1] = Jn[j] + (dt*vn/h)*np.roll(Jn[j+1],-1) + q*(dt*vn/h)*(alpha_n[j+1] * Jn[j+1] + alpha_p[j+1] * Jp[j+1])
    # (1+dt*vn/h) * Jp[j+1] = Jp[j] + (dt*vp/h)*np.roll(Jp[j+1],-1) + q*(dt*vn/h)*(alpha_n[j+1] * Jn[j+1] + alpha_p[j+1] * Jp[j+1])

    # # Update bias current
    # J_bias[j+1] = epsilion*(E[j+1,i] - E[j,i]) / tau

    # n[j+1] = Jn[j+1] /(vn)
    # p[j+1] = Jp[j+1] /(vp)

    # # Total current
    # J_total[j+1] = Jn[j+1] + Jp[j+1] + J_bias[j+1]

    