#----------------------------------------------Import--------------------------------------------------
import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.getcwd()+os.sep+"..")

import nuphysim
import numpy as np

# Plotting import and settinngs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator

linewidth = 5.92765763889 # inch
plt.rcParams["figure.figsize"] = (1.61*linewidth, linewidth)
plt.rcParams['figure.dpi'] = 256
plt.rcParams['font.size'] = 16
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

#-------------------------------------------Set Ouput file---------------------------------------------
filename = sys.argv[1]+sys.argv[2]

with open(filename+".out", 'a') as f:
    f.write('Simulation type:'+ str(sys.argv[2])+'\n')
    f.write('\n')

#----------------------------------------------Initialisation-------------------------------------------
Tt, Nt = float(sys.argv[3]), int(sys.argv[4])
dt = Tt/Nt

constraint = nuphysim.patchsim.Constraints(str(sys.argv[2]), dt, Tt, Nt)


#============================================================================================================
#                                        Set Rayleighian Constants
#============================================================================================================
k_b = 1.38e-23
T = 310

K_b = 25*k_b*T
H_0 = (1/2)*1e6
d = 4e-9
A0 = 57.0*(1e-10)**2
gamma_0 =  10*1e-3
Pa = 0.5*1e6
lambda_0 = 1e-4
eta_s = 1e-4 #1e2

a_0 = 50*(1e-10)**2
rho_c = 1496 
k_on = 7e4
k_off = 4e4
k_c = k_on/k_off
c_m = 0.014*rho_c
tilde_c_m = 1 #tilde(c_m,k_c)
c_0 = 1 # c_m
mu_0a = 1.5*k_b*T
mu_0b = k_b*T
xi = 25
R = 8.314

qe = 1.61e-19
epsilon = 3.5*8.85e-12
q = 0 # 1
rho_m =  0 # 0.5/(1e-9)**2
R_m = 1e9  #1e-3
C_m = 0.5e-4
C_D= 0.003
phi_0 = 1e-9


#============================================================================================================
#                                 Geometric parameters for simulation
#============================================================================================================
A = lambda H: 4 * np.pi * ((1e6) * (1 / H))**2 
H0  = 4.9e9                           # m-1    # Curvature
rad = (1e6) * (1 / H0)                # um     # Cell radius
A_m_0 = A(H0)                         # um2    # Initial area
h_m_0 = 4e-3                          # um     # Membrane thickness


#============================================================================================================
#                                 Electrophysiological parameters for simulation
#============================================================================================================
E_rest = -49.1 # mV
E_K = -92.34   # mV
E_Na = 62.94   # mV  
E_Leak = -54.3 # mV

G_K = 8.4e-5        # nS/um    
G_Na_fast = 1.41e-3 # nS/um
G_Na_slow = 2.76e-4 # nS/um
G_Leak = 5.6e-6     # nS/um

capa = 4.43e-5 # pF/um
cappar = (capa*A0/d)


#==========================================================================================================
#                           Preallocate the variables and calculate the initial values
#==========================================================================================================
i_inj = 100 #8e-8
I = np.zeros(Nt)
I[int(Nt*0.65):int(Nt*0.8)] = i_inj

arg  = (K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d)


#-----------------------------------------------Minimisation------------------------------------------
X, Xerr, V, t = nuphysim.patchsim.minimiser(H0, E_rest, I, Tt, Nt, arg, constraint, filename, verbose=True)



#-------------------------------------------Plot Data------------------------------------------
fig, ax = plt.subplots(5,1, figsize = (1.61*linewidth/2, 2.2*linewidth/2))

ax[0].plot(t[2:2300]*1e6, (X[2:2300,0]), label = r'$H$ (nm^1)') 
ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
ax[0].set_xlabel(r't ($\mu$s)')
ax[0].set_ylabel('$\Delta$ H (m$^{-1}$)', rotation = 90)
# ax[0].legend()

ax[1].plot(t[2:2300]*1e6, nuphysim.patchsim.tilde(X[2:2300,1], c_0), color =  ax[0]._get_lines.get_next_color(), label = r'$c$')
ax[1].set_xlabel(r't ($\mu$s)')
ax[1].set_ylabel(r'$\tilde{c}$', rotation = 90)
ax[1].set_ylim(0, ax[1].get_ylim()[1])

# ax1 = ax[1].twinx()
# ax1.plot(t[2:], tilde(X[2:,1], c_0), ':', color =  ax[1].lines[-1].get_color(), label = r'$\tilde{c}$')
# ax1.set_ylabel(r'$\tilde{c}$', rotation = 0)
# ax1.set_ylim(0, ax1.get_ylim()[1])

# handles0, labels0 = ax[1].get_legend_handles_labels()
# handles1, labels1 = ax1.get_legend_handles_labels()
# ax[1].legend(handles0+handles1, labels0+labels1, loc=[0.75, 0.4], frameon=False)

ax[2].plot(t[2:2300]*1e6, X[2:2300,2], color =  ax[0]._get_lines.get_next_color(), label = r'$\phi$')
ax[2].set_xlabel(r't ($\mu$s)', rotation = 0)
ax[2].set_ylabel(r'$\phi (V)$', rotation = 90)
ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1])
# ax[2].legend()
# ax[2].set_ylim(0,10000)

ax[3].plot(t[2:2300]*1e6, I[2:2300], color =  ax[0]._get_lines.get_next_color(), label = r'$I$')
ax[3].set_xlabel(r't ($\mu$s)', rotation = 0)
ax[3].set_ylabel(r'$I (A)$', rotation = 90)
ax[3].set_ylim(ax[3].get_ylim()[0], ax[3].get_ylim()[1])
# ax[3].legend()
# ax[3].set_ylim(0,10000)

ax[4].plot(t[2:2300]*1e6, V[2:2300], color =  ax[0]._get_lines.get_next_color(), label = r'$V$')
ax[4].set_xlabel(r't ($\mu$s)', rotation = 0)
ax[4].set_ylabel(r'$V (mV)$', rotation = 90)
# ax[4].set_ylim(ax[4].get_ylim()[0], ax[4].get_ylim()[1])
# ax[4].legend()
# ax[4].set_ylim(0,10000)

fig.tight_layout()
fig.savefig('Figures'+ os.sep + filename+'.pdf', bbox_inches = 'tight')




 #============================================================================================================
# #                                        Plot the Voltage vs. Time
# #============================================================================================================
# plt.figure(1)
# plt.plot(t, V, linewidth=2)
# plt.xlabel('Time (ms)')
# plt.ylabel('V_m (mV)')
# plt.gca().tick_params(axis='both', labelsize=16)
# plt.show()

# #============================================================================================================
# #                                       Plot the Conductance vs. Time
# #============================================================================================================
# plt.figure(2)
# p1, = plt.plot(t, ((G_Na_fast + G_Na_slow) * A0 / h_m) * (m ** 3) * h, 'g', linewidth=3)
# p2, = plt.plot(t, (G_K * A0 / h_m) * n ** 4, 'm', linewidth=3)
# p3, = plt.plot(t, G_Leak * A / h_m, 'b', linewidth=3)
# plt.xlabel('time (ms)')
# plt.ylabel('Conductance')
# plt.title('Conductances for Sodium, Potassium and the Leak')
# plt.show()
# ```

