#----------------------------------------------Import--------------------------------------------------
import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.getcwd()+os.sep+"..")

import nuphysim
import numpy as np

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
I[int(Nt*0.45):int(Nt*0.65)] = i_inj

arg  = (K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d)


#-----------------------------------------------Minimisation------------------------------------------
X, Xerr, V, t = nuphysim.patchsim.minimiser(H0, E_rest, I, Tt, Nt, arg, constraint, filename, verbose=True)
nuphysim.patchsim.minimserPlot(X, V, I, t, k_c, filename)

# X, t = nuphysim.patchsim.FDsimulation([H0, 0.2, E_rest], I, Tt, Nt, arg, filename, verbose=True)
# nuphysim.patchsim.FDPlot(X, I, t, filename)