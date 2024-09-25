import argparse
import sys
import os

sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.getcwd()+os.sep+"..")
cwd = os.getcwd()

parser = argparse.ArgumentParser(prog ="Finite Difference Simulation of Neuronal Patch",
                                 description = "The code can be used to simulate the mechno-electrophysiological and biochemical coupling in neurons. \n This code implements a finite difference model for simulating neural dynamics with the incorporation of neuromodulation constraints. The model is implemented in Python using scientific computing libraries such as NumPy, SciPy, and Numba. The code includes functions to calculate various model parameters, generate ultrasound pulses, and perform optimization using the Rayleighian function.",
                                 epilog ='See documentation: https://neuron-multiphysics-simulations.readthedocs.io/en/latest/index.html for further help',)

parser.add_argument('simulator',  metavar='simulator', action='store',      help='Simulator to run, either Finite Difference iterator or Rayleighian minimiser [FD or minimiser]')          
parser.add_argument('-v', '--verbose',                 action='store_true', help='Print output at each step')  
parser.add_argument('-e', '--explicit',                action='store_true', help='Set simulation type for minimiser as explicit (cannot be given alongside -e)')
parser.add_argument('-i', '--implicit',                action='store_true', help='Set simulation type for minimiser as implicit (cannot be given alongside -i)')
parser.add_argument('-o', '--output',      metavar='', action='store', default= cwd,   type=str,   help='Output directory')  
parser.add_argument('-n', '--num',         metavar='', action='store', default='0',    type=str,   help='Simulation number for data storage')  
parser.add_argument('-c', '--constraint',  metavar='', action='store', default='None', type=str,   help='Constraint method to apply constraints during minimization')  
parser.add_argument('-I', '--Injection',   metavar='', action='store', default='None', type=str,   help='Injected current used in simulation')  
parser.add_argument('-Nt', '--NSteps',     metavar='', action='store', default=500,    type=int,   help='Number of time steps (int)')  
parser.add_argument('-Tt', '--Total_time', metavar='', action='store', default=1e-6,   type=float, help='Total time for the simulation (float)')    

arg = parser.parse_args()

if arg.simulator == 'FD':
    simType = 'Finite Difference Iterator'
    filename = arg.num + arg.simulator

elif arg.simulator == 'minimiser':
    filename = arg.num + arg.constraint
    if arg.implicit == False and arg.explicit == False:
        solveType = 'implicit'    
        simType = 'Implicit minimisation'
    elif arg.implicit == True and arg.explicit == False:
        solveType = 'implicit'    
        simType = 'Implicit minimisation'
    elif arg.implicit == False and arg.explicit == True:
        solveType = 'explicit'    
        simType = 'Explicit minimisation'
    elif arg.implicit == True and arg.explicit == True:
        raise Exception('Cannot select both implicit and explicit simulation type')
    else:
        raise Exception('Error with selected simulation type, see help')

#----------------------------------------------Import--------------------------------------------------
import nuphysim
import numpy as np

#----------------------------------------------Initialisation-------------------------------------------
constraintType = arg.constraint
Tt, Nt = arg.Total_time, arg.NSteps
dt = Tt/Nt

#-------------------------------------------Set Ouput file---------------------------------------------
with open(arg.output+os.sep+"out"+os.sep+filename+".out", 'a') as f:
    f.write('Simulation info:\n')
    f.write('   Simulation type -'+ simType +'\n')
    f.write('   Constraint - '+ constraintType+'\n')
    f.write('   Duration - '+ str(Nt) +' Steps / '+ str(Tt*1e6) + ' milliseconds \n')
    f.write('\n')

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
H0  = 1.6e6                           # m-1    # Curvature
rad = (1e6) * (1 / H0)                # um     # Cell radius
A_m_0 = A(H0)                         # um2    # Initial area
h_m_0 = 4e-3                          # um     # Membrane thickness

#============================================================================================================
#                                 Electrophysiological parameters for simulation
#============================================================================================================
E_rest = -0.079 #-49.1 # mV
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
I = nuphysim.I_inj(arg.Injection, Nt)

args = (K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d)

constraints = nuphysim.patchsim.Constraints(constraintType, dt, Tt, Nt)

if arg.simulator == 'minimiser':
    #-----------------------------------------------Minimisation------------------------------------------
    X, Xerr, V, t = nuphysim.patchsim.minimiser(H0, E_rest, I, Tt, Nt, args, constraints, filename, cwd = arg.output,  solveType = solveType, verbose=arg.verbose)
    nuphysim.patchsim.minimserPlot(X, V, I, t, k_c, filename, cwd = arg.output)

elif arg.simulator == 'FD':
    X, t, terms= nuphysim.patchsim.FDsimulation([H0, 0, E_rest], I, Tt, Nt, args, filename, cwd = arg.output, verbose=arg.verbose)
    nuphysim.patchsim.FDPlot(X, I, t, filename, terms, cwd = arg.output)

else:
    raise Exception("No valid simulator given: try positional argument 'minimiser' or 'FD'")