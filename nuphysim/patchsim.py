# %% [markdown]
# # Introduction
# Authors: J. Giblin-Burnham
# 

# %% [markdown]
# ## Imports#----------------------------------------------Imports-----------------------------------------------
import sys
import os
import numpy as np
from numba import njit
import math
import scipy.optimize
from scipy import special
from scipy import signal

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

# %% [markdown]
#-------------------------------------------Active Functions---------------------------------------------
@njit
def gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi):
    '''Calculate the active tension.

    Args:
        tilde_c (float)  : Area fraction of chemical adsorbants.
        phi (float)      : Membrane potential.
        gamma_0 (float)  : Intrinsic tension.
        k_c (float)      : Tension parameter, equilibrium constant.
        C_D (float)      : Tension parameter, membrane diellectric capacitance.
        xi (float)       : Tension parameter, membrane chemical suceptabillity.

    Returns:
        float : Active tension.
    '''

    return gamma_0 + xi * (tilde_c**2 )/ (tilde_c - k_c) + (C_D * phi**2)/2 

@njit
def Cm(H,d, epsilon):
    ''' Calculate the membrane capacitance.

    Args:
        H (float)       : Membrane curvature.
        d (float)       : Membrane thickness.
        epsilon (float) : Relative permittivity of membrane.

    Returns:
        float : Membrane capacitance.
    '''

    return (epsilon/d)*(1-((H*d)**2)/4)

@njit
def Cm_dot(H, H_dot, d, epsilon):
    '''
    Calculate the derivative of the membrane capacitance.

    Args:
        H (float)       : Membrane curvature.
        H_dot (float)   : Time derivative of membrane curvature.
        d (float)       : Membrane thickness.
        epsilon (float) : Relative permittivity of membrane.

    Returns:
        float: Time derivative of the membrane capacitance.
    '''
    return -epsilon*d*H*H_dot/2

@njit
def mu(tilde_c, tilde_c_m, mu_0a, R, T, a_0):
    '''Calculate the chemical potential of membrane.

    Args:
        tilde_c (float)   : Area fraction of chemical adsorbants.
        tilde_c_m (float) : Maximum area fraction of chemical adsorbants.
        mu_0a (float)     : Baseline chemical potential of membrane.
        R (float)         : Universal gas constant.
        T (float)         : Temperature.
        a_0 (float)       : Molar area.

    Returns:
        float : Surface chemical potential.
    '''

    return mu_0a/a_0  +  (R*T/a_0) * ( tilde_c and np.log( tilde_c / (tilde_c_m - tilde_c)) ) 

@njit
def mu_a(H, tilde_c, phi, args):
    '''Calculate the active chemical potential.

    Args:
        H (float)       : Membrane curvature.
        tilde_c (float) : Area fraction of chemical adsorbants.
        phi (float)     : Membrane potential.
        args (tuple)    : Surface parameters.

    Returns:
        float : Active chemical potential.
    '''

    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q,  C_D, R, T, A = args
    return mu(tilde_c, tilde_c_m, mu_0a, R, T, a_0) + ((q*phi)/(2*a_0) )* phi - K_b * H_0 * (2 * H - H_0 * tilde_c)

@njit
def mu_b(c, c_0, mu_0b, R, T, a_0):
    '''Calculate the bulk chemical potential.

    Args:
        c (float)     : Concentration of chemical absorbates in bulk.
        c_0 (float)   : Reference bulk concentration.
        mu_0b (float) : Baseline chemical potential.
        R (float)     : Universal gas constant.
        T (float)     : Temperature.
        a_0 (float)   : Molar area.

    Returns:
        float : Bulk chemical potential.
    '''

    return mu_0b/a_0 + R*T * ( c and np.log(c /c_0) )

@njit
def tilde_k(H, tilde_c, phi, c, args):
    '''Calculate the Arrenius constant/ relative reaction rate.

    Args:
        H (float)       : Membrane curvature.
        tilde_c (float) : Area fraction of chemical adsorbants.
        phi (float)     : Membrane potential.
        c (float)       : Concentration of chemical absorbates in bulk.
        args (tuple)    : Surface parameters.

    Returns:
        float : Reaction rate.
    '''

    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q,  C_D, R, T, A = args
    return k_off*(  (c/c_0)*(tilde_c_m-tilde_c) - np.exp(-(mu_0a-mu_0b)/R*T)   )/(a_0* (mu(tilde_c, tilde_c_m, mu_0a, R, T, a_0) - mu_b(c, c_0, mu_0b, R, T, a_0)))

@njit
def tilde(c,k_c):
    '''Calculate the Area fraction of chemical adsorbants.

    Args:
        c (float)   : Concentration of chemical absorbates in bulk.
        k_c (float) : Equilbrium constant.

    Returns:
        float : Area fraction of chemical adsorbants.
    '''

    return c*k_c/(1+c*k_c)



#----------------------------------------Rayleighian Functions-------------------------------------------
# @njit
def Rayleighian(X_n, X, dt, args):
    '''Calculate the Rayleighian function.

    Args:
        X (array)    : Current state variables (Curvature H, Area fraction tilde_c, Potential phi).
        X_n (array)  : Previous state variables (Curvature H, Area fraction tilde_c, Potential phi).
        dt (float)   : Time step.
        args (tuple) : Surface parameters.

    Returns:
        float : Rayleighian value.
    '''

    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d, m, h,o, p,  n, I = args
    
    H, c, phi = X
    H_dot, c_dot, phi_dot = -(X-X_n)/dt
    tilde_c, tilde_c_dot  = tilde(c,k_c), tilde(c_dot,k_c)
    
    # H_n, c_n, phi_n = X_n
    # tilde_c, tilde_c_n = tilde(c,k_c), tilde(c_n,k_c)
    # H_dot, tilde_c_dot, phi_dot = (H-H_n)/dt, (tilde_c-tilde_c_n)/dt, (phi-phi_n)/dt

    C_m, C_m_dot =  Cm(H, d, epsilon), Cm_dot(H, H_dot, d, epsilon)

    # # Calculate the ionic currents
    # I_Na_fast = (G_Na_fast * A0 /d)  * (m ** 3) * h * (phi - E_Na)**2
    # I_Na_slow = (G_Na_slow * A0 /d)  * o * p *        (phi - E_Na)**2
    # I_K       = (G_K * A0 / d)   * n *            (phi - E_K)**2
    # I_Leak    = (G_Leak * A0 /d) *                (phi - E_Leak)**2
    
    I_ion = I #- (I_Na_fast + I_Na_slow + I_K + I_Leak)
    
    
    R = ( 
        2 * K_b * (2 * H - H_0 * tilde_c) * H_dot 
        - C_m * phi * phi_0 * H_dot  
        + (2 * eta_s + lambda_0)*float(H and H_dot/H)**2 
        - (phi**2 + 2 * phi_0 * phi * H) * C_m_dot 
        + (gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi) * H - Pa) * H_dot * float(H and 1/H)**2
        
        + 
        ( 0 
        + mu_a(H, tilde_c, phi, args[:-14])
        + mu_b(c, c_0, mu_0b, R, T, a_0)
        ) *tilde_c_dot 


         +  tilde_c_dot**2 /(tilde_k(H, tilde_c, phi, c, args[:-14])* a_0) 

        + 
        ( 0 
        - C_m * (phi + phi_0*H) 
        + 0.5 * (rho_0 + (q * tilde_c)/a_0)
        ) * phi_dot 

        + R_m * (I_ion)**2 
        # + R_m * (C_m*phi_dot)**2   
        + R_m * (C_m*(phi_dot+phi_0*H_dot) - C_m_dot*(phi+phi_0*H))**2 
        # + R_m * (C_m*(phi_dot+phi_0*H_dot) - C_m_dot*(phi+phi_0*H) + I_ion)**2 

        ) * A0

    return R


#-------------------------------------------Ultrasound Functions-------------------------------------------
def wavePulse(t, A, k, wavelets, Tt, A_0):
    '''Generate a wave pulse.

    Args:
        t (float)      : Time variable.
        A (float)      : Amplitude.
        k (float)      : Wave number.
        wavelets (int) : Number of wavelets.
        Tt (float)     : Pulse duration.
        A_0 (float)    : Initial amplitude.

    Returns:
        float : Generated wave pulse value.
    '''

    Nwavelets = 2*wavelets + 1

    f0     = k
    fmin   = Nwavelets/Tt
    fdelta = f0*Tt/Nwavelets#1
    f1     = f0*fmin/( f0 + (math.ceil(fdelta)-fdelta)*fmin )

    t0 = 1/2*(Tt-1/f1)
    t1 = - Tt/2  - 3/(2*f1) + 3/f1*(wavelets)

    return A_0 + A/2 * ( 1-int(wavelets != 0)*signal.square(np.pi*f1*(t+t1) ) ) * np.sin(2*np.pi*f0*(t-t0))

    # p = np.zeros(len(t))
    # for i in range(len(wavelets)):
    #     p += A[i] * (np.sin(2 * np.pi * k[i] * t) * np.exp(-(4 * np.log(2) * (t - wavelets[i])**2) / Tt**2) + A_0)
    # return p



def wavePeriod(f, wavelets, Tt):
    '''Calculate the periods of waves given frequency, number of wavelets, and total time.

    Args:
        f (float)      : Frequency of the wave.
        wavelets (int) : Number of wavelets.
        Tt (float)     : Total time.

    Returns:
        array : Array of periods that are less than or equal to total time.
    '''
    Nwavelets = 2*wavelets + 1 
    fmin      = Nwavelets/Tt
    fdelta    = f*Tt/Nwavelets#1
    f1        = f*fmin/( f + (math.ceil(fdelta)-fdelta)*fmin )

    periods   = np.sort(np.array([ abs(Tt/2 + ( (1+2*(wavelets-1))/(2*f1) ) -  i/f1) for i in range(Nwavelets-1)] ))

    return  periods[periods<=Tt]

def periodConditions(t, periods):
    '''Determine if a given time falls within any of the specified periods.

    Args:
        t (float)            : Time to check.
        periods (array) : Array of periods.

    Returns:
        bool: True if the time is within any period, False otherwise.
    '''
    p1 = periods[::2]
    p2 = np.roll(periods,-1)[::2]

    period = [(p1[i] <= t <= p2[i]) for i in range(len(p1))]

    return bool(np.sum(period))



#-------------------------------------------Constraint Class---------------------------------------------
class Constraints:
    '''Class to handle various constraints for the optimization problem.

    Args:
        constraint (str)         : Type of constraint to apply.
        dt (float)               : Time step.
        Tt (float)               : Total time.
        Nt (int)                 : Number of time steps.
        c_m (float, optional)    : Coefficient for Gaussian constraint. Default is 0.0014*1496.
        t0 (float, optional)     : Time offset for Gaussian constraint. Default is 1e-6.
        A (float, optional)      : Amplitude for Ultrasound constraint. Default is 1/1e-5.
        f (float, optional)      : Frequency for Ultrasound constraint. Default is 1.8e7.
        A_0 (float, optional)    : Initial amplitude for Ultrasound constraint. Default is 1e8.
        wavelets (int, optional) : Number of wavelets for Ultrasound constraint. Default is 2.
    '''

    def __init__(self, constraint, dt, Tt, Nt, c_m = 0.0014*1496, t0 = 1e-6, A = 1/1e-5, f = 1.8e7, A_0=1e8, wavelets=2 ):
        '''
        Initialize the Constraints class with given parameters.
        '''
        self.constraint = constraint
        self.Tt = Tt
        self.dt = dt
        self.Nt = Nt

        self.c_m = c_m
        self.t0  = t0 

        self.A = A
        self.f = f
        self.A_0 = A_0
        self.wavelets = wavelets
        self.periods = wavePeriod(f, wavelets, Tt)

        baseConcentration = 0 #0.2
        baseCurvature = 1.6e6 
            
        if self.constraint == 'None':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[1] - baseConcentration }]
        
        elif self.constraint == 'Gaussian':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[1] -  self.c_m*np.exp(-0.1*self.Nt*((t - self.Tt/2)/self.Tt)**2)}]

        elif self.constraint == 'Sigmoid':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[1] -  (1/(1+self.c_m*np.exp(-0.1*( t - self.Tt/2 )/self.dt))+0.01)}]
        
        elif self.constraint == 'Skewwed':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[1] -  (self.c_m/2)*np.exp(-0.015*self.Nt*((t-self.t0)/self.Tt)**2)*(1-special.erf(-1e8*(t-self.t0)) ) }]
            
        elif self.constraint == 'Ultrasound':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[0] - wavePulse(t, self.A, self.f, self.wavelets, self.Tt, self.A_0)+ 0.01 }]
        
        elif self.constraint == 'UltrasoundPulse':
            self.ConstraintFunction = self.PulseConstraints

        elif self.constraint == 'HodgkinHuxley':
            self.ConstraintFunction = lambda t : [{'type': 'eq', 'fun': lambda x: x[1] - baseConcentration},{'type': 'eq', 'fun': lambda x: x[0] - baseCurvature}]
        
        elif self.constraint == 'FD':
            None

        else:
            raise Exception("No Valid Constraint")


    def PulseConstraints(self, t):
        '''Define pulse constraints based on the time and internal periods.

        Args:
            t (float): Time.

        Returns:
            list: List of constraint dictionaries.
        '''

        baseConcentration = 0 #0.2

        if periodConditions(t, self.periods)==True :
            Xc = wavePulse(t, self.A, self.f, self.wavelets, self.Tt, self.A_0 )+0.01
            return  [{'type': 'eq', 'fun': lambda x: x[0] - Xc }] + [{'type': 'eq', 'fun': lambda x: x[1] - baseConcentration }]
        else: 
            return [{'type': 'eq', 'fun': lambda x: x[1] - baseConcentration  }] 
        


    def changeUltrasoundParameters(self, A = 1/1e-5, f = 1.8e7, A_0=1e6, wavelets=2):
        '''Change the parameters for the Ultrasound constraint.

        Args:
            A (float, optional)      : Amplitude for Ultrasound constraint. Default is 1/1e-5.
            f (float, optional)      : Frequency for Ultrasound constraint. Default is 1.8e7.
            A_0 (float, optional)    : Initial amplitude for Ultrasound constraint. Default is 1e6.
            wavelets (int, optional) : Number of wavelets for Ultrasound constraint. Default is 2.
        '''
        self.A = A
        self.f = f
        self.A_0 = A_0
        self.wavelets = wavelets
        self.periods = wavePeriod(f, wavelets, self.Tt)


    def changeDosingParameters(self, c_m = 5e-2, t0 = 1e-6):
        '''Change the parameters for the dosing constraint.

        Args:
            c_m (float, optional) : Coefficient for Gaussian constraint. Default is 5e-2.
            t0 (float, optional)  : Time offset for Gaussian constraint. Default is 1e-6.
        '''
        self.c_m = c_m
        self.t0  = t0 
    
    def constraintList(self, t):
        '''Get the list of constraints for a given time.

        Args:
            t (float) : Time.

        Returns:
            list : List of constraint dictionaries.
        '''
        return self.ConstraintFunction(t)

    

#==========================================================================================================
#                                       Define the Herzog model equations
#==========================================================================================================
@njit
def alpha_n(phi):
    """
    Calculate the alpha_n value for the n-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant alpha_n.
    """
    return 0.001265 * (phi + 14.273) / (1 - np.exp((phi + 14.273) / (-10)))

@njit
def beta_n(phi):
    """
    Calculate the beta_n value for the n-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant beta_n.
    """
    return 0.125 * np.exp((phi + 55) / (-2.5))


@njit
def alpha_m(phi):
    """
    Calculate the alpha_m value for the m-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant alpha_m.
    """
    return 11.49 / (1 + np.exp((phi + 8.58) / (-8.47)))

@njit
def beta_m(phi):
    """
    Calculate the beta_m value for the m-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant beta_m.
    """
    return 11.49 / (1 + np.exp((phi + 67.2) / 27.8))


@njit
def alpha_h(phi):
    """
    Calculate the alpha_h value for the h-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant alpha_h.
    """
    return 0.0658 * np.exp(-(phi + 120) / 20.33)

@njit
def beta_h(phi):
    """
    Calculate the beta_h value for the h-gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant beta_h.
    """
    return 3.0 / (1 + np.exp((phi - 6.8) / (-12.998)))


@njit
def alpha_o(phi):
    """
    Calculate the alpha_o value for a generic gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant alpha_o.
    """
    return 1.032 / (1 + np.exp((phi + 6.99) / (-14.87115)))

@njit
def beta_o(phi):
    """
    Calculate the beta_o value for a generic gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant beta_o.
    """
    return 5.79 / (1 + np.exp((phi + 130.4) / 22.9))


@njit
def alpha_p(phi):
    """
    Calculate the alpha_p value for a generic gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant alpha_p.
    """
    return 0.06435 / (1 + np.exp((phi + 73.26415) / 3.71928))

@njit
def beta_p(phi):
    """
    Calculate the beta_p value for a generic gate in the Hodgkin-Huxley model.

    Parameters:
    phi (float): The membrane potential in millivolts (mV).

    Returns:
    float: The rate constant beta_p.
    """
    return 0.13496 / (1 + np.exp((phi + 10.27853) / (-9.09334)))






def minimiser(H0, E_rest, I, Tt, Nt, arg, constraint, filename, verbose=False):
    '''Perform minimization using the Herzog model equations.

    Args:
        H0 (float)          : Initial value of the variable H.
        E_rest (float)      : Resting membrane potential.
        I (array)           : Input current array.
        Tt (float)          : Total time for the simulation.
        Nt (int)            : Number of time steps.
        arg (tuple)         : Additional arguments for the Rayleighian function.
        constraint (object) : Constraint object with methods to apply constraints during minimization.
        filename (str)      : Filename for saving output data.

    Returns:
        tuple : Arrays of X (variables), Xerr (errors), V (membrane potential), and t (time steps).
    '''

    if constraint.constraint == 'FD':
        raise Exception('Simulation requires FDsimulation function not minimisation') 
    
    t, dt = np.linspace(0,Tt,Nt), Tt/Nt

    X, Xerr= np.zeros([Nt, 3]), np.zeros(Nt)
    X[0] = np.array([H0, 0.2, E_rest])

    V = np.zeros(Nt)
    V[0] = E_rest

    m = np.zeros(Nt)
    m[0] = alpha_m(E_rest) / (alpha_m(E_rest) + beta_m(E_rest))

    n = np.zeros(Nt)
    n[0] = alpha_n(E_rest) / (alpha_n(E_rest) + beta_n(E_rest))

    h = np.zeros(Nt)
    h[0] = alpha_h(E_rest) / (alpha_h(E_rest) + beta_h(E_rest))

    o = np.zeros(Nt)
    o[0] = alpha_o(E_rest) / (alpha_o(E_rest) + beta_o(E_rest))

    p = np.zeros(Nt)
    p[0] = alpha_p(E_rest) / (alpha_p(E_rest) + beta_p(E_rest))

    #-----------------------------------------------Minimisation------------------------------------------
    for i in range(Nt-1):

        args = arg + (m[i], h[i],o[i], p[i], n[i], I[i])
        opt  = scipy.optimize.basinhopping(Rayleighian, x0=X[i], minimizer_kwargs={'args':(X[i], dt, args), 'constraints': constraint.constraintList(t[i]), 'tol': 1e-18}, target_accept_rate=0.1)
        X[i+1], Xerr[i+1]= opt['x'], opt['success']


        # # Calculate the ionic currents
        # I_Na_fast = (G_Na_fast * A_m_0 / h_m_0) * (m[i] ** 3) * h[i] * (X[i,2]*0.001 - E_Na)
        # I_Na_slow = (G_Na_slow * A_m_0 / h_m_0) * o[i] * p[i] * (X[i,2]*0.001 - E_Na)
        # I_K       = (G_K * A0 / h_m_0) * n[i] * (X[i,2]*0.001 - E_K)
        # I_Leak    = (G_Leak * A(X[i,0]*0.001)/h_m_0) * (X[i,2] - E_Leak)
        # I_ion     = (I_Na_fast + I_Na_slow + I_K + I_Leak)
        
        # # Update the membrane potential using Euler first order approximation
        # I[i]     = 0.01*Cm(X[i,0], d, epsilon)*(X[i+1,2]*0.001 - X[i,2]*0.001)/dt + I_ion 
    

        # # Update the H-H variables using Euler first order approximation
        # m[i + 1] = m[i] + dt * (alpha_m(X[i+1,2]*0.001) * (1 - m[i]) - beta_m(X[i+1,2]*0.001) * m[i])
        # n[i + 1] = n[i] + dt * (alpha_n(X[i+1,2]*0.001) * (1 - n[i]) - beta_n(X[i+1,2]*0.001) * n[i])
        # h[i + 1] = h[i] + dt * (alpha_h(X[i+1,2]*0.001) * (1 - h[i]) - beta_h(X[i+1,2]*0.001) * h[i])
        # o[i + 1] = o[i] + dt * (alpha_o(X[i+1,2]*0.001) * (1 - o[i]) - beta_o(X[i+1,2]*0.001) * o[i])
        # p[i + 1] = p[i] + dt * (alpha_p(X[i+1,2]*0.001) * (1 - p[i]) - beta_p(X[i+1,2]*0.001) * p[i])

        # A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d, m, h,o, p,  n, I = arg[-13:]

        # # Calculate the ionic currents
        # I_Na_fast = (G_Na_fast * A_m_0 / h_m_0) * (m[i] ** 3) * h[i] * (V[i] - E_Na)
        # I_Na_slow = (G_Na_slow * A_m_0 / h_m_0) * o[i] * p[i] * (V[i] - E_Na)
        # I_K = (G_K * A0 / h_m_0) * n[i] * (V[i] - E_K)
        # I_Leak = (G_Leak * A0 /h_m_0) * (V[i] - E_Leak) # X[i,2])#  A(X[i,0])
        
        # I_ion = I[i] - (I_Na_fast + I_Na_slow + I_K + I_Leak)
        # # print(m[i], n[i], h[i], o[i], p[i])
        # # print(I_ion,V[i+1])
        
        # #---Update the membrane potential using Euler first order approximation---#
        # V[i+1] = V[i] + I_ion /(cappar* dt) # 0.01 * Cm(X[i,0], d, epsilon)

        # # Update the H-H variables using Euler first order approximation
        # m[i+1] = m[i] + dt * (alpha_m(V[i + 1]) * (1 - m[i]) - beta_m(V[i + 1]) * m[i])
        # n[i+1] = n[i] + dt * (alpha_n(V[i + 1]) * (1 - n[i]) - beta_n(V[i + 1]) * n[i])
        # h[i+1] = h[i] + dt * (alpha_h(V[i + 1]) * (1 - h[i]) - beta_h(V[i + 1]) * h[i])
        # o[i+1] = o[i] + dt * (alpha_o(V[i + 1]) * (1 - o[i]) - beta_o(V[i + 1]) * o[i])
        # p[i+1] = p[i] + dt * (alpha_p(V[i + 1]) * (1 - p[i]) - beta_p(V[i + 1]) * p[i])


        if constraint.constraint == 'UltrasoundPulse' and periodConditions(t[i], constraint.periods) == False :
            constraint.changeUltrasoundParameters(A_0=X[i+1,0], A=X[i+1,0]/5)

        if verbose==True:
            print(str(i)+'-'+str(opt['success'])+'\n')

            #-------------------------------------------Export data------------------------------------------
            with open("out"+os.sep+filename+".out", 'a') as f:
                f.write(str(i)+'-'+str(opt['success'])+'\n')
            
            np.savetxt("data"+os.sep+"t-"+filename+".csv",t,delimiter=',')
            np.savetxt("data"+os.sep+"X-"+filename+".csv",X,delimiter=',')    
            np.savetxt("data"+os.sep+"I-"+filename+".csv",I,delimiter=',')
            np.savetxt("data"+os.sep+"V-"+filename+".csv",V,delimiter=',')
            np.savetxt("data"+os.sep+"XErr-"+filename+".csv",Xerr,delimiter=',')
        
    return X, Xerr, V, t



# def FDsimulation(X0, I, Tt, Nt, args, filename, verbose=False):
#     '''  

#     '''
#     K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d = args

#     c_eq = c_0*k_c*np.exp(-(mu_0a-mu_0b)/R*T)
#     g_m = 1/(2*R_m)
#     f=1

#     alpha_H = np.array([
#     [4*Pa, -4*gamma_0, (rho_0*epsilon*phi_0-d**3*Pa)/d, (gamma_0*d**3 - 16*K_b*d + 4*phi_0**2)/d, -(8+3*rho_0)/4, 4*d*(K_b*d - epsilon*phi_0), (epsilon*d**3*phi_0)/2, (3*epsilon*d**3*phi_0**2)/4, 0, 0, 0, 0],
#     [0, (4*xi*f)/a_0, (8*a_0*K_b*H_0*d + q*epsilon*phi_0)/(a_0*d), (xi*d*f)/a_0, (8*a_0*K_b*H_0*d**2 - 3*q*epsilon*d*phi_0)/(4*a_0), 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -(epsilon*d*rho_0)/2, -2*epsilon*d*phi_0, 0, -(epsilon*d**3*phi_0)/2, 0, 0, 0, 0, 0],
#     [0, 0, 0, -(epsilon*d*q)/(2*a_0), 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ],  dtype=float)


#     beta_H = np.array([
#     [0, 0, 16*phi_0, 0, -16*phi_0*d**2, 0, 3*phi_0*d**4, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -8*d**2, 0, 2*d**4, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ],  dtype=float)

#     alpha_c = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, -(2*a_0*H_0*K_b)/(R*T*np.log(c_eq)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [q/(2*R*T*np.log(c_eq)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [a_0*H_0**2/(2*R*T*np.log(c_eq)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ],  dtype=float)

#     alpha_phi = np.array([
#     [   (128*d**2*g_m*rho_0*(2*eta_s+lambda_0)+64*Pa*phi_0*epsilon**2)/epsilon**2,
#         (phi_0*(128*d*g_m*(2*eta_s+lambda_0) - 64*epsilon*gamma_0))/epsilon,
#         (16*epsilon**2*phi_0*alpha_H[0,2] - 64*epsilon**2*Pa*d**2*phi_0 - 32*d**4*g_m*rho_0*(2*eta_s+lambda_0))/epsilon**2,
#         (phi_0*(16*alpha_H[0,3] - 64*epsilon*gamma_0*d**2 -64*g_m*(2*eta_s+lambda_0)*d**3))/epsilon,
#         4*phi_0*(4*alpha_H[0,4]-4*d**2*alpha_H[0,2]-3*Pa*d**4),
#         (4*phi_0*(4*epsilon*alpha_H[0,5] - 4*epsilon*d**2*alpha_H[0,3] - 3*epsilon*gamma_0*d**4 - 2*g_m*(2*eta_s+lambda_0)*d**5))/epsilon,
#         phi_0*(16*alpha_H[0,6]-16*d**2*alpha_H[0,4]+3*alpha_H[0,2]*d**4),
#         phi_0*(16*alpha_H[0,7]-16*d**2*alpha_H[0,5]+3*alpha_H[0,3]*d**4),
#         phi_0*(-16*d**2*alpha_H[0,6]+3*alpha_H[0,4]*d**4),
#         phi_0*(-16*d**2*alpha_H[0,7]+3*alpha_H[0,5]*d**4),
#         3*alpha_H[0,6]*d**4,
#         3*alpha_H[0,7]*d**4 ],

#     [   (64*g_m*q*d**2*(2*eta_s+lambda_0))/(a_0*epsilon**2),
#         (64*phi_0*xi*f)/a_0,
#         (16*(8*a_0*epsilon*d*K_b*H_0*phi_0 + q*epsilon**2*phi_0**2 - g_m*q*d**5*(2*eta_s+lambda_0)))/(a_0*epsilon*d),
#         (16*xi*d*phi_0*(1-4*d**2)*f)/a_0,
#         (4*phi_0*(-24*K_b*H_0*a_0*d**2 - 7*epsilon*q*d*phi_0))/a_0,
#         (4*xi*d**3*phi_0*(3*d-4)*f)/a_0,
#         (-8*d**3*phi_0*(a_0*d*K_b*H_0 + epsilon*q*phi_0))/a_0,
#         (3*xi*d**5*phi_0*f)/a_0,
#         (14*xi*d**4*phi_0*f)/a_0,
#         0, 0, 0 ],

#     [   (128*g_m*(2*eta_s+lambda_0)*d)/epsilon,
#         -32*Pa*d**2,
#         (32*epsilon*d**2*gamma_0 - 64*g_m*d**3*(2*eta_s+lambda_0))/epsilon,
#         -8*(d**2*alpha_H[0,2] + epsilon*d*phi_0*rho_0 - d**4*Pa),
#         -8*(d**2*alpha_H[0,3] + gamma_0*d**4 + 4*epsilon*d*phi_0 - (g_m*(2*eta_s+lambda_0)*d**5)/epsilon),
#         -2*(4*d**3*alpha_H[0,4] - d**4*alpha_H[0,2] - 4*epsilon*d**3*rho_0*phi_0),
#         (2*d**4*alpha_H[0,3] - 8*d**2*alpha_H[0,5] + 24*epsilon*d**3*phi_0**2),
#         (2*d**4*alpha_H[0,4] - 8*d**2*alpha_H[0,6] + 24*epsilon*d**3*phi_0**2),
#         2*(3*alpha_H[0,5]*d**4-8*d**2*alpha_H[0,7] + 24*epsilon*d**3*phi_0**2),
#         2*(3*alpha_H[0,6]*d**4-8*d**2*alpha_H[0,7] + 24*epsilon*d**3*phi_0**2),
#         2*alpha_H[0,7]*d**4,
#         2*alpha_H[0,7]*d**4 ],
    
#     [0,0,0,4*epsilon*d*gamma_0*phi_0,0,2*d**3*alpha_H[0,3],2*d**3*alpha_H[0,4],0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 2*d**3*alpha_H[0,5], 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ],  dtype=float)

#     beta_phi =np.array( [
#     [128*d**2*(2*eta_s + lambda_0)/(epsilon**2*R_m), 0, 64*(4*phi_0**2*epsilon**2*R_m - d**4*(2*eta_s + lambda_0))/(epsilon**2*R_m), 0, 8*d**2*(64*phi_0**2*epsilon**2*R_m - d**4*(2*eta_s + lambda_0))/(epsilon**2*R_m), 160*d**4*phi_0**2, 0, -96*d**3*phi_0**2, 0, 9*d**8*phi_0**2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, -128*d**2*phi_0, -24*d**6*phi_0, 160*d**4*phi_0, 6*d**5*phi_0, -32*d**6*phi_0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 64*d**4, 0, -32*d**6, 0, 4*d**8, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ],  dtype=float)

#     t, dt = np.linspace(0,Tt,Nt), Tt/Nt

#     H = np.zeros(Nt)
#     tilde_c = X0[1]*np.ones(Nt)
#     phi = np.zeros(Nt)

#     H[0], tilde_c[0], phi[0] = X0

#     for i in range(Nt-1):
#         x = np.array([H[i],tilde_c[i],phi[i],tilde_c[i]*phi[i],phi[i]**2,tilde_c[i]**2,tilde_c[i]*phi[i]**2])
#         h = np.array([1,H[i],H[i]**2,H[i]**3,H[i]**4,H[i]**5,H[i]**6])
        
#         L = np.outer(x,h)

#         # Calculate dot(H)
#         # H[i+1] = H[i] + dt*( 1 / (2 * (2 * eta_s - lambda_0)*(4 - d**2 * H[i]**2)) )*( np.trace(alpha_H.T @ L) - (epsilon * R_m * I[i] / d)*np.trace(beta_H.T@ L)) 
    
#         # Calculate dot(c)
#         # tilde_c[i+1] = tilde_c[i] + dt * ( (k_on - k_c * k_off) / (2 * k_c) )*np.trace(alpha_c.T @ L)

#         # Calculate dot(phi)
#         phi[i+1] = phi[i] + dt*(2 /( (2 * eta_s - lambda_0)* (4 - d**2 * H[i]**2)**3))*( np.trace(alpha_phi.T @ L) - (epsilon * R_m * I[i] / d)*np.trace(beta_phi.T @ L))
        
#         if verbose==True:
#             print(str(i)+'-[',[H[i], tilde_c[i], phi[i]],']\n')

#             np.savetxt("data"+os.sep+"t-"+filename+".csv",t,delimiter=',')
#             np.savetxt("data"+os.sep+"X-"+filename+".csv",np.array([H, tilde_c, phi]),delimiter=',')    
#             np.savetxt("data"+os.sep+"I-"+filename+".csv",I,delimiter=',')

#     return np.array([H, tilde_c, phi]), t





def dphi_n(H, tilde_c, phi, I, args):
    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d = args
    
    g_m = 1/(2*R_m)
    chi_phi = 0.5 * (phi_0*H - 2*R_m*I) # + (rho_0*a_0 + q*tilde_c) / (2*Cm(H,d,epsilon)*a_0))
    
    term1 = I - g_m*(phi+ phi_0*H)

    term2 = (g_m/ (2*Cm(H,d,epsilon))) * (rho_0 + q*tilde_c/ a_0)
    
    term3_1 = 4 * phi_0 * Pa  - 2 * d**2 * Pa * ( phi + (4*gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi)*phi_0) / (2*d**2*Pa) ) * H

    term3_2 = ( 2 * d**2 * gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi) * phi - 4*K_b*H_0 * phi_0 * tilde_c + (4*epsilon*chi_phi*phi_0 - 3*d**2*Pa)/d*phi_0) * H**2
    term3_3 = 4*K_b*H_0*d**2 * (phi* (tilde_c- 2*epsilon*chi_phi*phi_0 / (K_b*H_0*d)) + (3*gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi)*d**2 - 16*K_b) / (4*K_b*H_0*d**2) * phi_0) * H**3
    term3_4 = (2*chi_phi * epsilon * d**3 * phi * (phi+ (4*K_b*d - epsilon*phi_0**2) / (chi_phi*epsilon*d**2)) - 6*d**2*phi_0*K_b*H_0*(tilde_c+ 2*epsilon*chi_phi*phi_0 / (K_b*H_0*d))) * H**4
    term3_5 = epsilon*d**3*phi_0 * (phi* (phi+ 6*chi_phi) + (12*K_b) / (epsilon*d)) * H**5
    term3_6 = -3*epsilon*d**3*phi_0**2 * (phi+ 3*chi_phi) * H**6
    
    term3 = term3_1 + term3_2 + term3_3 + term3_4 + term3_5 + term3_6
    
    term3 *= -epsilon / (8*d*(2*eta_s + lambda_0))
    
    dphi = -(1/Cm(H,d,epsilon)) * (term1 + term2 + term3)
    
    return dphi

def dH_n(H, phi, tilde_c, I, args):
    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d = args

    chi_phi = 0.5 * (phi_0*H - 2*R_m*I) # + (rho_0*a_0 + q*tilde_c) / (2*Cm(H,d,epsilon)*a_0))
    
    term1 = Pa - gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi) * H
    term2 = -2*K_b*H_0 * (tilde_c- epsilon * chi_phi * phi_0 / (K_b * H_0 * d)) * H**2
    term3 = -chi_phi * epsilon * d * (phi+ 4*K_b / (chi_phi * epsilon * d)) * H**3
    term4 = -0.5 * epsilon * d * phi_0 * (phi+ 3*chi_phi) * H**4
    
    dH = (1 / (2*(2*eta_s + lambda_0))) * (term1 + term2 + term3 + term4)
    return dH



def FDsimulation(X0, I, Tt, Nt, args, filename, verbose=False):
    '''  

    '''
    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_0, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d = args

    t, dt = np.linspace(0,Tt,Nt), Tt/Nt

    H = np.zeros(Nt)
    tilde_c= X0[1]*np.ones(Nt)
    phi= np.zeros(Nt)

    H[0], tilde_c[0], phi[0] = X0
    
    for i in range(Nt-1):
        # Calculate dot(H)
        H[i+1] = H[i] + dt*dH_n(H[i], phi[i], tilde_c[i], I[i], args)

        c = tilde_c[i]/(k_c*(tilde_c_m-tilde_c[i]))
        # Calculate dot(c)
        tilde_c[i+1] = 0 #tilde_c[i] + dt (tilde_k(H[i], tilde_c[i], phi[i], c, args[:-8])/2)*(mu_a(H[i], tilde_c[i], phi[i], args[:-8])-mu_b(c, c_0, mu_0b, R, T, a_0)) 

        # Calculate dot(phi)
        phi[i+1] = phi[i] + dt*dphi_n(H[i], phi[i], tilde_c[i], I[i], args)
        
        if verbose==True:            
            with open("out"+os.sep+filename+".out", 'a') as f:
                f.write(str(i)+'\n')

            print(str(i)+'-[',[H[i], tilde_c[i], phi[i]],']\n')

            np.savetxt("data"+os.sep+"t-"+filename+".csv",t,delimiter=',')
            np.savetxt("data"+os.sep+"X-"+filename+".csv",np.array([H, tilde_c, phi]),delimiter=',')    
            np.savetxt("data"+os.sep+"I-"+filename+".csv",I,delimiter=',')

    return np.array([H, tilde_c, phi]), t



def minimserPlot(X, V, I, t, k_c, filename):
    '''
    
    ''' 
    #-------------------------------------------Plot Data------------------------------------------
    fig, ax = plt.subplots(4,1, figsize = (1.61*linewidth/2, 2.2*linewidth/2))

    ax[0].plot(t[2:]*1e6, (X[2:,0]), label = r'$H$ (nm^1)') 
    ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
    ax[0].set_xlabel(r't ($\mu$s)')
    ax[0].set_ylabel('$\Delta$ H (m$^{-1}$)', rotation = 90)
    # ax[0].legend()

    ax[1].plot(t[2:]*1e6, tilde(X[2:,1], k_c), color =  ax[0]._get_lines.get_next_color(), label = r'$c$')
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
    ax[2].plot(t[2:]*1e6, X[2:,2], color =  ax[0]._get_lines.get_next_color(), label = r'$\phi$')
    ax[2].set_xlabel(r't ($\mu$s)', rotation = 0)
    ax[2].set_ylabel(r'$\phi (V)$', rotation = 90)
    ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1])
    # ax[2].legend()
    # ax[2].set_ylim(0,10000)

    ax[3].plot(t[2:]*1e6, I[2:], color =  ax[0]._get_lines.get_next_color(), label = r'$I$')
    ax[3].set_xlabel(r't ($\mu$s)', rotation = 0)
    ax[3].set_ylabel(r'$I (A)$', rotation = 90)
    ax[3].set_ylim(ax[3].get_ylim()[0], ax[3].get_ylim()[1])
    # ax[3].legend()
    # ax[3].set_ylim(0,10000)

    # ax[4].plot(t[2:2300]*1e6, V[2:], color =  ax[0]._get_lines.get_next_color(), label = r'$V$')
    # ax[4].set_xlabel(r't ($\mu$s)', rotation = 0)
    # ax[4].set_ylabel(r'$V (mV)$', rotation = 90)
    # ax[4].set_ylim(ax[4].get_ylim()[0], ax[4].get_ylim()[1])
    # ax[4].legend()
    # ax[4].set_ylim(0,10000)

    fig.tight_layout()
    fig.savefig('Figures'+ os.sep + filename+'.pdf', bbox_inches = 'tight')



def FDPlot(X, I, t, filename):
    '''
    
    ''' 
    #-------------------------------------------Plot Data------------------------------------------
    fig, ax = plt.subplots(4,1, figsize = (1.61*linewidth/2, 2.2*linewidth/2))

    ax[0].plot(t*1e6, X[0], label = r'$H$ (nm^1)') 
    ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
    ax[0].set_xlabel(r't ($\mu$s)')
    ax[0].set_ylabel('$\Delta$ H (m$^{-1}$)', rotation = 90)
    # ax[0].legend()

    ax[1].plot(t*1e6, X[1], color =  ax[0]._get_lines.get_next_color(), label = r'$c$')
    ax[1].set_xlabel(r't ($\mu$s)')
    ax[1].set_ylabel(r'$\tilde{c}$', rotation = 90)
    ax[1].set_ylim(0, ax[1].get_ylim()[1])

    ax[2].plot(t*1e6, X[2], color =  ax[0]._get_lines.get_next_color(), label = r'$\phi$')
    ax[2].set_xlabel(r't ($\mu$s)', rotation = 0)
    ax[2].set_ylabel(r'$\phi (V)$', rotation = 90)
    ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1])
    # ax[2].legend()
    # ax[2].set_ylim(0,10000)

    ax[3].plot(t*1e6, I, color =  ax[0]._get_lines.get_next_color(), label = r'$I$')
    ax[3].set_xlabel(r't ($\mu$s)', rotation = 0)
    ax[3].set_ylabel(r'$I (A)$', rotation = 90)
    ax[3].set_ylim(ax[3].get_ylim()[0], ax[3].get_ylim()[1])
    # ax[3].legend()
    # ax[3].set_ylim(0,10000)

    fig.tight_layout()
    fig.savefig('Figures'+ os.sep + filename+'.pdf', bbox_inches = 'tight')