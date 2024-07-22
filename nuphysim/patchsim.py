#----------------------------------------------Imports-----------------------------------------------
import sys
import os
import numpy as np
from numba import njit
import math
import scipy.optimize
from scipy import special
from scipy import signal

#-------------------------------------------Active Functions---------------------------------------------
@njit
def gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi):
    return gamma_0 + xi * (tilde_c**2 )/ (tilde_c - k_c) + (C_D * phi**2)/2 

@njit
def Cm(H,d, epsilon):
    return (epsilon/d)*(1-((H*d)**2)/4)
    # return (4*np.pi*epsilon/(d*H**2))*(1-((H*d)**2)/4)

@njit
def Cm_dot(H, H_dot, d, epsilon):
    return -epsilon*d*H*H_dot/2
    # return 2*(np.pi * epsilon - Cm(H, d, epsilon) )*H_dot/ H

@njit
def mu(tilde_c, tilde_c_m, mu_0a, R, T, a_0):
    return mu_0a/a_0  +  (R*T/a_0) * ( tilde_c and np.log( tilde_c / (tilde_c_m - tilde_c)) ) 

@njit
def mu_a(H, tilde_c, phi, args):
    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q,  C_D, R, T, A = args
    return mu(tilde_c, tilde_c_m, mu_0a, R, T, a_0) + ((q*phi)/(2*a_0) )* phi - K_b * H_0 * (2 * H - H_0 * tilde_c)

@njit
def mu_b(c, c_0, mu_0b, R, T, a_0):
    return mu_0b/a_0 + R*T * ( c and np.log(c /c_0) )

@njit
def tilde_k(H, tilde_c, phi, c, args):
    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q,  C_D, R, T, A = args
    return k_off*(  (c/k_c)*(tilde_c_m-tilde_c) - np.exp(-(mu_0a-mu_0b)/R*T)   )/(a_0* (mu_a(H, tilde_c, phi, args) - mu_b(c, c_0, mu_0b, R, T, a_0)))

@njit
def tilde(c,k_c):
    return c*k_c/(1+c*k_c)


# #----------------------------------------Rayleighian Functions-------------------------------------------
# @njit
# def Rayleighian(X, X_n, dt, args):

#     K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, q, R, T, A = args
    
#     H, c, phi = X
#     H_n, c_n, phi_n = X_n
#     tilde_c, tilde_c_n = tilde(c,k_c), tilde(c_n,k_c)

#     H_dot, tilde_c_dot, phi_dot = H-H_n/dt, tilde_c-tilde_c_n/dt, phi-phi_n/dt

#     C_m =  Cm(H, d, epsilon)
#     C_m_dot = Cm_dot(H, H_dot, d, epsilon)
    
#     R = ( 
#         2 * K_b * (2 * H - H_0 * tilde_c) * H_dot 
#         - C_m * phi * phi_0 * H_dot  
#         + (2 * eta_s + lambda_0)*float(H and H_dot/H)**2 
#         - (phi**2 + 2 * phi_0 * phi * H) * C_m_dot 
#         + (gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi) * H - Pa) * H_dot * float(H and 1/H)**2
        
#         + 
#         ( 0 
#         + mu_a(H, tilde_c, phi, args)
#         + mu_b(c, c_0, mu_0b, R, T, a_0)
#         ) *tilde_c_dot 


#          +  tilde_c_dot**2 /(tilde_k(H, tilde_c, phi, c, args)* a_0) 

#         + 
#         ( 0 
#         - C_m * (phi + phi_0*H) 
#         + 0.5 * (rho_m + (q * tilde_c)/a_0)
#         ) * phi_dot 

#         + R_m * (C_m*(phi_dot+phi_0*H_dot) - C_m_dot*(phi+phi_0*H))**2 

#         ) * A

#     return R



# def Rayleighian(X, X_n, dt, args):

#     K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, q, R, T, d, A = args
    
#     H, c, phi = X
#     H_n, c_n, phi_n = X_n
#     tilde_c, tilde_c_n = tilde(c,k_c), tilde(c_n,k_c)

#     H_dot, tilde_c_dot, phi_dot = H-H_n/dt, tilde_c-tilde_c_n/dt, phi-phi_n/dt

#     C_m =  Cm(H, d, epsilon)
#     C_m_dot = Cm_dot(H, H_dot, d, epsilon)
    
#     R = ( 
#         2 * K_b * (2 * H - H_0 * tilde_c) * H_dot 
#         - C_m * phi * phi_0 * H_dot  
#         + (2 * eta_s + lambda_0)*float(H and H_dot/H)**2 
#         - (phi**2 + 2 * phi_0 * phi * H) * C_m_dot 
#         + (gamma_a(tilde_c, phi, gamma_0, k_c, C_D, xi) * H - Pa) * H_dot * float(H and 1/H)**2
        
#         + 
#         ( 0 
#         + mu_a(H, tilde_c, phi, args)
#         + mu_b(c, c_0, mu_0b, R, T, a_0)
#         ) *tilde_c_dot 


#          +  tilde_c_dot**2 /(tilde_k(H, tilde_c, phi, c, args)* a_0) 

#         + 
#         ( 0 
#         - C_m * (phi + phi_0*H) 
#         + 0.5 * (rho_m + (q * tilde_c)/a_0)
#         ) * phi_dot 

#         + R_m * (C_m*(phi_dot+phi_0*H_dot) - C_m_dot*(phi+phi_0*H))**2 

#         ) * A 

#     return R




#----------------------------------------Rayleighian Functions-------------------------------------------
# @njit
def Rayleighian(X, X_n, dt, args):

    K_b, H_0, phi_0, eta_s, lambda_0, epsilon, gamma_0, xi, Pa, R_m, tilde_c_m, mu_0a, mu_0b, k_on, k_off, a_0, k_c, c_0, rho_m, q, C_D, R, T, A0, G_Na_fast, G_Na_slow, G_K, G_Leak, E_Na, E_K, E_Leak, d, m, h,o, p,  n, I = args
    
    H, c, phi = X
    H_n, c_n, phi_n = X_n
    tilde_c, tilde_c_n = tilde(c,k_c), tilde(c_n,k_c)

    H_dot, tilde_c_dot, phi_dot = H-H_n/dt, tilde_c-tilde_c_n/dt, phi-phi_n/dt

    C_m =  Cm(H, d, epsilon)
    C_m_dot = Cm_dot(H, H_dot, d, epsilon)

    # Calculate the ionic currents
    I_Na_fast = (G_Na_fast * A0 /d)  * (m ** 3) * h * (phi - E_Na)**2
    I_Na_slow = (G_Na_slow * A0 /d)  * o * p *        (phi - E_Na)**2
    I_K       = (G_K * A0 / d)   * n *            (phi - E_K)**2
    I_Leak    = (G_Leak * A0 /d) *                (phi - E_Leak)**2
    
    I_ion = I*phi - (I_Na_fast + I_Na_slow + I_K + I_Leak)
    
    
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


         +  tilde_c_dot**2 /(tilde_k(H, tilde_c, phi, c, args[:-16])* a_0) 

        + 
        ( 0 
        - C_m * (phi + phi_0*H) 
        + 0.5 * (rho_m + (q * tilde_c)/a_0)
        ) * phi_dot 

        + I_ion

        + R_m * (C_m*(phi_dot+phi_0*H_dot) - C_m_dot*(phi+phi_0*H))**2 

        ) * A0

    return R


#-------------------------------------------Ultrasound Functions-------------------------------------------
def wavePulse(t, A, k, wavelets, Tt, A_0):
    '''  '''
    Nwavelets = 2*wavelets + 1

    f0     = k
    fmin   = Nwavelets/Tt
    fdelta = f0*Tt/Nwavelets#1
    f1     = f0*fmin/( f0 + (math.ceil(fdelta)-fdelta)*fmin )

    t0 = 1/2*(Tt-1/f1)
    t1 = - Tt/2  - 3/(2*f1) + 3/f1*(wavelets)

    return A_0 + A/2 * ( 1-int(wavelets != 0)*signal.square(np.pi*f1*(t+t1) ) ) * np.sin(2*np.pi*f0*(t-t0))

def wavePeriod(f, wavelets, Tt):
    ''' '''
    Nwavelets = 2*wavelets + 1 
    fmin      = Nwavelets/Tt
    fdelta    = f*Tt/Nwavelets#1
    f1        = f*fmin/( f + (math.ceil(fdelta)-fdelta)*fmin )

    periods   = np.sort(np.array([ abs(Tt/2 + ( (1+2*(wavelets-1))/(2*f1) ) -  i/f1) for i in range(Nwavelets-1)] ))

    return  periods[periods<=Tt]

def periodConditions(t, periods):
    p1 = periods[::2]
    p2 = np.roll(periods,-1)[::2]

    period = [(p1[i] <= t <= p2[i]) for i in range(len(p1))]

    return bool(np.sum(period))

def PulseConstraints(self, t):
    ''' '''
    if periodConditions(t, self.periods)==True :
        Xc = wavePulse(t, self.A, self.f, self.wavelets, self.Tt, self.A_0 )
        return  [{'type': 'eq', 'fun': lambda x: x[0] - Xc }] + [{'type': 'eq', 'fun': lambda x: x[1] }]
    else: 
        return [{'type': 'eq', 'fun': lambda x: x[1] }]
    

#-------------------------------------------Constraint Class---------------------------------------------
class Constraints:
    '''  '''
    def __init__(self, constraint, dt, Tt, Nt, c_m = 0.0014*1496, t0 = 1e-6, A = 1/1e-5, f = 1.8e7, A_0=1e8, wavelets=2 ):
        ''' '''
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
            
        if self.constraint == 'None':
            self.ConstraintFunction = lambda self, t : [{'type': 'eq', 'fun': lambda x: x[1]-0.2 }]

        elif self.constraint == 'Gaussian':
            self.ConstraintFunction = lambda self, t : [{'type': 'eq', 'fun': lambda x: x[1] -  self.c_m*np.exp(-0.1*self.Nt*((t - self.Tt/2)/self.Tt)**2)}]

        elif self.constraint == 'Sigmoid':
            self.ConstraintFunction = lambda self, t : [{'type': 'eq', 'fun': lambda x: x[1] -  1/(1+self.c_m*np.exp(-0.1*( t - self.Tt/2 )/self.dt))}]
        
        elif self.constraint == 'Skewwed':
            self.ConstraintFunction = lambda self, t : [{'type': 'eq', 'fun': lambda x: x[1] -  (self.c_m/2)*np.exp(-0.015*self.Nt*((t-self.t0)/self.Tt)**2)*(1-special.erf(-1e8*(t-self.t0)) ) }]
            
        elif self.constraint == 'Ultrasound':
            self.ConstraintFunction = lambda self, t : [{'type': 'eq', 'fun': lambda x: x[0] - wavePulse(t, self.A, self.f, self.wavelets, self.Tt, self.A_0 ) }]
        
        elif self.constraint == 'UltrasoundPulse':
            self.ConstraintFunction = PulseConstraints

        else:
            raise Exception("No Valid Constraint")

    def changeUltrasoundParameters(self, A = 1/1e-5, f = 1.8e7, A_0=1e6, wavelets=2):
        ''' '''
        self.A = A
        self.f = f
        self.A_0 = A_0
        self.wavelets = wavelets
        self.periods = wavePeriod(f, wavelets, self.Tt)


    def changeDosingParameters(self, c_m = 5e-2, t0 = 1e-6):
        ''' '''
        self.c_m = c_m
        self.t0  = t0 
    
    def constraintList(self, t):
        ''' '''
        return self.ConstraintFunction(self, t)

#==========================================================================================================
#                                       Define the Herzog model equations
#==========================================================================================================
alpha_n = lambda x: 0.001265 * (x + 14.273) / (1 - np.exp((x + 14.273) / (-10)))
beta_n = lambda x: 0.125 * np.exp((x + 55) / (-2.5))

alpha_m = lambda x: 11.49 / (1 + np.exp((x + 8.58) / (-8.47)))
beta_m = lambda x: 11.49 / (1 + np.exp((x + 67.2) / 27.8))

alpha_h = lambda x: 0.0658 * np.exp(-(x + 120) / 20.33)
beta_h = lambda x: 3.0 / (1 + np.exp((x - 6.8) / (-12.998)))

alpha_o = lambda x: 1.032 / (1 + np.exp((x + 6.99) / (-14.87115)))
beta_o = lambda x: 5.79 / (1 + np.exp((x + 130.4) / 22.9))

alpha_p = lambda x: 0.06435 / (1 + np.exp((x + 73.26415) / 3.71928))
beta_p = lambda x: 0.13496 / (1 + np.exp((x + 10.27853) / (-9.09334)))


def minimiser(H0, E_rest, I, Tt, Nt, arg, constraint, filename):
    '''   
    
    '''
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

        opt = scipy.optimize.basinhopping(Rayleighian, x0=X[i], minimizer_kwargs={'args':(X[i], dt, args), 'constraints': constraint.constraintList(t[i]), 'tol': 1e-18}, target_accept_rate=0.1)
        X[i+1], Xerr[i+1]= opt['x'], opt['success']


        # # Calculate the ionic currents
        # I_Na_fast = (G_Na_fast * A_m_0 / h_m_0) * (m[i] ** 3) * h[i] * (X[i,2]*0.001 - E_Na)
        # I_Na_slow = (G_Na_slow * A_m_0 / h_m_0) * o[i] * p[i] * (X[i,2]*0.001 - E_Na)
        # I_K       = (G_K * A0 / h_m_0) * n[i] * (X[i,2]*0.001 - E_K)
        # I_Leak    = (G_Leak * A(X[i,0]*0.001)/h_m_0) * (X[i,2] - E_Leak)
        # I_ion     = (I_Na_fast + I_Na_slow + I_K + I_Leak)
        
        # # Update the membrane potential using Euler first order approximation
        # I[i]     = 0.01*Cm(X[i,0], d, epsilon)*(X[i+1,2]*0.001 - X[i,2]*0.001)/dt + I_ion 
    

        # Update the H-H variables using Euler first order approximation
        m[i + 1] = m[i] + dt * (alpha_m(X[i+1,2]*0.001) * (1 - m[i]) - beta_m(X[i+1,2]*0.001) * m[i])
        n[i + 1] = n[i] + dt * (alpha_n(X[i+1,2]*0.001) * (1 - n[i]) - beta_n(X[i+1,2]*0.001) * n[i])
        h[i + 1] = h[i] + dt * (alpha_h(X[i+1,2]*0.001) * (1 - h[i]) - beta_h(X[i+1,2]*0.001) * h[i])
        o[i + 1] = o[i] + dt * (alpha_o(X[i+1,2]*0.001) * (1 - o[i]) - beta_o(X[i+1,2]*0.001) * o[i])
        p[i + 1] = p[i] + dt * (alpha_p(X[i+1,2]*0.001) * (1 - p[i]) - beta_p(X[i+1,2]*0.001) * p[i])

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


        # print(str(i)+'-'+str(opt['success'])+'\n')

        # #-------------------------------------------Export data------------------------------------------
        # with open('Simulations'+os.sep+filename+".out", 'a') as f:
        #     f.write(str(i)+'-'+str(opt['success'])+'\n')
        
        # np.savetxt("data"+os.sep+"t-"+filename+".csv",t,delimiter=',')
        # np.savetxt("data"+os.sep+"X-"+filename+".csv",X,delimiter=',')    
        # np.savetxt("data"+os.sep+"I-"+filename+".csv",I,delimiter=',')
        # np.savetxt("data"+os.sep+"V-"+filename+".csv",V,delimiter=',')
        # np.savetxt("data"+os.sep+"XErr-"+filename+".csv",Xerr,delimiter=',')

    return X, Xerr, V, t





