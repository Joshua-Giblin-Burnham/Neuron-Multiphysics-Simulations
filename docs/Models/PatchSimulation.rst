Membrane Patch Simulation 
============================



Finite difference simulation of neuronal membrane patch
-------------------------------------------------------

.. image::  https://raw.githubusercontent.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/main/docs/_figures/PatchSim_graphic.png

As a simple application of our variational model we consider a infinitesimal patch of membrane with homogeneous properties such that all positional gradients are zero. 
Consequently, all integrals over :math:`dA` simplify to products with the area of the patch :math:`A` and we loss the terms associated with :math:`d\dot{A}`. We treat the infinitesimal 
patch of membrane as an umbilical point, approximating the local curvature as spherical with radius :math:`r_c = 1/H`. Therefore, the normal velocity `v_n` becomes the radial 
velocity :math:`\dot{r}_c = -\dot{H}/H^2`. Using the definition of the rate-of-deformation we can then define :math:`\text{tr}|\boldsymbol{d}| \approx -\dot{H}/H` and :math:`\boldsymbol{d}:\boldsymbol{d}\approx 2\left(\dot{H}/H\right)^2`. 
Applying this simplification to the Rayleighian we define a Rayleighian for a finite membrane patch:  

.. math:: \mathfrak{R} = \Biggl\{ 2K_b (2H-H_0\tilde{c})\dot{H} - C_m\phi\phi_0\dot{H}  + \left(\frac{2\eta_{s}+\lambda}{H^2}\right)\dot{H}^2   
    - \Bigl(\phi^2+2\phi_0\phi H \Bigr)\dot{C}_m + \frac{\gamma_aH-P}{H^2} \dot{H} 
    + \Bigl(\frac{\mathfrak{\mu}^{\text{chem}}_{a}-\mu_{b}}{a_0} + \frac{1}{2}\frac{q\phi}{a_0}- K_b H_0 (2H-H_0\tilde{c})\Bigr)\dot{\tilde{c}} + \frac{1}{ a_0 \tilde{k}} \dot{\tilde{c}}^2
    + \biggl[ -C_m (\phi+\phi_0H) + \frac{1}{2}\left( \rho_0+ \frac{q\tilde{c}}{a_0}\right)\biggr]\dot{\phi}
    + R_m\cdot\Bigl[ C_m(\dot{\phi}+\phi_0\dot{H}) + \dot{C}_m(\phi+\phi_0H) \Bigr]^2\Biggr\}A  ,


where the evolution of areal capacitance is given by

.. math:: \dot{C}_m  = -\frac{\varepsilon d}{2}H \dot{H} . 

Applying a finite difference approximation to the time derivatives of :math:`X = \{H,\tilde{c},\phi\}` we simulate the systems evolution by minimising at each time step. 
For time step n, the Rayleighian is minimised with respect to :math:`X_{n+1}`, giving the evolution of the curvature, membrane saturation and potential across a single patch 
of the membrane. Model parameters are taken from the literature.

Within our Python implementation we define an array of discrete time steps in which we evolve the system. Starting from the initial state of the system we iterate over 
each time. At each time step, we minimise to find the evolved system configuration by minimising the Rayleighian. We apply the basenhopping algorithm, from the Scipy 
library, to minimise the above Rayleighian. A constraint object is used to vary constraints based on physical parameters of the simulation. For simulations looking at 
varying doses of anesthetic, these are imposed using constraints on :math:`\tilde{c}`. Ultrasound simulations are produced by allowing unconstrained simulations 
(only :math:`\tilde{c}=0`) between ultrasound pulses, then constraining the curvature to produce wave trains at set times.



Code Overview
--------

This project implements the Herzog model for simulating neural dynamics with the incorporation of ultrasound constraints. The model is implemented in Python using 
scientific computing libraries such as NumPy, SciPy, and Numba. The code includes functions to calculate various model parameters, generate ultrasound pulses, and 
perform optimization using the Rayleighian function. The provided code offers a comprehensive framework for simulating neural dynamics using the Herzog model with the inclusion of ultrasound constraints. 
By utilizing the provided functions and classes, users can customize and run simulations to explore the impact of various parameters and constraints on the 
model's behavior

The code is organized into several key components:

1. **Gamma Functions**: These functions compute various parameters such as ``gamma_a``, ``Cm``, ``Cm_dot``, ``mu``, ``mu_a``, ``mu_b``, and ``tilde_k`` based on the model equations.

2. **Rayleighian Function**: This function calculates the Rayleighian, which is used in the optimization process.

3. **Ultrasound Functions**: These functions generate ultrasound wave pulses (``wavePulse``), calculate wave periods (``wavePeriod``), and determine period conditions (``periodConditions``).

4. **Constraints Class**: The ``Constraints`` class manages different types of constraints (e.g., ultrasound) and includes methods to change parameters and generate constraint lists.

5. **Herzog Model Equations**: These functions compute the alpha and beta parameters for various state variables (e.g., ``alpha_n``, ``beta_n``, ``alpha_m``, ``beta_m``).

6. **Minimiser Function**: The ``minimiser`` function runs the optimization process for the Herzog model given the initial conditions, parameters, and constraints.

