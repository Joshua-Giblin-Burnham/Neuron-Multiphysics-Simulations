<p align="center">
   <img width="550" height="350" src="https://github.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/blob/main/docs/_static/NuPhySim_text_logo.svg">
</p>


# Neuron Multiphysics Simulations

## Introduction
Repository for code DPhil project simulating neuronal membrane, multiphysics and various FEM simulations. 


<p align="center">
   <img width="900" height="700" src="https://github.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/blob/main/docs/_figures/Nuphysim.png">
</p>


Recent experiments have underscored the active nature of neuronal membranes, revealing an intricate coupling between their mechanical, electrophysiological, and biochemical properties. To address this complexity, we develop a coupled mechano-electrophysiological and biochemical model for neuronal lipid bilayers. Extending the Onsager variational principle to neuronal membranes, we describe the non-equilibrium dynamics of infinitesimal membrane patches. Through minimising the energetic fluxes and dissipative potentials, we can derive the system's equations of motion. Applying a finite difference simulation we provide a framework for further simulational work and experimental comparison. We hope to simulate the chemical influence of anesthetics on neurons, producing modulation in curvature and voltage seen in experiments. Similarly, we hope to demonstration computationally that mechanical oscillations at ultrasound frequencies can directly influence membrane potential. The underlying mechanics may reveal profound implications for understanding brain function.

View documentation here,  https://neuron-multiphysics-simulations.readthedocs.io/en/latest/index.html#.

## Modules

### PatchSim : Finite difference simulation of neuronal membrane patch

<p align="center">
   <img width="450" height="575" src="https://github.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/blob/main/docs/_figures/PatchSim.png">
</p>


As a simple application of our variational model we consider a infinitesimal patch of membrane with homogeneous properties such that all positional gradients are zero. Consequently, all integrals over $dA$ simplify to products with the area of the patch $A$ and we loss the terms associated with $d\dot{A}$. We treat the infinitesimal patch of membrane as an umbilical point, approximating the local curvature as spherical with radius $r_c = 1/H$. Therefore, the normal velocity $v_n$ becomes the radial velocity $\dot{r}_c = -\dot{H}/H^2$. Using the definition of the rate-of-deformation we can then define $\text{tr}|\boldsymbol{d}| \approx -\dot{H}/H$ and $\boldsymbol{d}:\boldsymbol{d}\approx 2(\dot{H}/H)^2$. Applying this simplification to the Rayleighian we define a Rayleighian for a finite membrane patch:  


$$ R = \Biggl( 2K_b (2H-H_0\tilde{c})\dot{H} - C_m \phi \phi_{0} \dot{H}  + \left( \frac{2\eta_{s} + \lambda }{H^2} \right)\dot{H}^2 - \Bigl(\phi^2 + 2 \phi_{0} \phi H \Bigr)\dot{C_m} + \frac{ gamma_{a} H - P}{H^2} \dot{H} + \Bigl(\frac{ \mu_{a}^{\text{chem}} - \mu_{b} }{a_0} + \frac{1}{2}\frac{q \phi}{a_{0}}- K_{b} H_{0} (2H - H_{0}\tilde{c}) \Bigr) \dot{\tilde{c}} + \frac{1}{ a_{0} \tilde{k}} \dot{\tilde{c}}^2 + \biggl( - C_{m} ( \phi + \phi_{0} H) + \frac{1}{2} \left( \rho_{0} + \frac{q \tilde{c}}{a_{0}} \right) \biggr)\dot{\phi} + R_{m} \cdot \Bigl( C_{m} (\dot{\phi} + \phi_{0} \dot{H}) + \dot{C}_{m} (\phi + \phi_{0} H) \Bigr)^2 \Biggr) A  $$

where the evolution of areal capacitance is given by

$$\dot{C}_m  = -\frac{\varepsilon d}{2}H \dot{H} . $$

Applying a finite difference approximation to the time derivatives of $X = \{H,\tilde{c},\phi\}$ we simulate the systems evolution by minimising at each time step. For time step n, the Rayleighian is minimised with respect to $X_{n+1}$, giving the evolution of the curvature, membrane saturation and potential across a single patch of the membrane. Model parameters are taken from the literature.

Within our Python implementation we define an array of discrete time steps in which we evolve the system. Starting from the initial state of the system we iterate over each time. At each time step, we minimise to find the evolved system configuration by minimising the Rayleighian. We apply the basenhopping algorithm, from the Scipy library, to minimise the above Rayleighian. A constraint object is used to vary constraints based on physical parameters of the simulation. For simulations looking at varying doses of anesthetic, these are imposed using constraints on $\tilde{c}$. Ultrasound simulations are produced by allowing unconstrained simulations (only $\tilde{c}=0$) between ultrasound pulses, then constraining the curvature to produce wave trains at set times.

### AxonSim: Finite element modelling of axon 






## Running Simulator
The code can be used to simulate the mechno-electrophysiological and biochemical coupling in neurons. 

This project implements the Herzog model for simulating neural dynamics with the incorporation of ultrasound constraints. The model is implemented in Python using 
scientific computing libraries such as NumPy, SciPy, and Numba. The code includes functions to calculate various model parameters, generate ultrasound pulses, and 
perform optimization using the Rayleighian function. The provided code offers a comprehensive framework for simulating neural dynamics using the Herzog model with the inclusion of ultrasound constraints. 
By utilizing the provided functions and classes, users can customize and run simulations to explore the impact of various parameters and constraints on the 
model's behavior




### Importing Python files
Within a seperate python script the simulator code can be imported by either appending the package using system command and path to directory holding the files:

    import sys
    sys.path.insert(1, 'C:\\path\\to\\directory\\nuphysim') 
    
Or by either copying the nuphysim package to the same directory or to the main python path (for jupyter notebook/spyder this will be main anaconda directory). Packages can be imported in various ways importing as:

    import nuphysim

    nuphysim.patchsim.minimiser(H0, E_rest, I, Tt, Nt, args, constraint, 'output_filename')


Alternative:

    from nuphysim import *

    patchsim.minimiser(H0, E_rest, I, Tt, Nt, args, constraint, 'output_filename')

Alternative (can have conflicting functions do not do for all as shown):

    from nuphysim.afm import *

    minimiser(H0, E_rest, I, Tt, Nt, args, constraint, 'output_filename')

Then, the simulator can simply be run by defining the required variables and running main function:


### Example

1. **Initialize Constraints**:
   Create a `Constraints` object with the desired constraint type and parameters. For example, to use ultrasound constraints:

        from nuphysim.patchsim import Constraints

        dt = 0.01
        Tt = 100
        Nt = 1000
        constraint = Constraints('Ultrasound', dt, Tt, Nt)

2. **Set Model Parameters**:
   Define the parameters for the Herzog model, including initial conditions (`H0`), resting potential (`E_rest`), input current (`I`), total time (`Tt`), and number of time steps (`Nt`).

        H0 = 1.0
        E_rest = -70
        I = np.zeros(Nt)
        Tt = 100
        Nt = 1000
        args = (gamma_0, k_c, C_D, xi, mu_0a, R, T, a_0, c_0, mu_0b, epsilon)

3. **Run Minimiser**:
   Use the `minimiser` function to run the simulation with the specified parameters and constraints. The results will be saved to a file.

        from nuphysim.patchsim import minimiser

        minimiser(H0, E_rest, I, Tt, Nt, args, constraint, 'output_filename')


## Common errors:
- Package files not located in working directory or system path
- Some modules may require Python 3.9 or newer. 
- You must be careful to change path syntaax if using mac or linux.
- Require the following modules: 

