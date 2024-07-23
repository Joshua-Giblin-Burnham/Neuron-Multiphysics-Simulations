# Neuron Multiphysics Simulations

## Introduction
Repository for code DPhil project simulating neuronal membrane, multiphysics and various FEM simulations. 


<p align="center">
   <img width="650" height="300" src="https://github.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/blob/main/docs/_figures/Nuphysim.png">
</p>


Recent experiments have underscored the active nature of neuronal membranes, revealing an intricate coupling between their mechanical, electrophysiological, and biochemical properties. To address this complexity, we develop a coupled mechano-electrophysiological and biochemical model for neuronal lipid bilayers. Extending the Onsager variational principle to neuronal membranes, we describe the non-equilibrium dynamics of infinitesimal membrane patches. Through minimising the energetic fluxes and dissipative potentials, we can derive the system's equations of motion. Applying a finite difference simulation we provide a framework for further simulational work and experimental comparison. We hope to simulate the chemical influence of anesthetics on neurons, producing modulation in curvature and voltage seen in experiments. Similarly, we hope to demonstration computationally that mechanical oscillations at ultrasound frequencies can directly influence membrane potential. The underlying mechanics may reveal profound implications for understanding brain function.

View documentation here,  .

## Modules

### PatchSim : Finite difference simulation of neuronal membrane patch

<p align="center">
   <img width="650" height="300" src="https://github.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/blob/main/docs/_figures/PatchSim.png">
</p>


As a simple application of our variational model we consider a infinitesimal patch of membrane with homogeneous properties such that all positional gradients are zero. Consequently, all integrals over $dA$ simplify to products with the area of the patch $A$ and we loss the terms associated with $d\dot{A}$. We treat the infinitesimal patch of membrane as an umbilical point, approximating the local curvature as spherical with radius $r_c = 1/H$. Therefore, the normal velocity $v_n$ becomes the radial velocity $\dot{r}_c = -\dot{H}/H^2$. Using the definition of the rate-of-deformation we can then define $\text{tr}|\boldsymbol{d}| \approx -\dot{H}/H$ and $\boldsymbol{d}:\boldsymbol{d}\approx 2\left(\dot{H}/H\right)^2$. Applying this simplification to the Rayleighian we define a Rayleighian for a finite membrane patch:  

\begin{align}
\begin{split}
    \mathfrak{R} = 
    & \Biggl\{ 2K_b (2H-H_0\tilde{c})\dot{H} - C_m\phi\phi_0\dot{H}  + \left(\frac{2\eta_{s}+\lambda}{H^2}\right)\dot{H}^2 \\    
    & - \Bigl(\phi^2+2\phi_0\phi H \Bigr)\dot{C}_m + \frac{\gamma_aH-P}{H^2} \dot{H} \\
    & + \Bigl(\frac{\mathfrak{\mu}^{\text{chem}}_{a}-\mu_{b}}{a_0} + \frac{1}{2}\frac{q\phi}{a_0}- K_b H_0 (2H-H_0\tilde{c})\Bigr)\dot{\tilde{c}} + \frac{1}{ a_0 \tilde{k}} \dot{\tilde{c}}^2\\
    & + \biggl[ -C_m (\phi+\phi_0H) + \frac{1}{2}\left( \rho_0+ \frac{q\tilde{c}}{a_0}\right)\biggr]\dot{\phi}\\
    & + R_m\cdot\Bigl[ C_m(\dot{\phi}+\phi_0\dot{H}) + \dot{C}_m(\phi+\phi_0H) \Bigr]^2\Biggr\}A  ,
\end{split}
\end{align}

where the evolution of areal capacitance is given by

\begin{align}
    \dot{C}_m  = -\frac{\varepsilon d}{2}H \dot{H} . 
\end{align}

Applying a finite difference approximation to the time derivatives of $X = \{H,\tilde{c},\phi\}$ we simulate the systems evolution by minimising at each time step. For time step n, the Rayleighian is minimised with respect to $X_{n+1}$, giving the evolution of the curvature, membrane saturation and potential across a single patch of the membrane. Model parameters are taken from the literature.

Within our Python implementation we define an array of discrete time steps in which we evolve the system. Starting from the initial state of the system we iterate over each time. At each time step, we minimise to find the evolved system configuration by minimising the Rayleighian. We apply the basenhopping algorithm, from the Scipy library, to minimise the above Rayleighian. A constraint object is used to vary constraints based on physical parameters of the simulation. For simulations looking at varying doses of anesthetic, these are imposed using constraints on $\tilde{c}$. Ultrasound simulations are produced by allowing unconstrained simulations (only $\tilde{c}=0$) between ultrasound pulses, then constraining the curvature to produce wave trains at set times.

### AxonSim: Finite element modelling of axon 






## Running Simulator
The code calculates scan variables and export them to csv files then runs ABAQUS using seperate python scripts that import the variable data. ABAQUS can be run locally, however, they are designed to be run on remote servers, using SSH to upload files and run ABAQUS on HPC queues. Cloning the git page and pip installing 'nuphysim' will add all packages/ modules to your python enviroment. All Jupyter notebooks(.ipynb) are self contained, they produce the input files, in the specified local working directory, for each simulation so best run from own self contained directory. The notebooks contain breakdown and description of code function. Seperate Python(.py) files for the AFM simulation are available in the 'Python Scripts' folder. For more lightweight code the simulator can be run from separate python kernal/notebook by importing the AFM_ABAQUS_Simulation_Code.py file (the ABAQUS scripts will need to be copied into the working directory (localPath) specified in simulator).




### Importing Python files
Within a seperate python script the simulator code can be imported by either appending the package using system command and path to directory holding the files:

    import sys
    sys.path.insert(1, 'C:\\path\\to\\directory\\nuphysim') 
    
Or by either copying the nuphysim package to the same directory or to the main python path (for jupyter notebook/spyder this will be main anaconda directory). Packages can be imported in various ways importing as:

     import nuphysim

     nuphysim.afm.AFMSimulation(...)


Alternative:

     from nuphysim import *

     afm.AFMSimulation(...)

Alternative (can have conflicting functions do not do for all as shown):

     from nuphysim.afm import *
     
     AFMSimulation(...) 

Then, the simulator can simply be run by defining the required variables and running main function:



## Common errors:
- Package files not located in working directory or system path
- Some modules may require Python 3.9 or newer. 
- You must be careful to change path syntaax if using mac or linux.
- Require the following modules: 

