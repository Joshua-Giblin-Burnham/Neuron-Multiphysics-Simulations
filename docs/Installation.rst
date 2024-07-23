Running Simulator
===================================
The code can be used to simulate the mechno-electrophysiological and biochemical coupling in neurons. 

This project implements the Herzog model for simulating neural dynamics with the incorporation of ultrasound constraints. The model is implemented in Python using 
scientific computing libraries such as NumPy, SciPy, and Numba. The code includes functions to calculate various model parameters, generate ultrasound pulses, and 
perform optimization using the Rayleighian function. The provided code offers a comprehensive framework for simulating neural dynamics using the Herzog model with the inclusion of ultrasound constraints. 
By utilizing the provided functions and classes, users can customize and run simulations to explore the impact of various parameters and constraints on the 
model's behavior


Importing Python files
===================================

Within a seperate python script the simulator code can be imported by either appending the package using system command and path to directory holding the files:

.. code-block:: python

    import sys
    sys.path.insert(1, 'C:\\path\\to\\directory\\nuphysim') 
    
Or by either copying the nuphysim package to the same directory or to the main python path (for jupyter notebook/spyder this will be main anaconda directory). Packages can be imported in various ways importing as:

.. code-block:: python

    import nuphysim

    nuphysim.patchsim.minimise(...)

Alternative:

.. code-block:: python

    from nuphysim import *

    patchsim.minimise(...)

Alternative (can have conflicting functions do not do for all as shown):

.. code-block:: python

    from nuphysim.patchsim import *
    
    minimise(...) 


Then, the simulator can simply be run by defining the required variables and running main function:

.. code-block:: python

        host, port, username, password, None, localPath, abqCommand, fileName, subData,              
        pdb, rotation, surfaceApprox, indentorType, rIndentor, theta_degrees, tip_length,             
        indentionDepth, forceRef, contrast, binSize, clearance, meshSurface, meshBase, meshIndentor,   
        timePeriod, timeInterval = ...
        
         ...minimise(...)

Example
=======

1. **Initialize Constraints**:
   Create a `Constraints` object with the desired constraint type and parameters. For example, to use ultrasound constraints:

   .. code-block:: python

      from nuphysim.patchsim import Constraints

      dt = 0.01
      Tt = 100
      Nt = 1000
      constraint = Constraints('Ultrasound', dt, Tt, Nt)

2. **Set Model Parameters**:
   Define the parameters for the Herzog model, including initial conditions (`H0`), resting potential (`E_rest`), input current (`I`), total time (`Tt`), and number of time steps (`Nt`).

   .. code-block:: python

      H0 = 1.0
      E_rest = -70
      I = np.zeros(Nt)
      Tt = 100
      Nt = 1000
      args = (gamma_0, k_c, C_D, xi, mu_0a, R, T, a_0, c_0, mu_0b, epsilon)

3. **Run Minimiser**:
   Use the `minimiser` function to run the simulation with the specified parameters and constraints. The results will be saved to a file.

   .. code-block:: python

      from nuphysim.patchsim import minimiser

      minimiser(H0, E_rest, I, Tt, Nt, args, constraint, 'output_filename')



Common Errors
===================================
 * ABAQUS scripts/ package files not located in working directory or system path
 * Some modules may require Python 3.9 or newer. 
 * You must be careful to change path syntaax if using mac or linux.
 * Require the following modules: py3Dmol, nglview, biopython, mendeleev, pyabaqus==2022, paramiko (view requirements.txt)



