.. NuPhySim documentation master file, created by
   sphinx-quickstart on Fri Oct 13 14:16:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NuPhySim Package
===================================

Authors: J. Giblin-Burnham 


Repository for DPhil project code simulating multiphysics and neuronal membrane dynamics with various FEM simulations at multiscale. 

.. image::  https://raw.githubusercontent.com/Joshua-Giblin-Burnham/Neuron-Multiphysics-Simulations/main/docs/_figures/Nuphysim.png

Recent experiments have underscored the active nature of neuronal membranes, revealing an intricate coupling between their mechanical, electrophysiological, 
and biochemical properties. To address this complexity, we develop a coupled mechano-electrophysiological and biochemical model for neuronal lipid bilayers. 
Extending the Onsager variational principle to neuronal membranes, we describe the non-equilibrium dynamics of infinitesimal membrane patches. Through minimising 
the energetic fluxes and dissipative potentials, we can derive the system's equations of motion. Applying a finite difference simulation we provide a framework for 
further simulational work and experimental comparison. We hope to simulate the chemical influence of anesthetics on neurons, producing modulation in curvature and 
voltage seen in experiments. Similarly, we hope to demonstration computationally that mechanical oscillations at ultrasound frequencies can directly influence membrane 
potential. The underlying mechanics may reveal profound implications for understanding brain function.


.. toctree::
   :maxdepth: 4
   :caption: Installation

   Installation


.. toctree::
   :maxdepth: 4
   :caption: Models

   Models/PatchSimulation


.. toctree::
   :maxdepth: 4
   :caption: Documentation

   nuphysim


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`