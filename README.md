# brown_circadian_dual-feedback_loop

This repository contains the code for the Brown and Doyle dual-feedback loop model of the molecular circadian oscillator and the initial model predictive control simulations using the model for multi-input control.

The model is written in Python 2.7 and integrated using CasADi.

A description of each of the main files is given below:
* model.py: the set of differential equations and parameters for the model, including a sample of how the phase and amplitude response curves are calculated
* control.py: the code for the model predictive control simulations comparing the phase trajectory and control input profiles for different combinations of known small molecules
* controlmodel.py: the model reparameterized so that the degradation rate of nuclear CRY is expressed as a single parameter for CRY1 and CRY2 to capture the action of KL001 in a single parameter
* GA.py: the cost function and genetic algorithm implemented in deap used to fit the parameters of the model

This repository also includes locally used imports with basic tools for circadian analysis and plotting (LimitCycle.py, PlotOptions.py, Utilities.py) which includes code that was previously developed by other authors for other papers cited in the accompanying manuscript (currently under review):

Abel JH, Chakrabarty A, Klerman EB, Doyle III FJ. Pharmaceutical-based entrainment of circadian phase via nonlinear model predictive control. Automatica. 2019;100:336-348.

Hirota T, Lee JW, St John PC, Sawa M, Iwaisako K, Noguchi T, et al. Identification of small molecule activators of cryptochrome. Science. 2012; 337(6098):1094-1097.

