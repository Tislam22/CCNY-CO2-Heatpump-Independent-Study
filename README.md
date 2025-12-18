# CCNY-CO2-Heatpump-Independent-Study
Python Code for the Black Box model of Heat Pump as well as the code for calculating efficiencies of various gas cooler models. 

The script titled CO2 black box model is a hybrid black-box system model that predicts heating/cooling capacity, COP, compressor power, and key state points as a function of outdoor temperature. User inputs various system parameters when prompted and the code calculates the resulting parameters and generates graph of predicted cop values from 0-100F. The script also plots the experimental data onto the same COP curve to help vizualise the difference. 

The script titled gas cooler calculations calculates the efficiencies of various gas cooler models. The script will prompt for user input of various gas cooler parameters, and will then provide the gas cooler efficiency once given the inputs. 

Both scripts require installations of CoolProp, numpy, pandas, matplotlib, and xlsxwriter.
