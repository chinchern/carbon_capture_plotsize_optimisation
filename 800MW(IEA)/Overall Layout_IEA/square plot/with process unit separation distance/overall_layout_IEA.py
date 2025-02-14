
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:22:26 2021

@author: chin chern
"""

# --------------Set-up--------------
# Import functions
from pulp import *
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.stats import norm
import math
import csv
import time

# Create the model object
layout = pulp.LpProblem("Layout_Problem_Model",LpMinimize)

# --------------Errors--------------
class NoFEIError(Exception):
    """Exception raised when protection devices enabled without FEI constraints"""
pass

# --------------Switches--------------
# CPLEX free edition only supports up to 1000 variables. For larger land sizes, use CBC or increase coarseness (g)
# 1 = CPLEX, 2 = CBC
solver = 1

# Toggle constraints in layout problem (1 is on; 0 is off)
# Land Use Constraints
SwitchLandUse = 1
# Square Plot (else: Rectangular Plot)
SwitchSquarePlot = 1
# FEI Constraints
SwitchFEI = 1
# Toggle protection devices (must have FEI enabled if 1)
SwitchProt = 0
# FEI Cost of Life Constraints (only works if SwitchFEI is on)
SwitchFEIVle = 1
# CEI Constraints
SwitchCEI = 1
# Pumping Cost Calculation (only works if SwitchConnection Cost is on)
SwitchPumpingCost =1
# Safety Distances
SwitchSafetyDistances = 1


# Check for errors if SwitchProt == 1 and SwitchFEI == 0:
#    raise NoFEIError("FEI cost of life constraints not allowed without FEI constraints")        
if SwitchProt == 1 and SwitchFEI == 0:
    raise NoFEIError("Protection devices not allowed without FEI constraints") 

# --------------Define Sets--------------
# Define the process units
#Iteration 1
# units = ['direct_contact_cooler', 'absorber','lean_amine_cooler','cross_heat_exchanger','stripper','storage_tank']

#Iteration 2
# units = ['direct_contact_cooler', 'absorber','lean_amine_cooler','cross_heat_exchanger','stripper','storage_tank','reclaimer'
#           ,'compressor_house','compressor_KOdrum','compressor_interstage_cooler','TEG_dehydration']
#Iteration 3
units = ['direct_contact_cooler', 'absorber','lean_amine_cooler','cross_heat_exchanger','stripper','storage_tank','reclaimer'
          ,'compressor_house','compressor_KOdrum','compressor_interstage_cooler','TEG_dehydration'
          ,'cooling_tower','coolingwater_pumphouse']

pertinent_units_FEI = ['absorber', 'stripper','storage_tank']
pertinent_units_Vlc = ['absorber', 'stripper','storage_tank']
hazardous_chemicals = ['MEA']
Nunits = len(units)

if SwitchSafetyDistances == 1:
    # type of process unit    
    cooling_tower_process_unit= ['cooling_tower']
    moderate_hazard_process_unit =['direct_contact_cooler','absorber','cross_heat_exchanger','stripper','lean_amine_cooler','compressor_KOdrum','compressor_interstage_cooler','reclaimer','TEG_dehydration']
    storage_tank_process_unit =['storage_tank']
    compressor_process_unit=['compressor_house']
    large_pump_house=['coolingwater_pumphouse']
    
# --------------Define Parameters and Values--------------
# Base layout model

# M (m) is used in the contraints which set A or B, L or R, and Din or Dout.
# It should be big enough not to constrain the size of the plot, but not too big
M = 1e3

# Dimensions of each unit (m)
alpha = dict.fromkeys(units)
beta = dict.fromkeys(units)
alpha['direct_contact_cooler'] = 66
alpha['absorber'] =46
alpha['lean_amine_cooler'] = 30
alpha['cross_heat_exchanger'] =16
alpha['stripper'] =60
alpha['storage_tank'] = 34
alpha['reclaimer'] = 18

beta['direct_contact_cooler'] = 66
beta['absorber'] =46
beta['lean_amine_cooler'] = 30
beta['cross_heat_exchanger'] =16
beta['stripper'] =60
beta['storage_tank'] = 34
beta['reclaimer'] = 18

alpha['compressor_house'] = 37.5
alpha['compressor_KOdrum'] =26
alpha['compressor_interstage_cooler'] = 22
alpha['TEG_dehydration'] = 14

beta['compressor_house'] = 8
beta['compressor_KOdrum'] =26
beta['compressor_interstage_cooler'] = 22
beta['TEG_dehydration'] = 14

alpha['cooling_tower'] = 52
beta['cooling_tower'] = 52
alpha['coolingwater_pumphouse'] = 17
beta['coolingwater_pumphouse'] = 8

#Annualised Factor
annualised_factor = 0.1102
#Cost of Electricity in 2021
electricity_cost = 4.5*1e-5  #GBP/Wh
#Operating hours
operating_hour = 8000   #hrs
    
if SwitchPumpingCost == 1:
    pump_efficiency = 0.6

# Installed cost of each unit (GBP)
Cp = dict.fromkeys(units)
Cp['direct_contact_cooler'] = 3299179
Cp['absorber'] =64274466
Cp['lean_amine_cooler'] = 11781343
Cp['cross_heat_exchanger'] =1877261
Cp['stripper'] =25735317
Cp['storage_tank'] = 8732425
Cp['reclaimer'] = 2432000

Cp['compressor_house'] = 27122430
Cp['compressor_KOdrum'] =5022332
Cp['compressor_interstage_cooler'] = 981428
Cp['TEG_dehydration'] = 3000000

Cp['cooling_tower'] =28531392
Cp['coolingwater_pumphouse'] =28531392

# Connection/piping costs (cost per unit length)
C = np.zeros((len(units), len(units)))  
# assign values where flowsheet connection
C[0][1] = 214 
C[1][3] = 56 
C[1][2] = 53 
C[2][3] = 53 
C[3][4] = 89 
C[5][3] = 56 
C[6][3] = 53

C[4][8] = 182 
C[8][7] = 182 
C[9][7] = 182 
C[10][7] = 182 

C[11][12] = 96 

# Cost data is made into a dictionary
C = makeDict([units,units],C,0)

# number of pipes between process units
n_pipe = np.zeros((len(units), len(units)))  
# assign values where flowsheet connection

n_pipe[0][1] = 2 
n_pipe[1][3] = 2 
n_pipe[1][2] = 2 
n_pipe[2][3] = 2 
n_pipe[3][4] = 2 
n_pipe[5][3] = 2 
n_pipe[6][3] = 2

n_pipe[4][8] = 2 
n_pipe[8][7] = 5 
n_pipe[9][7] = 5 
n_pipe[10][7] = 2 

n_pipe[11][12] = 8 

# number of pipe is made into a dictionary
n_pipe = makeDict([units,units],n_pipe,0)   

if SwitchPumpingCost == 1:   
    # Cost of Piping Calculation
    # Fluid mass flowrate in pipe connecting unit i and j (kg/s)
    mass_flowrate = np.zeros((len(units), len(units))) # square matrix with elements unit,unit
    # Fluid velocity (m/s)
    fluid_velocity = np.zeros((len(units), len(units)))
    # Fluid density (kg/m^3)
    fluid_density = np.zeros((len(units), len(units)))
    # Diameter (m)   
    DIA = np.zeros((len(units), len(units)))
    # Friction Factor (-)   
    friction_factor = np.zeros((len(units), len(units)))
    
    #Assign values of mass flowrate where flowsheet connection
    mass_flowrate[0][1] = 65 
    mass_flowrate[1][3] = 53 
    mass_flowrate[1][2] = 49 
    mass_flowrate[2][3] = 49 
    mass_flowrate[3][4] = 53 
    mass_flowrate[5][3] = 53 
    mass_flowrate[6][3] = 49
    
    mass_flowrate[4][8] = 4
    mass_flowrate[8][7] = 4 
    mass_flowrate[9][7] = 4 
    mass_flowrate[10][7] = 4 

    mass_flowrate[11][12] = 98 
    
    #Assign values of fluid velocity where flowsheet connection    
    fluid_velocity[0][1] = 30 
    fluid_velocity[1][3] = 3 
    fluid_velocity[1][2] = 3 
    fluid_velocity[2][3] = 3 
    fluid_velocity[3][4] = 3 
    fluid_velocity[5][3] = 3 
    fluid_velocity[6][3] = 3
    
    fluid_velocity[4][8] =30
    fluid_velocity[8][7] = 30 
    fluid_velocity[9][7] = 30
    fluid_velocity[10][7] = 30 

    fluid_velocity[11][12] = 3 
    
    #Assign values of fluid density where flowsheet connection
    fluid_density[0][1] = 1.2 
    fluid_density[1][3] =998 
    fluid_density[1][2] = 998
    fluid_density[2][3] = 998
    fluid_density[3][4] = 583 
    fluid_density[5][3] = 993
    fluid_density[6][3] = 993
    
    fluid_density[4][8] =1.98
    fluid_density[8][7] = 1.98 
    fluid_density[9][7] = 1.98
    fluid_density[10][7] = 1.98 

    fluid_density[11][12] = 997
    #Assign values of pipe diameter where flowsheet connection
    DIA[0][1] = 1.52
    DIA[1][3] =0.15
    DIA[1][2] = 0.144
    DIA[2][3] = 0.144
    DIA[3][4] = 0.196 
    DIA[5][3] = 0.150
    DIA[6][3] = 0.144
    
    DIA[4][8] =0.299
    DIA[8][7] =0.299 
    DIA[9][7] = 0.299
    DIA[10][7] =0.299

    DIA[11][12] = 0.204
    #Assign values of friction factor where flowsheet connection
    friction_factor[0][1] = 0.0433
    friction_factor[1][3] =0.0162
    friction_factor[1][2] = 0.0162
    friction_factor[2][3] = 0.0162
    friction_factor[3][4] = 0.0149 
    friction_factor[5][3] = 0.016
    friction_factor[6][3] = 0.016

    friction_factor[4][8] =0.014
    friction_factor[8][7] =0.014
    friction_factor[9][7] = 0.014
    friction_factor[10][7] =0.014

    friction_factor[11][12] = 0.0153
    
    # Flow data is made into a dictionary
    mass_flowrate = makeDict([units,units],mass_flowrate,0)
    fluid_velocity = makeDict([units,units],fluid_velocity,0)
    fluid_density = makeDict([units,units],fluid_density,0)
    DIA = makeDict([units,units],DIA,0)
    friction_factor = makeDict([units,units],friction_factor,0)
          
# literature safety distances between process units (Industrial Risk Insurers)
D_min = np.zeros((len(units),len(units)))
D_min = makeDict([units,units],D_min,0)

if SwitchSafetyDistances == 1:
    
    for i in cooling_tower_process_unit:
        for j in storage_tank_process_unit:
            D_min[i][j] =76
    for i in cooling_tower_process_unit:
        for j in moderate_hazard_process_unit:
            D_min[i][j] =31
    for i in cooling_tower_process_unit:
        for j in compressor_process_unit:
            D_min[i][j] =31
    for i in cooling_tower_process_unit:
        for j in large_pump_house:
            D_min[i][j] =31
                    
    for i in moderate_hazard_process_unit:
        for j in storage_tank_process_unit:
            D_min[i][j] =76
    for i in moderate_hazard_process_unit:
        for j in cooling_tower_process_unit:
            D_min[i][j] =31
    for i in moderate_hazard_process_unit:
        for j in moderate_hazard_process_unit:
            D_min[i][j] =15
            D_min[j][i] =D_min[i][j]
    for i in moderate_hazard_process_unit:
        for j in compressor_process_unit:
            D_min[i][j] =9
    for i in moderate_hazard_process_unit:
        for j in large_pump_house:
            D_min[i][j] =9           
             
    for i in storage_tank_process_unit:
        for j in cooling_tower_process_unit:
            D_min[i][j] =76
    for i in storage_tank_process_unit:
        for j in moderate_hazard_process_unit:
            D_min[i][j] =76
    for i in storage_tank_process_unit:
        for j in compressor_process_unit:
            D_min[i][j] =76            
    for i in storage_tank_process_unit:
        for j in large_pump_house:
            D_min[i][j] =76
            
    for i in large_pump_house:
        for j in large_pump_house:
            D_min[i][j] =9
            D_min[j][i] =D_min[i][j]
    for i in large_pump_house:
        for j in cooling_tower_process_unit:
            D_min[i][j] =31
    for i in large_pump_house:
        for j in moderate_hazard_process_unit:
            D_min[i][j] =9      
    for i in large_pump_house:
        for j in compressor_process_unit:
            D_min[i][j] =9
    for i in large_pump_house:
        for j in storage_tank_process_unit:    
             D_min[i][j] =76
             
    for i in compressor_process_unit:
        for j in compressor_process_unit:
            D_min[i][j] =9
            D_min[j][i] =D_min[i][j]
    for i in compressor_process_unit:
        for j in cooling_tower_process_unit:
            D_min[i][j] =31
    for i in compressor_process_unit:
        for j in moderate_hazard_process_unit:
            D_min[i][j] =9       
    for i in compressor_process_unit:
        for j in compressor_process_unit:
            D_min[i][j] =9
    for i in compressor_process_unit:
        for j in storage_tank_process_unit:    
             D_min[i][j] =76    
            
else:
    for i in units:
        for j in units:
            D_min[i][j]= 0
       
#frequency of chemical release event
freq_c = dict.fromkeys(pertinent_units_Vlc)
freq_c['absorber'] = 2.55e-4
freq_c['stripper'] = 2.55e-4
freq_c['storage_tank'] = 2.55e-4

#frequency of fire and explosion event
freq_f = dict.fromkeys(pertinent_units_FEI)
freq_f['absorber'] = 3.14e-2
freq_f['stripper'] = 3.14e-2
freq_f['storage_tank'] = 3.14e-2

# Land use model
if SwitchLandUse == 1:
    # Land cost (GBP per unit distance squared)
    LC = 125
    # Number of grid points in square plot
    N = 250
    # Size of one grid square (m)
    g = 2   
    if SwitchSquarePlot==1:
    # Define the set for binary variable Gn
        gridsize = list(range(1,N))         
    else: 
       N1=250
       N2=250
       gridsize1 = list(range(1,N1))
       gridsize2 = list(range(1,N2))
else:
    # Maximum plot size if land use model not switched on
    xmax = 1000
    ymax = 1000

# F&EI model
if SwitchFEI == 1:
    # Operating conditions
    MF = dict.fromkeys(pertinent_units_FEI)
    F1 = dict.fromkeys(pertinent_units_FEI)
    F2 = dict.fromkeys(pertinent_units_FEI)
    F = dict.fromkeys(pertinent_units_FEI)
    De1 = dict.fromkeys(pertinent_units_FEI)
    De2 = dict.fromkeys(pertinent_units_FEI)
    De = dict.fromkeys(pertinent_units_FEI)
    DF = dict.fromkeys(pertinent_units_FEI)
    
    # Assign values
    MF['absorber'] = 10
    MF['stripper'] = 10
    MF['storage_tank'] = 10

    F1['absorber'] = 1.2
    F1['stripper'] = 1.2
    F1['storage_tank'] = 1.2

    F2['absorber'] = 1.6
    F2['stripper'] = 3.7
    F2['storage_tank'] = 1.6
    
    F3 = {i: F1[i]*F2[i] for i in pertinent_units_FEI}
    F = {i: MF[i]*F3[i] for i in pertinent_units_FEI}
    
    #F&EI distance calculations
    De1 = {i: 0.256*F[i] for i in pertinent_units_FEI}
    De2 = {i: ((alpha[i]**2) + (beta[i]**2))**0.5 for i in pertinent_units_FEI}
    De = {i: ((De1[i] +(De2[i]/2))) for i in pertinent_units_FEI}    

# damage factor equations based on value of MF, F3[i] = 8 used when it is greater than 8
    def DFcalc(value):
        if value == 1:
            if F3[i] > 8:
                return 0.003907 + 0.002957*8 + 0.004031*8**2 - 0.00029*8**3
            return 0.003907 + 0.002957*F3[i] + 0.004031*F3[i]**2 - 0.00029*F3[i]**3
        elif value == 4:
            if F3[i] > 8:
                return 0.025817 + 0.019071*8 -0.00081*8**2 + 0.000108*8**3
            return 0.025817 + 0.019071*F3[i] -0.00081*F3[i]**2 + 0.000108*F3[i]**3
        elif value == 10:
            if F3[i] > 8:
                return 0.098582 + 0.017596*8 + 0.000809*8**2 - 0.000013*8**3
            return 0.098582 + 0.017596*F3[i] + 0.000809*F3[i]**2 - 0.000013*F3[i]**3
        elif value == 14:
            if F3[i] > 8:
                return 0.20592 + 0.018938*8 + 0.007628*8**2 - 0.00057*8**3
            return 0.20592 + 0.018938*F3[i] + 0.007628*F3[i]**2 - 0.00057*F3[i]**3
        elif value == 16:
            if F3[i] > 8:
                return 0.256741 + 0.019886*8 + 0.011055*8**2 - 0.00088*8**3
            return 0.256741 + 0.019886*F3[i] + 0.011055*F3[i]**2 - 0.00088*F3[i]**3
        elif value == 21:
            if F3[i] >8:
                return 0.340314 + 0.076531*8 + 0.003912*8**2 - 0.00073*8**3
            return 0.340314 + 0.076531*F3[i] + 0.003912*F3[i]**2 - 0.00073*F3[i]**3
        elif value == 24:
            if F3[i] > 8:
                return 0.395755 + 0.096443*8 - 0.00135*8**2 - 0.00038*8**3
            return 0.395755 + 0.096443*F3[i] - 0.00135*F3[i]**2 - 0.00038*F3[i]**3
        elif value == 29:
            if F3[i] > 8:
                return 0.484766 + 0.094288*8 - 0.00216*8**2 - 0.00031*8**3
            return 0.484766 + 0.094288*F3[i] - 0.00216*F3[i]**2 - 0.00031*F3[i]**3
        elif value == 40:
            if F3[i] > 8:
                return 0.554175 + 0.080772*8 + 0.000332*8**2 - 0.00044*8**3
            return 0.554175 + 0.080772*F3[i] + 0.000332*F3[i]**2 - 0.00044*F3[i]**3
        else:
            raise ValueError(value)
    for i in pertinent_units_FEI:
        DF[i] = DFcalc(MF[i])

    # With literature values for DF and De

    # Upper bound for actual maximum probable property damage cost
    U = 1e8

    # Protection device model
    if SwitchProt == 1:
        # Define protection device configuration
        configurations = list(range(1, len(units)))
        # Loss control credit factor of protection device configuration k on item i
        gamma = np.zeros((len(pertinent_units), len(configurations)))
        gamma = makeDict([pertinent_units,configurations],gamma,0)
        # assign values
        gamma['ABS'][1] = 1

        # purchase cost of configuration k on unit i
        P = np.zeros((len(pertinent_units),len(configurations)))
        P = makeDict([pertinent_units,configurations],P,0)
        # assign values
        P['ABS'][1] = 0
        
        P['STR'][1] = 0

# CEI Factors
if SwitchCEI == 1:
 # Initialise dictionaries
    # Operating conditions
    T = dict.fromkeys(pertinent_units_Vlc)
    Pg = dict.fromkeys(pertinent_units_Vlc)
    w = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}
    rhol = dict.fromkeys(pertinent_units_Vlc)
    MWunit = dict.fromkeys(pertinent_units_Vlc)
    INV = dict.fromkeys(pertinent_units_Vlc)
    # Assign values
    T['absorber'] = 33  # degc
    T['stripper'] = 119
    T['storage_tank'] = 30

    Pg['absorber'] = 5  # kPa
    Pg['stripper'] = 57
    Pg['storage_tank'] = 400
    
    w['absorber']['MEA'] = 1  # mass frac
    w['stripper']['MEA'] = 1
    w['storage_tank']['MEA'] = 1

    rhol['absorber'] = 993
    rhol['stripper'] = 993
    rhol['storage_tank'] = 993

    MWunit['absorber'] = 61.08  # g/mol
    MWunit['stripper'] = 61.08
    MWunit['storage_tank'] = 61.08

    INV['absorber'] = 2062421  # kg
    INV['stripper'] = 254208
    INV['storage_tank'] = 2780400
    
    AQ = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}  # airborne quantity produced
    AQf = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}  # airborne quantity produced by flash
    AQp = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}  # airborne quantity produced by pool
    Fv = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}  # fraction flashed
    Pvap = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}
    CEI = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}
    Dc = {i: dict.fromkeys(hazardous_chemicals) for i in pertinent_units_Vlc}
    Dh = dict.fromkeys(pertinent_units_Vlc)
    Liq = dict.fromkeys(pertinent_units_Vlc)
    Deltah = dict.fromkeys(pertinent_units_Vlc)
    WT = dict.fromkeys(pertinent_units_Vlc)
    WP = dict.fromkeys(pertinent_units_Vlc)
    AP = dict.fromkeys(pertinent_units_Vlc)
    heatratio = dict.fromkeys(hazardous_chemicals)
    Tb = dict.fromkeys(hazardous_chemicals)
    antoineA = dict.fromkeys(hazardous_chemicals)
    antoineB = dict.fromkeys(hazardous_chemicals)
    antoineC = dict.fromkeys(hazardous_chemicals)
    MW = dict.fromkeys(hazardous_chemicals)
    Pr = dict.fromkeys(hazardous_chemicals)
    # prob_death = dict.fromkeys(hazardous_chemicals)
    CONC = dict.fromkeys(hazardous_chemicals)
    
    # # Assign values
    # te = 10  # minutes of exposure time
    # prob_death['MEA'] = 0.5  # probability of death for probit
    CONC['MEA']=124.9
    Dh['absorber'] = 50.8  # mm
    Dh['stripper'] = 50.8  # hole diameter
    Dh['storage_tank'] = 50.8 
    
    Deltah['absorber'] = 1  # m
    Deltah['stripper'] = 1
    Deltah['storage_tank'] = 1

    heatratio['MEA'] = 0.0044  # 1/degc
    Tb['MEA'] = 170  # degc
    antoineA['MEA'] = 4.29252
    antoineB['MEA'] = 1408.9
    antoineC['MEA'] = -116.1
    MW['MEA'] = 61.08
    
       # Compute CEI factors
       # Probit equation
    # for h in hazardous_chemicals:
    #     Pr[h] = norm.ppf(prob_death[h]) + 5
    #     CONC[h] = math.sqrt(math.exp(Pr[h] + 17.5)/te)  # mg/m^3
    for i in pertinent_units_Vlc:
        for h in hazardous_chemicals:
            # Vapour pressure (kPa)
            Pvap[i][h] = 100 * 10**(antoineA[h] - antoineB[h]/(antoineC[h] + (Tb[h] + 273.15)))
    for i in pertinent_units_Vlc:
        # Liquid flowrate (kg/s)
        Liq[i] = 9.44E-07 * Dh[i]**2 * rhol[i] * math.sqrt(1000*Pg[i]/rhol[i] + 9.8*Deltah[i])
        if 300*Liq[i] >= INV[i]:
            Liq[i] = INV[i]/300
        else:
              Liq[i] = 9.44E-07 * Dh[i]**2 * rhol[i] * math.sqrt(1000*Pg[i]/rhol[i] + 9.8*Deltah[i])
        # Total liquid release (kg)
        if Liq[i] < INV[i]/900:
            WT[i] = 900*Liq[i]
        elif Liq[i] >= INV[i]/900:
            WT[i] = INV[i]
        else:
            pass
        # Fraction flashed
        for h in hazardous_chemicals:
            if Tb[h]<T[i]:
             Fv[i][h] = w[i][h] * heatratio[h] * (T[i] - Tb[h])
             if Fv[i][h]/w[i][h] < 0.2:
                # Air quantity from flashed (kg/s)
                AQf[i][h] = 5*Fv[i][h]*Liq[i]
                # Pool size (kg)
                WP[i] = WT[i] * (1 - 5*Fv[i][h]/w[i][h])
             elif Fv[i][h]/w[i][h] >= 0.2:
                AQf[i][h] = w[i][h]*Liq[i]
                WP[i] = 0
            if Tb[h]>T[i]:
                AQf[i][h] = 0
                WP[i] = WT[i]
    
           # Area of pool (m^2) 
        AP[i] = 100 * WP[i]/rhol[i]
        for h in hazardous_chemicals:
            # Air quantity from evaporating pool (kg/s)
            if T[i]<Tb[h]:
             AQp[i][h] = 9E-04 * AP[i]**0.95 * ((MW[h] * Pvap[i][h] * w[i][h])/(T[i] + 273))
            if T[i]>=Tb[h]:
             AQp[i][h] = 9E-04 * AP[i]**0.95 * ((MW[h] * Pvap[i][h] * w[i][h])/(Tb[h] + 273))   
            # Total air quantity
            AQ[i][h] = AQf[i][h] + AQp[i][h]
            # Chemical  exposure index
            CEI[i][h] = 655.1 * math.sqrt(AQ[i][h]/CONC[h])
            # Hazard distance (m)
            Dc[i][h] = 6551 * math.sqrt(AQ[i][h]/CONC[h])
            if Dc[i][h] > 10000:
                Dc[i][h] = 10000
                
    # # Find max chemical hazard distance
    Dcvalues = []
    for a, b in Dc.items():
        for c, d in b.items():
            Dcvalues.append(d)
    maxDc = max(Dcvalues)
    
# Dc = {'abs': {'MEA': 135.8}, 'circ':{'MEA': 136.5}, 'strip':{'MEA': 120}} #numbers calculated using Excel (CEI Guide)

# # Occupancy calculations
# # Number of workers
if SwitchFEIVle == 1 or SwitchCEI == 1 :
    Nw = dict.fromkeys(units)
    Nw['direct_contact_cooler']=1
    Nw['absorber'] =1
    Nw['lean_amine_cooler'] = 1
    Nw['cross_heat_exchanger'] =1
    Nw['stripper'] =1
    Nw['storage_tank'] = 1
    Nw['reclaimer'] =1    
    
    Nw['compressor_house'] = 1
    Nw['compressor_KOdrum'] =1
    Nw['compressor_interstage_cooler'] = 1
    Nw['TEG_dehydration'] = 1
    
    Nw['cooling_tower'] =1
    Nw['coolingwater_pumphouse'] =1    
    
    # Percentage of time present at unit
    tw = dict.fromkeys(units)
    tw['direct_contact_cooler']=0.1
    tw['absorber'] =0.1
    tw['lean_amine_cooler'] = 0.1
    tw['cross_heat_exchanger'] =0.1
    tw['stripper'] =0.1
    tw['storage_tank'] = 0.1
    tw['reclaimer'] =0.1    
    
    tw['compressor_house'] = 0.1
    tw['compressor_KOdrum'] =0.1
    tw['compressor_interstage_cooler'] = 0.1
    tw['TEG_dehydration'] =0.1
   
    tw['cooling_tower'] =0.1
    tw['coolingwater_pumphouse'] =0.1    
        
    # Occupancy
    OCC = dict.fromkeys(units)
    OCC = {i: Nw[i]*tw[i] for i in units}
    # Cost of life (GBP)
    Cl = dict.fromkeys(units)
    for i in units:
        Cl[i] = 1.8e6

# --------------Define Variables--------------
# Base layout model

# 1 if length of item i is equal to alpha; 0 otherwise
O = LpVariable.dicts("O",(units),lowBound=0,upBound=1,cat="Integer")
# pair-wise values to ensure equipment items do not overlap
E1 = LpVariable.dicts("E1",(units,units),lowBound=0,upBound=1,cat="Integer")
E2 = LpVariable.dicts("E2",(units,units),lowBound=0,upBound=1,cat="Integer")
# is unit i to the right or above unit j
Wx = LpVariable.dicts("Wx",(units,units),lowBound=0,upBound=1,cat="Integer")
Wy = LpVariable.dicts("Wy",(units,units),lowBound=0,upBound=1,cat="Integer")

# Define continuous variables for base layout model
l = LpVariable.dicts("l",(units),lowBound=0,upBound=None,cat="Continuous")
    # breadth of item i
d = LpVariable.dicts("d",(units),lowBound=0,upBound=None,cat="Continuous")
    # relative distance in x coordinates between items i and j, if i is to the right of j
R = LpVariable.dicts("R",(units,units),lowBound=0,upBound=None,cat="Continuous")
    # relative distance in x coordinates between items i and j, if i is to the left of j
L = LpVariable.dicts("L",(units,units),lowBound=0,upBound=None,cat="Continuous")
    # relative distance in y coordinates between items i and j, if i is above j
A = LpVariable.dicts("A",(units,units),lowBound=0,upBound=None,cat="Continuous")
    # relative distance in y coordinates between items i and j, if i is below j
B = LpVariable.dicts("B",(units,units),lowBound=0,upBound=None,cat="Continuous")
    # total rectilinear distance between items i and j
D = LpVariable.dicts("D",(units,units),lowBound=0,upBound=None,cat="Continuous")
    # coordinates of the geometrical centre of item i
x = LpVariable.dicts("X",(units),lowBound=None,upBound=None,cat="Continuous")
y = LpVariable.dicts("Y",(units),lowBound=None,upBound=None,cat="Continuous")

# Define continuousnterunit connection cost
CD = LpVariable.dicts("CD",(units,units),lowBound=0,upBound=None,cat="Continuous")
# Total connection cost
SumCD = LpVariable("SumCD",lowBound=0,upBound=None,cat="Continuous")

if SwitchLandUse == 1:
    # Total land cost
    TLC = LpVariable("TLC",lowBound=0,upBound=None,cat="Continuous")
    if SwitchSquarePlot==1:
    # N binary variables representing plot grid (squareplot)
        Gn = LpVariable.dicts("Gn",(gridsize),lowBound=0,upBound=1,cat="Integer")      
    else:    
    # N1 and N2 binary variables representing plot grid (rectangle plot)
        Gn1n2 = LpVariable.dicts("Gn1n2",(gridsize1, gridsize2),lowBound=0,upBound=1,cat="Integer")
    
if SwitchFEI == 1:
    # 1 if j is allocated within the area of exposure if i; 0 otherwise
    Psie = LpVariable.dicts("Psie",(pertinent_units_FEI,units),lowBound=0,upBound=1,cat="Integer")
    # total rectilinear distance if D < De
    Din = LpVariable.dicts("Din",(pertinent_units_FEI,units),lowBound=0,upBound=None,cat="Continuous")
    # total rectilinear distance if D > De
    Dout = LpVariable.dicts("Dout",(pertinent_units_FEI,units),lowBound=0,upBound=None,cat="Continuous")
    # value of area of exposure of incident on i
    Ve = LpVariable.dicts("Ve",(pertinent_units_FEI),lowBound=0,upBound=None,cat="Continuous")
    # second term for value of area of exposure of incident on i
    #Ve2 = LpVariable.dicts("Ve2",(pertinent_units),lowBound=0,upBound=None,cat="Continuous")
    Ve2 = LpVariable.dicts("Ve2",(pertinent_units_FEI,units),lowBound=0,upBound=None,cat="Continuous")

    # base maximum probable property damage cost for pertinent unit i
    Omega0 = LpVariable.dicts("Omega0",(pertinent_units_FEI),lowBound=0,upBound=None,cat="Continuous")
    # Total actual maximum probable property damage cost
    SumOmega = LpVariable("SumOmega",lowBound=0,upBound=None,cat="Continuous")

if SwitchProt == 1:
    # 1 if protection device configuration k is installed on item i; 0 otherwise
    Z = LpVariable.dicts("Z",(pertinent_units_FEI,configurations),lowBound=0,upBound=1,cat="Integer")
    # actual maximum probable property damage cost for pertinent unit i
    Omega = LpVariable.dicts("Omega",(pertinent_units_FEI),lowBound=0,upBound=None,cat="Continuous")
    # linearisation variable denoting product of Omega0 and Z
    linOmega0 = LpVariable.dicts("linOmega0",(pertinent_units_FEI,configurations),lowBound=0,upBound=None,cat="Continuous")
    # Total protection devices cost
    SumPZ = LpVariable("SumPZ",lowBound=0,upBound=None,cat="Continuous")

if SwitchCEI == 1:
    # 1 if j is allocated within chemical the area of exposure if i; 0 otherwise
    Psic = LpVariable.dicts("Psic",(pertinent_units_Vlc,units,hazardous_chemicals),lowBound=0,upBound=1,cat="Integer")
    #'Distances' involved in chemical area of exposure
    Dc_in = LpVariable.dicts("Dc_in",(pertinent_units_Vlc,units),lowBound=0,upBound=None,cat="Continuous")
    Dc_out = LpVariable.dicts("Dc_out",(pertinent_units_Vlc,units),lowBound=0,upBound=None,cat="Continuous")
    # value of life in chemical area of exposure calculations
    Vlc = LpVariable.dicts("Vlc",(pertinent_units_Vlc,units),lowBound=0,upBound=None,cat="Continuous")
    Vlc2 = LpVariable.dicts("Vlc2",(pertinent_units_Vlc,units,hazardous_chemicals),lowBound=0,upBound=None,cat="Continuous")
    # Total cost of life due to chemical release
    SumVlc = LpVariable("SumVlc",lowBound=0,upBound=None,cat="Continuous")
    
if SwitchPumpingCost ==1:
    #Pressure dop of pipe connecting unit i and j
    pressure_drop =  LpVariable.dicts("pressure_drop",(units,units),lowBound=0,upBound=None,cat="Continuous")
    #Power of pumping for pipe connecting unit i and j
    pumping_power =  LpVariable.dicts("pumping_power",(units,units),lowBound=0,upBound=None,cat="Continuous")
    #Cost of pumping for pipe connecting unit i and j
    pumping_cost =  LpVariable.dicts("pumping_cost",(units,units),lowBound=0,upBound=None,cat="Continuous")
    #Cost of pumping for pipe connecting unit i and j
    sum_pumping_cost =  LpVariable("sum_pumping_cost",lowBound=0,upBound=None,cat="Continuous")
     
# --------------Define Objective Function--------------
# if SwitchFEI == 1 and SwitchProt == 1 and SwitchLandUse == 1 and SwitchCEI == 1 and SwitchPumpingCost==1:
#     layout += SumCD + TLC + SumOmega + SumPZ + SumVlc+sum_pumping_cost   
# elif SwitchFEI == 1 and SwitchProt == 1 and SwitchLandUse == 1 and SwitchCEI == 1 and SwitchPumpingCost==0:  
#     layout += SumCD + TLC + SumOmega + SumPZ + SumVlc
# elif SwitchFEI == 1 and SwitchProt == 1 and SwitchLandUse == 1 and SwitchCEI == 0  and SwitchPumpingCost==1:
#     layout += SumCD + TLC + SumOmega + SumPZ
# elif SwitchFEI == 1 and SwitchProt == 1 and SwitchLandUse == 0 and SwitchCEI == 0  and SwitchPumpingCost==1:
#     layout += SumCD + SumOmega + SumPZ
# elif SwitchFEI == 1 and SwitchProt == 0 and SwitchLandUse == 0 and SwitchCEI == 0  and SwitchPumpingCost==1:
#     layout += SumCD + SumOmega
# elif SwitchFEI == 1 and SwitchProt == 0 and SwitchLandUse == 1:
#     layout += SumCD + TLC + SumOmega
# elif SwitchFEI == 0 and SwitchProt == 0 and SwitchLandUse == 1:
#     layout += SumCD + TLC   
# else:
layout  += SumCD + TLC + SumOmega+SumVlc+sum_pumping_cost   

# --------------Define Constraints and Objective Function Contributions--------------
# Base model constraints for all units i
for i in units:
    # Orientation constraints (1 - 2)
    layout += l[i] == alpha[i]*O[i] + beta[i]*(1 - O[i])
    layout += d[i] == alpha[i] + beta[i] - l[i]
    # Lower and upper bounds of coordinates (19 - 22)
    layout += x[i] >= 0.5*l[i]
    layout += y[i] >= 0.5*d[i]

for idxj, j in enumerate(units):
    for idxi, i in enumerate(units): 
        if idxj > idxi:
            # Distance calculation constraints (3 - 10)
            layout += R[i][j] - L[i][j] == x[i] - x[j]
            layout += A[i][j] - B[i][j] == y[i] - y[j]
            layout += R[i][j] <= M*Wx[i][j]
            layout += L[i][j] <= M*(1 - Wx[i][j])
            layout += A[i][j] <= M*Wy[i][j]
            layout += B[i][j] <= M*(1 - Wy[i][j])
            layout += D[i][j] == R[i][j] + L[i][j] + A[i][j] + B[i][j]
            layout += D[i][j] == D[j][i]
            # Nonoverlapping constraints (15 - 18)
            layout += x[i] - x[j] + M*(E1[i][j] + E2[i][j]) >= ((l[i] + l[j])/2)+D_min[i][j]
            layout += x[j] - x[i] + M*(1 - E1[i][j] + E2[i][j]) >= ((l[i] + l[j])/2)+D_min[i][j]
            layout += y[i] - y[j] + M*(1 + E1[i][j] - E2[i][j]) >= ((d[i] + d[j])/2)+D_min[i][j]
            layout += y[j] - y[i] + M*(2 - E1[i][j] - E2[i][j]) >= ((d[i] + d[j])/2)+D_min[i][j]
            # These constraints ensure consistency in interdependent variables
            layout += L[i][j] == R[j][i]
            layout += R[i][j] == L[j][i]
            layout += A[i][j] == B[j][i]
            layout += B[i][j] == A[j][i]
            layout += Wx[i][j] == 1 - Wx[j][i]
            layout += Wy[i][j] == 1 - Wy[j][i]
# piping cost (GBP/yr)
for i in units:
    for j in units:
            layout += CD[i][j] == C[i][j]*D[i][j]*n_pipe[i][j]            

# Objective function contribution for base model (GBP/yr)
layout += SumCD == lpSum([CD[i][j] for i in units for j in units])

# Land use constraints (or set max plot size if not used)
if SwitchLandUse == 1:
    if SwitchSquarePlot==1:  
        for i in units:
            # Land area approximation constraints for square plot (24 - 25)
            layout += x[i] + 0.5*l[i] <= lpSum(n*g*Gn[n] for n in range(1,N))
            layout += y[i] + 0.5*d[i] <= lpSum(n*g*Gn[n] for n in range(1,N))
            # Only 1 grid size selected (23)
            layout += lpSum(Gn[n] for n in range(1,N)) == 1
            # Objective function contribution for land use model (GBP/yr)
        layout += TLC == (LC*lpSum(Gn[n]*(n*g)**2 for n in range(1,N)))*annualised_factor
    else:    
        for i in units:
        # for rectangular plot
            layout += x[i] + 0.5*l[i] <= lpSum(n1*g*Gn1n2[n1][n2] for n1 in range(1,N1) for n2 in range(1,N2))
            layout += y[i] + 0.5*d[i] <= lpSum(n2*g*Gn1n2[n1][n2] for n1 in range(1,N1) for n2 in range(1,N2))
        # rectangular plot
            layout += lpSum(Gn1n2[n1][n2] for n1 in range(1,N1) for n2 in range (1,N2)) == 1   
        # rectangular plot
        layout += TLC == (LC*lpSum(Gn1n2[n1][n2]*((n1*n2*g**2)) for n1 in range(1,N1) for n2 in range(1,N2)))*annualised_factor
else:
    layout += x[i] + 0.5*l[i] <= xmax
    layout += y[i] + 0.5*d[i] <= ymax

# F&EI Constraints
if SwitchFEI == 1:
    # for all i in pertinent units, j in units, j != i
    for i in units:
        if i in pertinent_units_FEI:
            for j in units:
                if i != j:
                    # Area of exposure constraints (26 - 28)
                    layout += Din[i][j] + Dout[i][j] == D[i][j]
                    layout += Din[i][j] <= De[i]*Psie[i][j]*1.414
                    layout += Dout[i][j] >= De[i]*(1 - Psie[i][j])*1.414
                    layout += Dout[i][j] <= M*(1 - Psie[i][j])
                    # Cost of life (37)
                    if SwitchFEIVle == 1:
                        layout += Ve2[i][j] == Cp[j]*Psie[i][j] + OCC[j]*Cl[j]*Psie[i][j] - Cp[j]*Din[i][j]/(De[i]*1.414) - OCC[j]*Cl[j]*Din[i][j]/(De[i]*1.414)
                    else:
                        layout += Ve2[i][j] == Cp[j]*Psie[i][j] - Cp[j]*Din[i][j]/(De[i]*1.414)
                else:
                    layout += Ve2[i][j] == 0
                    
    # for all i in pertinent units
    for i in pertinent_units_FEI:
        # Value of area of exposure constraint (30ii)
        if SwitchFEIVle == 1:
            layout += Ve[i] == Cp[i] + OCC[i]*Cl[i] + lpSum([Ve2[i][j] for j in units])
        else:
            layout += Ve[i] == Cp[i] + lpSum([Ve2[i][j] for j in units])
        # Base maximum probable property damage cost (31)
        layout += Omega0[i] == DF[i]*Ve[i]*freq_f[i]

    if SwitchProt == 1:
        # for all i in pertinent units,k in configurations
        for i in pertinent_units_FEI:
            for k in configurations:
                # Linearisation of product of base MPPD and configuration binary variable (34)
                layout += linOmega0[i][k] <= U*Z[i][k]
        # for all i in pertinent units
        for i in pertinent_units_FEI:
            # Only one configuration active (33)
            layout += lpSum([Z[i][k] for k in configurations]) == 1
            # Base maximum probable property damage cost is sum of linearisation term (35)
            layout += Omega0[i] == lpSum([linOmega0[i][k] for k in configurations])
            # Actual maximum probable property damage cost (36)
            layout += Omega[i] == lpSum([gamma[i][k]*linOmega0[i][k] for k in configurations])

            # Objective function contribution with protection devices
            layout += SumOmega == lpSum(Omega[i] for i in pertinent_units_FEI) 
            layout += SumPZ == (lpSum(P[i][k]*Z[i][k] for i in pertinent_units_FEI for k in configurations))*annualised_factor
    else:
        # Objective function contribution without protection devices
        layout += SumOmega == lpSum(Omega0[i] for i in pertinent_units_FEI)

if SwitchCEI == 1:
    # for all i in pertinent units, j in units, j != i
    for i in units:
        if i in pertinent_units_Vlc:
            for j in units:
                if i != j:
                    for h in hazardous_chemicals:
                        #Area of exposure constraints (39 - 40)
                        layout += Dc_in[i][j] + Dc_out[i][j] == D[i][j]
                        layout += Dc_in[i][j] <= Dc[i][h]*Psic[i][j][h]
                        layout += Dc_out[i][j] >= Dc[i][h]*(1 - Psic[i][j][h])
                        layout += Dc_out[i][j] <= M*(1 - Psic[i][j][h])
                        layout += Vlc2[i][j][h] == OCC[j]*Cl[j]*Psic[i][j][h] - OCC[j]*Cl[j]*Dc_in[i][j]/Dc[i][h]
                else:
                    layout += Vlc2[i][j][h] == 0

    for i in pertinent_units_Vlc:
        # Value of chemical area of exposure constraint (43)
        layout += Vlc[i] == (OCC[i]*Cl[i] + lpSum([Vlc2[i][j][h] for j in units for h in hazardous_chemicals]))*freq_c[i]

    # Objective function term
    layout += SumVlc == lpSum(Vlc[i] for i in pertinent_units_Vlc)

if SwitchPumpingCost ==1:
    for i in units:
        for j in units:
            #Pressure drop of pipe connecting unit i and j  
            if DIA[i][j]>0 and fluid_density[i][j]>0:
                layout += pressure_drop[i][j]*2*DIA[i][j] == 8*friction_factor[i][j]*D[i][j]*fluid_density[i][j]*(fluid_velocity[i][j])**2  
                layout += pumping_power[i][j]*fluid_density[i][j]*pump_efficiency ==  (mass_flowrate[i][j]*pressure_drop[i][j])
            else:
                layout += pressure_drop[i][j]==0
                layout += pumping_power[i][j]==0
            #Cost of pumping for pipe connecting unit i and j
            layout += pumping_cost[i][j] ==  (electricity_cost*operating_hour*pumping_power[i][j])*n_pipe[i][j]  
            #Objective fucntion contribution from cost of pumping for pipe connecting unit i and j
    layout +=sum_pumping_cost ==  lpSum([pumping_cost[i][j] for i in units for j in units])
            
# --------------Fixing Variable Values--------------
# Define function to fix value
def fix_variable(variable, value):
      variable.setInitialValue(value)
      variable.fixValue()
fix_variable(y['direct_contact_cooler'],alpha['direct_contact_cooler']/2)
fix_variable(x['storage_tank'],alpha['storage_tank']/2)
fix_variable(x['reclaimer'],alpha['reclaimer']/2)

fix_variable(x['direct_contact_cooler'],158.00000000000006)
fix_variable(x['absorber'],87.00000000000006)
fix_variable(x['lean_amine_cooler'],34.00000000000006)
fix_variable(x['cross_heat_exchanger'],41.00000000000006)
fix_variable(x['stripper'],140)
fix_variable(x['storage_tank'],17.0)
fix_variable(x['reclaimer'],9.0)
fix_variable(x['compressor_house'],165.0)
fix_variable(x['compressor_KOdrum'],139.0)
fix_variable(x['compressor_interstage_cooler'],189.0)
fix_variable(x['TEG_dehydration'],192.0)

fix_variable(y['direct_contact_cooler'],33.0)
fix_variable(y['absorber'],33.00000000000001)
fix_variable(y['lean_amine_cooler'],33.00000000000001)
fix_variable(y['cross_heat_exchanger'],72.00000000000045)
fix_variable(y['stripper'],111.00000000000045)
fix_variable(y['storage_tank'],174.00000000000045)
fix_variable(y['reclaimer'],72.00000000000045)
fix_variable(y['compressor_house'],169.00000000000045)
fix_variable(y['compressor_KOdrum'],169.00000000000045)
fix_variable(y['compressor_interstage_cooler'],169.00000000000045)
fix_variable(y['TEG_dehydration'],136.00000000000045)




# --------------Initiate Solve--------------
layout.writeLP("DowFEI.lp")
#CPLEXsolver = CPLEX_PY(msg=1, warmStart=1, gapRel=0, logPath='cplex.log')
CPLEXsolver = CPLEX_PY(msg=1, gapRel=0)
CBCsolver = PULP_CBC_CMD(msg=1)
starttime = time.perf_counter()
if solver == 1:
    layout.solve(CPLEXsolver)
elif solver == 2:
    layout.solve(CBCsolver)
totaltime = time.perf_counter() - starttime
# Print solver status
print("Status: ", LpStatus[layout.status])

#--------------Print Results--------------
# Print variable and objective function values
for v in layout.variables():
    print(v.name, "=", v.varValue)
print("Total cost of connections =", SumCD.varValue)
if SwitchLandUse == 1:
    if SwitchSquarePlot == 1:
        for n in range(1, N):
            if Gn[n].varValue == 1:
                print("Size of land area =", (n*g)**2, "metres square")
        print("Total cost of land =", TLC.varValue, "GBP/yr")
    
    else:
        for n1 in range(1, N1):
            for n2 in range(1,N2):
                if Gn1n2[n1][n2].varValue == 1:
                    print("Size of land area =", (n1*n2)*g**2, "metres square")
        print("Total cost of land =", TLC.varValue)
if SwitchFEI == 1:
    print("Total actual MPPD =", SumOmega.varValue)
if SwitchProt == 1:
    print("Total cost of protection devices =", SumPZ.varValue)
if SwitchCEI == 1:
    print("Total cost of fatality due to chemical release =", SumVlc.varValue)
print("Elapsed time =", totaltime)

#--------------Export Results--------------
filename = 'Optimisation_Plot.csv'
with open(filename, 'w', newline='') as file:
    # Write objective function
    writer = csv.writer(file)
    writer.writerow(['Objective', value(layout.objective)])
    
    # Write coordinates
    fieldnames = ['unit','x', 'y', 'l', 'd']
    writer = csv.DictWriter(file, fieldnames=fieldnames)    
    writer.writeheader()
    for i in units:
        writer.writerow({'unit': i, 'x': x[i].varValue, 'y': y[i].varValue, 'l': l[i].varValue, 'd': d[i].varValue})

filename = 'Optimisation_Results.csv'
with open(filename, 'w', newline='') as file:
    # Write value of all variables
    writer = csv.writer(file, delimiter=',')    
    for v in layout.variables():
        writer.writerow([v.name, v.varValue])

filename = 'ConnectionCosts_Results.csv'
with open(filename, 'w', newline='') as file:
    # Write value of all variables
    if SwitchPumpingCost==1:
        fieldnames = ['unit i','unit j','Annual capital cost per metre of pipe (GBP/m.yr)','Total Rectilinear Distance (m)','Piping Cost (GBP/yr)','Total Piping Cost (GBP/yr)',
                    'Inner Pipe Diameter (m)','Pressure Drop (N/m^2)','Power of Pump (W)','Pumping Cost (GBP/yr)','Total Pumping Cost (GBP/yr)']                    
        writer = csv.DictWriter(file, fieldnames=fieldnames)    
        writer.writeheader()
        for i in units:
            for j in units:
                  writer.writerow({ 'unit i':i,'unit j':j,'Annual capital cost per metre of pipe (GBP/m.yr)':C[i][j], 'Total Rectilinear Distance (m)':D[i][j].varValue,'Piping Cost (GBP/yr)':CD[i][j].varValue,
                                    'Inner Pipe Diameter (m)':DIA[i][j],'Pressure Drop (N/m^2)':pressure_drop[i][j].varValue,'Power of Pump (W)':pumping_power[i][j].varValue,'Pumping Cost (GBP/yr)':pumping_cost[i][j].varValue})
        writer.writerow({'Total Piping Cost (GBP/yr)':SumCD.varValue})
        writer.writerow({'Total Pumping Cost (GBP/yr)':sum_pumping_cost.varValue})
                          
if SwitchFEI ==1:
    filename = 'FEI_Results.csv'
    with open(filename, 'w', newline='') as file:
        # Write value of all variables
          fieldnames = ['Units','General Process Hazards,F1','Special Process Hazard,F2', 'Process Unit Hazard Factor,F3','Fire & Explosion Index','Distance of Exposure','Damage Factor','Value of Exposure Area','Base Maximum Property Damage Cost']
          writer = csv.DictWriter(file, fieldnames=fieldnames)    
          writer.writeheader()
          for i in pertinent_units_FEI:
              writer.writerow({ 'Units':i,'General Process Hazards,F1':F1[i],'Special Process Hazard,F2':F2[i],'Process Unit Hazard Factor,F3':F3[i],'Fire & Explosion Index':F[i],'Distance of Exposure':De[i],'Damage Factor':DF[i],'Value of Exposure Area':Ve[i].varValue,'Base Maximum Property Damage Cost':Omega0[i].varValue}) 

if SwitchCEI ==1:      
    filename = 'CEI_Results.csv'
    with open(filename, 'w', newline='') as file:
        # Write value of all variables
          fieldnames = ['unit','liquid flowrate of release,Liq','total liquid release,WT','fraction flashed,Fv','airborne quantity produced by flashed,AQf','total mass of liquid entering pool,WP','pool area,AP','vapor pressure,Pvap','airborne quantity from pool surface,AQp','total airborne quantity,AQ','CEI','Dc','Value of chemical exposure']
          writer = csv.DictWriter(file, fieldnames=fieldnames)    
          writer.writeheader()
          for i in pertinent_units_Vlc:
        # writer.writerow({ 'unit':i,'liquid flowrate of release,Liq': Liq.get(i),'total liquid release,WT':WT.get(i),'fraction flashed,Fv':Fv.get(i),'airborne quantity produced by flashed,AQf':AQf.get(i),'vapor pressure,Pvap':Pvap.get(i),'total mass of liquid entering pool,WP':WP.get(i),'pool area,AP':AP.get(i),'airborne quantity from pool surface,AQp':AQp.get(i),'total airborne quantity,AQ':AQ.get(i),'CEI':CEI.get(i),'Dc':Dc.get(i),'Value of chemical exposure':Vlc.get(i)})
            writer.writerow({ 'unit':i,'liquid flowrate of release,Liq': Liq[i],'total liquid release,WT':WT[i],'fraction flashed,Fv':Fv[i],'airborne quantity produced by flashed,AQf':AQf[i],'vapor pressure,Pvap':Pvap[i],'total mass of liquid entering pool,WP':WP[i],'pool area,AP':AP[i],'airborne quantity from pool surface,AQp':AQp[i],'total airborne quantity,AQ':AQ[i],'CEI':CEI[i],'Dc':Dc[i],'Value of chemical exposure':Vlc[i][j]}) 

#--------------Plot Results--------------
xpos, ypos = [], []
for i in units:
    xpos.append(x[i].varValue)
    ypos.append(y[i].varValue)

# Plot invisible scatter
fig, ax = plt.subplots()
ax.scatter(xpos,ypos,alpha=0)
# Set bounds of axis
plt.axis('square')
scalebar = AnchoredSizeBar(ax.transData,50,sep=10,label="bar scale: 50 m",loc ='lower left',bbox_to_anchor=(0,0.8),bbox_transform=ax.transAxes,frameon=False,size_vertical=0.05)
ax.add_artist(scalebar)

ax.set_xlim(0,max(xpos)+50)
ax.set_ylim(0,max(ypos)+50)

# Place unit number at each scatter point
numbers = list(range(1,len(xpos)+1))
for i,txt in enumerate(numbers):
       ax.annotate(txt, (xpos[i]-5,ypos[i]-5))
# Value of objective function as title of plot
sepcost = value(layout.objective) 
sepcost1 = "{0:,.0f}".format(sepcost)  
ax.set_title('Objective Function (GBP/yr) = ' + str(sepcost1),loc='center', fontsize = 11)
ax.set_xlabel('Distance(m)', fontsize = 11)
ax.set_ylabel('Distance(m)', fontsize = 11)
# Build rectangles with respective unit sizes around units
# Create dictionary for coordinates of bottom left of rectangle of each unit
xrectdict = dict.fromkeys(units)
yrectdict = dict.fromkeys(units)
# Subtract half the dimension based on orientation and store into dictionary
for i in units:
    xrectdict[i] = x[i].varValue - l[i].varValue/2
    yrectdict[i] = y[i].varValue - d[i].varValue/2
# Convert dictionary to array
xrect, yrect = [],[]
for i in units:
    xrect.append(xrectdict[i])
    yrect.append(yrectdict[i])
# Extract length and depth data of units
length,depth = [],[]
for i in units:
    length.append(l[i].varValue)
    depth.append(d[i].varValue)
# Plot rectangles
for i in range(len(numbers)):
    label_1 = i+1, units[i]
    rect = mpatch.Rectangle((xrect[i], yrect[i]), length[i], depth[i], fill=None, edgecolor="black",label = label_1)
    ax.add_patch(rect)
    leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True, bbox_to_anchor=(1.04,1), loc="upper left", fontsize = 9)
    for item in leg.legendHandles:
            item.set_visible(False)
            
plt.savefig('pip3.jpg', format='jpg', dpi=1200, bbox_inches='tight')


