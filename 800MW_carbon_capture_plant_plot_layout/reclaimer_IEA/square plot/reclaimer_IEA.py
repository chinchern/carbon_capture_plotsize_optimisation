
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

# --------------Switches--------------
# CPLEX free edition only supports up to 1000 variables. For larger land sizes, use CBC or increase coarseness (g)
# 1 = CPLEX, 2 = CBC
solver = 1

# Toggle constraints in layout problem (1 is on; 0 is off)
# Land Use Constraints
SwitchLandUse = 1
# Square Plot (else: Rectangular Plot)
SwitchSquarePlot = 0
# Pumping Cost Calculation
SwitchPumpingCost =1
# Safety Distances
SwitchSafetyDistances = 1

# --------------Define Sets--------------
# Define the process units
units = ['reclaimer_1','reclaimer_2']
Nunits = len(units)

if SwitchSafetyDistances == 1:
    # type of equipment      
   heat_exchanger =['reclaimer_1','reclaimer_2']

# --------------Define Parameters and Values--------------
# Base layout model

# M (m) is used in the contraints which set A or B, L or R, and Din or Dout.
# It should be big enough not to constrain the size of the plot, but not too big
M = 1e3

# Dimensions of each unit (m)
alpha = dict.fromkeys(units)
beta = dict.fromkeys(units)
alpha['reclaimer_1'] = 8
alpha['reclaimer_2'] =8

beta['reclaimer_1'] = 10
beta['reclaimer_2'] = 10


#Annualised Factor
annualised_factor = 0.1102
#Cost of Electricity in 2021
electricity_cost = 4.5*1e-5  #GBP/Wh
#Operating hours
operating_hour = 8000   #hrs

# Piping costs (GBP/m.yr)
C = np.zeros((len(units), len(units)))  
# assign values where flowsheet connection

# Cost data is made into a dictionary
C = makeDict([units,units],C,0)
    
if SwitchPumpingCost == 1:   
    # Cost of Pumping Calculation
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
   
    #Assign values of fluid velocity where flowsheet connection


    #Assign values of density where flowsheet connection

    #Assign values of pipe diameter where flowsheet connection

    #Assign values of friction factor where flowsheet connection

    # Flow data is made into a dictionary
    mass_flowrate = makeDict([units,units],mass_flowrate,0)
    fluid_velocity = makeDict([units,units],fluid_velocity,0)
    fluid_density = makeDict([units,units],fluid_density,0)
    DIA = makeDict([units,units],DIA,0)
    friction_factor = makeDict([units,units],friction_factor,0)
          
# literature safety distances(Industrial Risk Insurers)
D_min = np.zeros((len(units),len(units)))
D_min = makeDict([units,units],D_min,0)

if SwitchSafetyDistances == 1:
    
    for i in heat_exchanger:
        for j in heat_exchanger:
            D_min[i][j]= 2
            D_min[j][i]=D_min[i][j]
else:
    for i in units:
        for j in units:
            D_min[i][j]= 0

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
    
if SwitchPumpingCost == 1:
    pump_efficiency = 0.6
    
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
if SwitchPumpingCost == 1 and SwitchLandUse == 1:
    layout += SumCD + TLC + sum_pumping_cost
elif SwitchPumpingCost == 0 and SwitchLandUse == 1:
    layout += SumCD + TLC   
else:
    layout += SumCD

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

for i in units:
    for j in units:
            layout += CD[i][j] == C[i][j]*D[i][j]            

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
            layout += pumping_cost[i][j] ==  electricity_cost*operating_hour*pumping_power[i][j]  
            #Objective fucntion contribution from cost of pumping for pipe connecting unit i and j
    layout +=sum_pumping_cost ==  lpSum([pumping_cost[i][j] for i in units for j in units])
            
# --------------Fixing Variable Values--------------
# Define function to fix value
# def fix_variable(variable, value):
#       variable.setInitialValue(value)
#       variable.fixValue()
      
# fix_variable(x['stripper_T1'],4)
# fix_variable(x['reboiler_T1_1'],12.2)
# fix_variable(x['reboiler_T1_2'],16.6)

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
print("Total cost of piping =", SumCD.varValue, "GBP/yr")

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
        
if SwitchPumpingCost == 1:
    print("Total cost of pumping =", sum_pumping_cost.varValue, "GBP/yr")
    
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
scalebar = AnchoredSizeBar(ax.transData,2,sep=10,label="bar scale: 2 m",loc ='lower left',bbox_to_anchor=(1.1,0),bbox_transform=ax.transAxes,frameon=False,size_vertical=0.05)
ax.add_artist(scalebar)

# plt.axis('Rectangle')
ax.set_xlim(0,max(xpos)+10)
ax.set_ylim(0,max(ypos)+10)

# Place unit number at each scatter point
numbers = list(range(1,len(xpos)+1))
for i,txt in enumerate(numbers):
       ax.annotate(txt, (xpos[i]-.5,ypos[i]-.5))
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
    leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True, bbox_to_anchor=(1.04,1), loc="upper left", fontsize = 8)
    for item in leg.legendHandles:
            item.set_visible(False)
            
plt.savefig('pip3.jpg', format='jpg', dpi=1200, bbox_inches='tight')
