'''
Traceotools testfile

Runs TRACEO several times under slightly different configurations to showcase
all plotting and auxiliary functions.
Also contains code for other bathymetry, sound speed and object examples.

CT Pedro Mendes Diniz
Instituto de Estudos do Mar Almirante Paulo Moreira
Arraial do Cabo, 28/11/2022

'''

import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from traceotools import *

### CHANGE TO CORRECT PATH ###
traceopath = 'C:/dummy_path/traceo.exe'

case_title = 'Test file'
fname = 'test'

runtype = 'PAV'
    
#==================================================================
# 
# Define source data:
#
#==================================================================

freq =  300 # Central frequency [Hz]
Rmax = 100000 # Max range for bounding box [m]
Dmax =  5000 # Max depth [m]

zs = 1300 # Depth of source [m]
rs = 0 # Horizontal position of source [m]
thetamax = 17 # Maximum ray launching angle (0 is horizontal)
nthetas = 35 # Number of rays
thetas = np.linspace(-thetamax,thetamax,nthetas) # Launching angles

ray_step = 10 # Step of integration 
             # If set to 0, model uses Rmax to calculate preliminary step

rbox = [-1,Rmax+1] # Horizontal limits of bounding box.

source_data = {}
source_data['ds']       = ray_step
source_data['position'] = [rs,zs]
source_data['rbox']     = rbox
source_data['f']        = freq
source_data['thetas']   = thetas

#==================================================================
#
# Define altimetry data:
# 
#==================================================================

altr = [-2, Rmax+2] # Limits of surface. Must lie outside of rbox for stability.
altimetry = np.array([altr,[0,0]])

surface_data = {}
surface_data['type']        = "V" # Vacuum
surface_data['ptype']       = "H" # Homogeneous
surface_data['units']       = "W" # Attenuation unit: dB/wavelenght
surface_data['itype']       = "2P" # Interpolation type: linear
surface_data['x']           = altimetry # Surface coordinates
surface_data['properties']  = [0.0,0.0,0.0,0.0,0.0]

#==================================================================
#
# Define sound speed data:
#
#==================================================================
# # Example with linear profile
# c0 = 1550
# c1 = 1530

# ssp_data = {}
# ssp_data['cdist']   = "c(z,z)" # Sound speed profile
# ssp_data['cclass']  = "LINP" # Profile class: linear
# ssp_data['z']       = np.array([0, Dmax])
# ssp_data['r']       = []
# ssp_data['c']       = np.array([c0, c1]) 

# Example with munk profile:
depths = np.linspace(0,Dmax,1000)
c = munk(depths)

ssp_data = {}
ssp_data['cdist']     = "c(z,z)" # Sound speed profile
ssp_data['cclass']    = "TABL" # Profile class: tabulated
ssp_data['z']         = depths
ssp_data['r']         = []
ssp_data['c']         = c

# # Example with munk field:
# npts = 500
# ranges = np.linspace(0,Rmax,npts)
# depths = np.linspace(0,Dmax,npts)
# C = np.zeros([npts,npts])
# z1 = np.linspace(1000,2000,npts)
# for i in range(npts):
#     c = munk(depths,z1[i])
#     C[:,i] = c

# ssp_data = {}
# ssp_data['cdist']     = "c(r,z)" # Sound speed field
# ssp_data['cclass']    = "TABL" # Profile class: tabulated
# ssp_data['z']         = depths
# ssp_data['r']         = ranges
# ssp_data['c']         = C

#==================================================================
#  
#  Define object data:
#  
#==================================================================

# No objects
object_data = {}
object_data['nobjects'] = 0

# # Example with 2 circular objects:
# nobj = 2 # Number of objects
# npo = 50 # Number of points defining object
# cobjp = 2000 # p-wave speed
# cobjs = 200 # s-wave speed
# rhoo = 5 # Density
# alphao = 0 # Attenuation
# R0 = 500
# factor = 1.5
# ro = [80000,30000]
# zo = [2000,4000]

# robj1  =  factor*R0*np.cos(np.linspace(np.pi,0,npo)) + ro[0]
# robj2  =  factor*R0*np.cos(np.linspace(np.pi,0,npo)) + ro[1]
# zup    =         R0*np.sin(np.linspace(np.pi,0,npo))
# zdn    =        -R0*np.sin(np.linspace(np.pi,0,npo)) 
# zdn1   =  zdn + zo[0]
# zdn2   =  zdn + zo[1]
# zup1   =  zup + zo[0]
# zup2   =  zup + zo[1]

# xboj1 = np.array([robj1,zdn1,zup1])
# xboj2 = np.array([robj2,zdn2,zup2])

# object_data = {}
# object_data['nobjects']   = nobj # 2 objects
# object_data['npobjects']  = [npo, npo] # Number of points in each object
# object_data['type']       = ["R", "R"] # Type: rigid
# object_data['units']      = ["W", "W"]# Attenuation unit: dB/wavelenght
# object_data['itype']      = "2P" # Objects interpolation type: linear
# object_data['x']          = [xboj1, xboj2]
# object_data['properties'] = [cobjp, cobjs, rhoo, alphao, alphao], [cobjp, cobjs, rhoo, alphao, alphao]

#==================================================================
# 
# Define bathymetry data:
#
#==================================================================

# Example with linear bathymetry
batr = np.array([-2, Rmax+2]) # Horizontal limits must lie outside of rbox
batz = np.array([Dmax, 4*Dmax/5])
bathymetry = np.array([batr, batz])

bottom_data = {}
bottom_data['type']         = "E" # Elastic bottom
bottom_data['ptype']        = "H" # Homogeneous bottom
bottom_data['units']        = "W" # Attenuation unit: dB/wavelenght
bottom_data['itype']        = "2P" # Bottom interpolation type: linear
bottom_data['x']            = bathymetry # Bottom coordinates 
bottom_data['properties']   = [1800.0,0.0,2,0.6,0.0] # p-speed, s-speed, density, p-attenuation, s-attenuation

# # Example with sea mount:
# batr = [-2,60000,60010,79990,80000,Rmax+2] # Horizontal limits must lie outside of rbox
# batz = [Dmax, Dmax, 3000, 3000, Dmax,Dmax]
# bathymetry = np.array([batr, batz])

# bottom_data = {}
# bottom_data['type']       = "E" # Elastic bottom
# bottom_data['ptype']      = "H" # Homogeneous bottom
# bottom_data['units']      = "W" # Attenuation unit: dB/wavelenght
# bottom_data['itype']      = "2P" # Bottom interpolation type: linear
# bottom_data['x']          = bathymetry # Bottom coordinates 
# bottom_data['properties'] = [2000.0, 0.0, 2.0, 0.5, 0.0] # p-speed, s-speed, density, p-attenuation, s-attenuation

#==================================================================
#
# Define output data:
#
#==================================================================

nra = 501 # Number of arrays distributed along range
rarray = np.linspace(0,Rmax,nra) # Ranges of arrays
nza = 501  # Number of arrays distributed along depth
zarray = np.linspace(0,Dmax,nza) # Depths of arrays

output_data={}
output_data['ctype']       = runtype # Run type
output_data['array_shape'] = "RRY" # Geometry of array: rectangular
output_data['r']           = rarray
output_data['z']           = zarray
output_data['miss']        = 1 # Proximity limit for eigenray and arrivals calculation

#%%
#==================================================================
#  
# RUNNING THE MODEL:
#  
#==================================================================
# Run showcasing coherent pressure, TL and particle velocity plotting

wtraceoinfil(fname,case_title,source_data,surface_data,ssp_data,object_data,bottom_data,output_data)
runtraceo(traceopath,fname)

graph = plotssp(fname)
graph = plotcpr(fname) # CPR CTL PAV
graph = plottlr(fname,[1000,80000]) # CPR CTL PAV
graph = plottlz(fname,[zs,4000]) # CPR CTL PAV
graph = plotpvl(fname) # PVL PAV

#%%
# Runs showcasing ray tracing, eigenrays and environment plotting

runtype = 'ARI'
output_data['ctype']       = runtype # Run type
wtraceoinfil(fname,case_title,source_data,surface_data,ssp_data,object_data,bottom_data,output_data)
runtraceo(traceopath,fname)

graph = plotray(fname) # RCO ARI ERF EPR

runtype = 'ERF'
nthetas = 101 # Number of rays
thetas = np.linspace(-thetamax,thetamax,nthetas) # Launching angles
source_data['thetas']   = thetas
output_data['ctype']       = runtype # Run type
rarray = np.array([Rmax]) # Ranges of arrays
zarray = np.array([1000,1300,2000]) # Depths of arrays
output_data['array_shape'] = "VRY" # Geometry of array: vertical
output_data['r']           = rarray
output_data['z']           = zarray
wtraceoinfil(fname,case_title,source_data,surface_data,ssp_data,object_data,bottom_data,output_data)
runtraceo(traceopath,fname)

graph = plotenv(fname,ssp=True)
graph = plotray(fname) # RCO ARI ERF EPR

#%%
# Run showcasing amplitudes and delays and simulated transmission

runtype = 'ADR'
output_data['ctype']       = runtype # Run type
wtraceoinfil(fname,case_title,source_data,surface_data,ssp_data,object_data,bottom_data,output_data)
runtraceo(traceopath,fname)

graph = plotaad(fname) # ADR ADP

# TRANSMITTING
# Creates a chirp and transmits using an amplitudes and delays output file
# with added noise before plotting.

if runtype in ['ADR','ADP']:
    Fs = 2000
    t = np.linspace(0,1,Fs+1)
    data = 1e5*chirp(t,200,1,400)
    P = transmit(data,Fs)
    Pn,noise,SNR = add_noise(P,NLdb=-10)
    
    for i in range(len(zarray)):
        fig, ax = plt.subplots()
        c = ax.specgram(Pn[i],Fs=Fs,cmap='jet')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Receiver ' + str(i+1))
        ax.set_ylim(0,Fs/2)
        fig.colorbar(c[3])
        fig.tight_layout()

#%%
# Modifying figure after creation

runtype = 'CPR'
nthetas = 35 # Number of rays
thetas = np.linspace(-thetamax,thetamax,nthetas) # Launching angles
source_data['thetas']   = thetas
output_data['ctype']       = runtype # Run type
nra = 501 # Number of arrays distributed along range
rarray = np.linspace(0,Rmax,nra) # Ranges of arrays
nza = 501  # Number of arrays distributed along depth
zarray = np.linspace(0,Dmax,nza) # Depths of arrays
output_data['array_shape'] = "RRY" # Geometry of array: rectangular
output_data['r']           = rarray
output_data['z']           = zarray
wtraceoinfil(fname,case_title,source_data,surface_data,ssp_data,object_data,bottom_data,output_data)
runtraceo(traceopath,fname)

graph = plotcpr(fname)
print(graph.__dict__) # Shows objects contained in graph
# Editing colormap
graph.image.set_cmap('jet_r')
graph.image.set_clim(60,110)
# Removing original colorbar and creating a new one
graph.colorbar.remove()
graph.colorbar = plt.colorbar(graph.image,orientation='horizontal',extend='both',
                              label='TL (dB)')
graph.colorbar.ax.invert_yaxis()
# Removing bottom
graph.sediment.set_facecolor('white')
graph.sediment.set_edgecolor('white')
graph.bathy.set_visible(False)
# Editing axes
graph.axes.set_title('Test file CPR (modified)')
graph.axes.set_xticks(np.linspace(0,Rmax,6))
graph.axes.set_xticklabels(np.linspace(0,Rmax,6))
graph.axes.set_xlabel('Range (m)')
graph.axes.scatter([Rmax,Rmax,Rmax],[1000,1300,2000],marker='d',color='r',s=50,
                   label = 'Receivers')
graph.axes.legend()
# Changing figure size and updating plot
graph.fig.set_size_inches(8,4)
plt.show()

#%% PLOTTING FUNCTIONS
# graph = plotssp(fname)
# graph = plotenv(fname,ssp=True)
# graph = plotray(fname) # RCO ARI ERF EPR
# graph = plotcpr(fname) # CPR CTL PAV
# graph = plottlr(fname,[1000,80000]) # CPR CTL PAV
# graph = plottlz(fname,[zs,4000]) # CPR CTL PAV
# graph = plotpvl(fname) # PVL PAV
# graph = plotaad(fname) # ADR ADP