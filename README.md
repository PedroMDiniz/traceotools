# TRACEOTOOLS

Python plotting package and auxiliary functions for TRACEO ray tracing model.

Pedro Mendes Diniz  
Instituto de Estudos do Mar Almirante Paulo Moreira  
Arraial do Cabo, 28/11/2022

## Overview

Plotting and auxiliary functions to be used alongside TRACEO ray tracing model (available at http://www.siplab.fct.ualg.pt/models.shtml). Code developed based on Acoustics Toolbox MATLAB plotting routines by Michael Porter (http://oalib.hlsresearch.com/AcousticsToolbox/) and TRACEO Python test cases package by Orlando Camargo Rodríguez (http://www.siplab.fct.ualg.pt/python.shtml). Input file writing function, by Simone Pacheco and Orlando Camargo Rodríguez, is as provided in that package.

TRACEO ray tracing model developed by:

Orlando Camargo Rodríguez  
Universidade do Algarve  
Physics Department  
Signal Processing Laboratory  
Copyright (C) 2010 

## Plotting functions

Includes functions for plotting data from every output file from TRACEO. Functions return an instance of a TraceoPlot class object containing figure, axes and plot elements (such as source marker, bottom line and colorbar). These objects can then be interacted with as regular Matplotlib objects. For example, source marker (and other elements) can be easily turned off with:

```python
TraceoPlot.source.set_visible(False)
```
 Contained objects depend on the plotting function called and a list of them can be accessed by:

```python
TraceoPlot.__dict__
```

These functions read information from the TRACEO input file (.in), specified by fname.

### plotssp(fname)
Plots either sound speed profile or sound speed field, depending on input file configuration.

### plotenv(fname,ssp=True)
Plots environment elements (surface, bottom, objects, source and receivers).
The ssp flag defines if it will also plot either the sound speed profile in a separate subplot or the sound speed field as a colormesh overlay over the main plot, depending on
the input file SSP configuration.

### plotray(fname,color_default='k',colorRSR='g',colorRBR='b',colorSRBR='r')
Ray trace plot. Used for RCO (ray coordinates), EPR, ERF (eigenrays) or ARI (all ray information) runs. Takes optional color arguments used to color rays depending on their paths on ARI runs.

### plotcpr(fname,cmap='viridis_r')
Plots a colormesh of coherent pressure transmission loss in dB scale with the environment overlaid. Can be used with CPR (coherent pressure), CTL (transmission loss) and PAV (pressure and particle velocity) runs. Colormap default range is estimated statistically. Takes an optional argument for which colormap to use.

### plottlz(fname,z)
Plots transmission loss section along a desired depth or list of depths. Works with CPR, CTL or PAV runs. Interpolates from a 2D TL field, so receiver geometry must be set as "RRY".

### plottlr(fname,r)
Plots transmission loss section along a desired range or list of ranges. Works with CPR, CTL or PAV runs. Interpolates from a 2D TL field, so receiver geometry must be set as "RRY".

### plotpvl(fname,cmap='viridis_r')
Plots coherent transmission loss for horizontal and vertical components of particle velocity from PVL (particle velocity) or PAV runs as colormeshes on separate axes. Objects contained within TraceoPlot instance are duplicated, one for each subplot, and organized in lists. Colormap default range is estimated statistically. Takes an optional argument for which colormap to use.

### plotaad(fname,which=[]):
Plots eigenrays arrivals from ADR or ADP (amplitudes and delays) runs in a ray amplitude x travel time graph. Series of arrivals are grouped by receiver. Accepts as parameter a list of which receivers' results should be plotted. Defaults to all receivers, but the graph can become too crowded if there are too many. Not a true stem plot, markers and stems are plotted individually and can be configured independently.

## Auxiliary functions

### runtraceo(path,fname)
Checks operating system (Windows or Linux) and runs TRACEO on the specified path with the input file specified by fname. Manually copies and deletes LOGFIL and WAVFIL files during execution with system-specific commands.
If a previous TRACEO output (.mat) of the same type with the default name exists, deletes it to avoid file corruption on overwriting.

### munk(z,z1=1300,c1=1500)
Generates a sound speed profile as specified by Munk (1974) along the list of depths specified by z. z1 and c1 are depth and sound speed of sound channel and default to the canonical values.

### transmit(data,Fs,filename='aad.mat')
Simulates a signal transmission along a waveguide by using a modeled impulse response given by a TRACEO amplitudes and delays run (ADR or ADP). Reads aad.mat file, constructs a time domain impulse response for each of the receivers and convolves it with the signal given by data with sampling frequency Fs. Analytic signal is used because impulse response is complex, otherwise the imaginary part will not be convolved, which discards phase changes. Only the real part of the received signal is returned.  
Outputs a NxM array where N is the number of receivers and M is the number of samples for the maximum duration resulting from a convolution, such as that all received signals have the same length. If no eigenrays reach a receiver, its row will be filled with zeros.

### add_noise(signal,SNRdb=None,NLdb=None,seed=None)
Adds white Gaussian noise to a signal from either a desired SNR (dB) or noise level (db). If signal is multichannel, a different realization of noise is added to each channel. In case SNRdb is provided, it will be treated as the average SNR across channels in such a way that the noise added to each channel all have the same level.  
If signal is complex, noise power is split between real and imaginary parts, each with half variance (or standard deviation scaled by sqrt(2)).  
Seed for random noise array can be specified.
Returns the noisy signal array, pure noise array added and the average SNR.

## Test file
Also included is a test file that configures and calls TRACEO multiple times for various run types and calls the plotting functions accordingly. Also uses the auxiliary functions to generate the sound speed profile and simulate a transmission with added noise. If all goes well, 13 figures should be produced.  
The only necessary change is the complete path for traceo.exe on the first line of the script after imports.

## License notes

TRACEOTOOLS is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). This is a human-readable summary of (and not a substitute for) the license. Please read the attached "license.txt" file for the full text of the license.  

You are free to:

Share — copy and redistribute the material in any medium or format.  
Adapt — remix, transform, and build upon the material.

The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

NonCommercial — You may not use the material for commercial purposes.

ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
