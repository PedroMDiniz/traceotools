'''
TRACEOTOOLS

Python plotting package for TRACEO ray tracing model.

Plotting and auxiliary functions to be used alongside TRACEO ray tracing model
(available at http://www.siplab.fct.ualg.pt/models.shtml).
Code developed based on Acoustics Toolbox MATLAB plotting routines by Michael
Porter (http://oalib.hlsresearch.com/AcousticsToolbox/) and TRACEO Python test
cases package by Orlando Camargo Rodríguez (http://www.siplab.fct.ualg.pt/python.shtml).
Input file writing function, by Simone Pacheco and Orlando Camargo Rodriguez,
is as provided in that package.

Includes functions for plotting data from every output file from TRACEO.
These functions read information from the TRACEO input file (.in).
Also includes auxiliary functions for simulating signal transmission.

CT Pedro Mendes Diniz  
Instituto de Estudos do Mar Almirante Paulo Moreira  
Arraial do Cabo, 28/11/2022

WARNING: This version of TraceoTools is for the legacy (2015) version of Traceo.
If using the current version of Traceo, runtraceo() won't work.
Download the appropriate TraceoTools version (main branch @ https://github.com/PedroMDiniz/traceotools)

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import loadmat
from itertools import product
from scipy.interpolate import interp2d
from scipy.signal import hilbert, fftconvolve
from os import system
from os.path import exists, join
from platform import system as platsys

### AUXILIARY FUNCTIONS ###

def runtraceo(path,fname):
    '''
    Calls traceo.exe with the specified input file.
    Checks operating system and uses commands for either Linux or Windows.
    If a previous TRACEO output (.mat) of the same type with the default name
    exists, deletes it to avoid file corruption on overwriting.

    Parameters
    ----------
    path : path of traceo.exe.
          type: string
    fname : name of the .in file (without extension).
           type: string
    
    '''

    infile, _ = _setup(fname)
    if infile[-2][1:4] in ['ADR','ADP']:
        prev_file = 'aad.mat'
    elif infile[-2][1:4] in ['EPR','ERF']:
        prev_file = 'eig.mat'
    else:
        prev_file = '{}.mat'.format(infile[-2][1:4].lower())
        
    if platsys() == 'Linux':
        if exists(prev_file):
            system('rm {}'.format(prev_file))
        system('cp {:s} WAVFIL'.format(fname + '.in'))
        system('{}'.format(join(path,'traceo.exe')))
        system('cp LOGFIL {:s}'.format(fname + '.log'))
        system('rm WAVFIL')
        system('rm LOGFIL')
    else:
        if exists(prev_file):
            system('del {}'.format(prev_file))
        system('copy {:s} WAVFIL'.format(fname + '.in'))
        system('{}'.format(join(path,'traceo.exe')))
        system('copy LOGFIL {:s}'.format(fname + '.log'))
        system('del WAVFIL')
        system('del LOGFIL')

def munk(z,z1=1300,c1=1500):
    '''
    Generates Munk sound speed profile as defined by Munk (1974).
    
    Parameters
    ----------
    z : depths at which sound speed is to be calculated (m)
        type: array
    z1 : depth of deep sound channel (m). Optional, defaults to 1300 m
        type: float
    c1 : sound speed at deep sound channel (m/s). Optional, defaults to 1500 m/s
        type: float
    
    Returns
    -------
    c : sound speed profile (m/s)
       type: numpy array
    '''
    c = np.ones(len(z))
    eps = 7.37e-3
    eta = 2 * (z - z1)/1300
    c = c1 * (1 + eps*(eta + np.exp(-eta) - 1))
    
    return c

def transmit(data,Fs,filename='aad.mat'):
    '''
    Simulates a signal transmission along a waveguide by using a modeled impulse
    response given by a TRACEO amplitudes and delays run (ADR or ADP).
    Reads aad.mat file, constructs a time domain impulse response for each of
    the receivers and convolves it with the signal. Analytic signal is used
    because impulse response is complex, otherwise the imaginary part will not
    be convolved, which discards phase changes. Only the real part of the received
    signal is returned.
    Outputs a NxM array where N is the number of receivers and M is the number
    of samples for the maximum duration resulting from a convolution. If no
    eigenrays reach a receiver, its row will be filled with zeros.
    
    Parameters
    ----------
    data : original signal array (source time series).
          type: numpy 1D array
    Fs : sampling frequency of signal.
        type: float
    filename : filename (with path) for TRACEO ADR or ADP output.
              Defaults to 'aad.mat'.
             type: string
    
    Returns
    -------
    received_signal : convolved signal for all chanels.
                     type: numpy array
      
    '''
    # Constructing received signal based on model results
    # Getting receiver coordinates from .mat file
    ampdel = loadmat(filename)
    rrecs = list(zip(*ampdel['rarray2d']))
    zrecs = list(zip(*ampdel['zarray2d']))
    recs = list(product(rrecs,zrecs))
    nrays = int(ampdel['neigrays'][0,0])
    if nrays == 0:
        print('No eigenrays found.')
        return
    name = 'aad00000'
    
    # Creating a list of received signals because they initially have different lengths
    signals = []
    for rec in recs:
        delay = []
        amplitude = []
        for i in range(1,nrays+1):
            ri = str(i)
            aad = ampdel[name[0:-len(ri)] + ri]
            if tuple((aad[0,0],aad[0,1])) == rec: # Grouping by receiver
                delay.append(aad[0,2])
                amplitude.append((aad[0,3]+1j*aad[0,4])*np.exp(1j*aad[0,5]))
        
        # Constructing time domain impulse responses
        try:
            h = np.zeros(int(max(delay)*Fs)+len(data),dtype=complex)
            for k,j in enumerate(delay):
                index = int(j*Fs)
                h[index] = amplitude[k]
        except ValueError:
            h = np.array([0])
        signals.append(fftconvolve(hilbert(data),h))

    # Converting list of arrays into a matrix where each line corresponds to a receiver
    maxlen = 0
    for s in signals: 
        if len(s) > maxlen: maxlen = len(s)
    received_signal = np.zeros([len(signals),maxlen])
    for i,s in enumerate(signals):
        received_signal[i,0:len(s)] = s.real
    
    return received_signal

def add_noise(signal,SNRdb=None,NLdb=None,seed=None):
    '''
    Adds white Gaussian noise to a signal from a desired SNR (dB)
    or noise level (dB). If signal is multichanel, a different realization of
    noise is added to each chanel. In case SNRdb is provided, it will be treated
    as the average SNR across channels in such a way that the noise added to
    each channel all have the same level.
    If signal is complex, noise power is split between real and imaginary parts,
    each with half variance (or standard deviation scaled by sqrt(2)).
    Seed for random noise array can be specified.
    Returns the noisy signal array, pure noise array added and the average SNR.
    
    Parameters
    ----------
    signal : original signal array. 
            type : numpy array
    SNRdb : desired signal-to-noise ratio (dB). Provide either this or NLdb.
           type: float
    NLdb : desired noise level (dB). Provide either this or SNRdb.
          type: float
    seed : seed for noise reproducibility. Optional.
          type: int
    
    Returns
    -------
    signal_noise : noisy signal array. 
                  type : numpy array
    noise : pure noise array.
           type: numpy array
    SNR_estimate : average SNR estimated from input signal and noise variances.
                  type: float
                  
    '''
    signal_average_power = np.var(signal)
    signal_average_power_db = 10*np.log10(signal_average_power)    
    
    if SNRdb == None and NLdb == None:
        raise TypeError('Either SNR (dB) or noise level required.')
    elif SNRdb != None and NLdb != None:
        raise TypeError('Only one of either SNR (dB) or noise variance can be used.')
    elif SNRdb != None:
        noise_desired_power_db = signal_average_power_db - SNRdb
        noise_desired_power = 10**(noise_desired_power_db/10)
    elif NLdb != None:
        noise_desired_power_db = NLdb
        noise_desired_power = 10**(noise_desired_power_db/10)

    np.random.seed(seed)
    if np.iscomplex(signal).any():
        sigma = np.sqrt(noise_desired_power)/np.sqrt(2)
        noise = np.random.normal(loc=0,scale=sigma,size=(signal.size * 2)).view(np.complex128)
    else:
        sigma = np.sqrt(noise_desired_power)
        noise = np.random.normal(loc=0,scale=sigma,size=(signal.size))
        
    noise = noise.reshape(signal.shape)
    noise_average_power_db = 10*np.log10(np.var(noise))   
    SNR_estimate = signal_average_power_db - noise_average_power_db    
    signal_noise = signal + noise
    return signal_noise, noise, SNR_estimate

### PRIVATE FUNCTIONS ###
# To be used only inside module

def _setup(fname):
    # Loading parameters from .in file
    file = open(fname + '.in','r')
    infile = file.readlines()
    file.close()
    sep = []
    for i, j in enumerate(infile): # gets index of separators in .in file
        if j == '--------------------------------------------------------------------------------\n':
            sep.append(i)
    return infile, sep

def _get_extent(infile,sep):

    rarray = np.array(infile[sep[-1]-2].split(' '))[0:-1].astype(np.float64)
    r1 = rarray[0]
    r2 = rarray[-1]
    zarray = np.array(infile[sep[-1]-1].split(' '))[0:-1].astype(np.float64)
    z1 = zarray[0]
    z2 = zarray[-1]
    return [r1,r2,z1,z2]
    
def _axes_config(axes,title,zmax=None,rbox=None):
    if zmax is not None:
        axes.set_ylim(0, zmax)
    if rbox is not None:
        xticks = axes.get_xticks()
        axes.set_xticks(xticks)
        axes.set_xticklabels(xticks/1000)
        axes.set_xlim(0,rbox[1])
        axes.set_xlabel('Range (km)')
    axes.set_ylabel('Depth (m)')
    axes.set_title(title)
    axes.tick_params(labelsize=10)
    axes.invert_yaxis()

def _fig_config(fig):
    fig.set_size_inches(8,6)

### TraceoPlot CLASS ###

class TraceoPlot:
    
    def __init__(self,naxes=1):
        fig, axes = plt.subplots(naxes,1,sharex=True)
        self.fig = fig
        self.axes = axes
        
    def _set_image(self,img):
        self.image = img
        
    def _set_colorbar(self,cbar):
        self.colorbar = cbar
        
    def _add_surface(self,infile,sep):
        alti = infile[sep[1]+7:sep[2]]
        for i in range(len(alti)):
            alti[i] = np.array(alti[i].split(' ')).astype(np.float64)
        alti = np.array(alti)
        
        if np.ndim(self.axes) == 0:
            surf = self.axes.plot(alti[:,0],alti[:,1],'k')
            self.axes.fill_between(alti[:,0],np.zeros(len(alti)),alti[:,1],color='white')
            self.surface = surf[0]
        else:
            self.surface = []
            for ax in self.axes:
                surf = ax.plot(alti[:,0],alti[:,1],'k')
                ax.fill_between(alti[:,0],np.zeros(len(alti)),alti[:,1],color='white')
                self.surface.append(surf[0])
                
        
    def _add_bottom(self,infile,sep,tlsection=False):
        bathy = infile[sep[4]+7:sep[5]]
        for i in range(len(bathy)):
            bathy[i] = np.array(bathy[i].split(' ')).astype(np.float64)
        bathy = np.array(bathy) 
        zmax = bathy[:,1].max()
        self.maxdepth = zmax
        if tlsection == True:
            return
        
        if np.ndim(self.axes) == 0:
            bat = self.axes.plot(bathy[:,0],bathy[:,1],'k')
            sediment = self.axes.fill_between(bathy[:,0],(zmax*2)*np.ones(len(bathy)),bathy[:,1],color='sienna')
            self.bathy = bat[0]
            self.sediment = sediment
        else:
            self.bathy = []
            self.sediment = []
            for ax in self.axes:
                bat = ax.plot(bathy[:,0],bathy[:,1],'k')
                sediment = ax.fill_between(bathy[:,0],(zmax*2)*np.ones(len(bathy)),bathy[:,1],color='sienna')
                self.bathy.append(bat[0])
                self.sediment.append(sediment)
        
    def _add_objects(self,infile,sep):
        nobj = int(infile[sep[3]+1])
        if nobj > 0:
            self.objects = []
            npobj = int(infile[sep[3]+5])
            
            if np.ndim(self.axes) == 0:
                for i in range(nobj):
                    obj_contour = infile[sep[3]+2+(i+1)*5+i*npobj-int(bool(i)):sep[3]+2+(i+1)*5+i*npobj+npobj-int(bool(i))]
                    for j in range(len(obj_contour)):
                        obj_contour[j] = np.array(obj_contour[j].split(' ')).astype(np.float64)
                    obj_contour = np.array(obj_contour)
                    obj = self.axes.fill_between(obj_contour[:,0],obj_contour[:,1],obj_contour[:,2],color='k')
                    self.objects.append(obj)
            else:
                for ax in self.axes:
                    obj = []
                    for i in range(nobj):
                        obj_contour = infile[sep[3]+2+(i+1)*5+i*npobj-int(bool(i)):sep[3]+2+(i+1)*5+i*npobj+npobj-int(bool(i))]
                        for j in range(len(obj_contour)):
                            obj_contour[j] = np.array(obj_contour[j].split(' ')).astype(np.float64)
                        obj_contour = np.array(obj_contour)
                        obj.append(ax.axes.fill_between(obj_contour[:,0],obj_contour[:,1],obj_contour[:,2],color='k'))
                    self.objects.append(obj)
    
    def _add_source(self,infile):
        src = np.array(infile[3].split(' ')).astype(np.float64)
        if np.ndim(self.axes) == 0:
            self.source = self.axes.scatter(*src,s=50,c='m',zorder=10,label='Source')
        else:
            self.source = []
            for ax in self.axes:
                self.source.append(ax.scatter(*src,s=50,c='m',zorder=10,label='Source'))
                
    def _add_receivers(self,infile,sep):
        rarray = np.array(infile[sep[-1]-2].split(' '))[0:-1].astype(np.float64)
        zarray = np.array(infile[sep[-1]-1].split(' '))[0:-1].astype(np.float64)
        self.receivers = self.axes.scatter(*zip(*product(rarray,zarray)),s=50,c='g',marker='d',zorder=9,label='Receiver')
    
    def _create_rays_list(self):
        self.rays = []

    def _add_ray(self,ray):
        self.rays.append(ray)
        
    def _create_aad_list(self):
        self.ampdel_markers = []
        self.ampdel_stems = []

    def _add_aad(self,aad):
        self.ampdel_markers.append(aad[0])
        self.ampdel_stems.append(aad[1])

### PLOTTING FUNCTIONS ###

def plotssp(fname):
    '''
    Plots either sound speed profile or sound speed field, depending on
    input file configuration.
    Imports SSP data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    graph = TraceoPlot()
            
    # Case for sound speed profile (range-independent)
    if infile[sep[2]+1] == '\'c(z,z)\'\n': 
        ssp = infile[sep[2]+4:sep[3]]
        for i in range(len(ssp)):
            ssp[i] = np.array(ssp[i].split(' ')).astype(np.float64)
        ssp = np.array(ssp)
        graph.axes.plot(ssp[:,1],ssp[:,0],'b')
        
        # Axes config
        title = infile[0] + 'Sound speed profile'
        _axes_config(graph.axes,title)
        graph.axes.set_ylim(ssp[-1,0],0)
        graph.axes.set_xlabel('Sound speed (m/s)')
        graph.axes.grid(True)
    
    # Case for sound speed field (range-dependent)
    elif infile[sep[2]+1] == '\'c(r,z)\'\n':
        r = infile[sep[2]+4]
        r = np.array(r.split(' '))
        r = r[0:-1].astype(np.float64)
        z = infile[sep[2]+5]
        z = np.array(z.split(' '))
        z = z[0:-1].astype(np.float64)
        c = infile[sep[2]+6:sep[3]]
        C = np.zeros([len(r),len(z)])
        for i in range(len(c)):
            temp = np.array(c[i].split(' '))
            C[i,:] = temp[0:-1].astype(np.float64)
        r, z = np.meshgrid(r,z)
        img = graph.axes.pcolormesh(r, z, C, cmap='jet',shading='auto')
        graph._set_image(img)
        cbar = graph.fig.colorbar(graph.image)
        graph._set_colorbar(cbar)
        
        
        # Axes config
        title = infile[0] + 'Sound speed field'
        _axes_config(graph.axes,title,rbox=rbox)
        graph.colorbar.set_label('Sound speed (m/s)')
        graph.colorbar.ax.tick_params(labelsize=10)
    
    _fig_config(graph.fig)

    return graph

def plotenv(fname,ssp=True):
    '''
    Plots environment elements (surface, bottom, opbjects, source and receivers).
    If ssp == True, plots either the sound speed profile in a separate subplot or
    the sound speed field as a colormesh overlay over the main plot, depending on
    the input file SSP configuration.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    ssp : flag for sound speed profile or field plotting.
         type: boolean

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency
    
    # Adding graph elements
    graph = TraceoPlot()
    graph._add_surface(infile,sep)
    graph._add_bottom(infile,sep)
    graph._add_objects(infile,sep)
    graph._add_source(infile)
    graph._add_receivers(infile,sep)

    if ssp == True:
        # Case for sound speed profile (range-independent)
        if infile[sep[2]+1] == '\'c(z,z)\'\n': 
            gs = gridspec.GridSpec(1, 5)
            graph.axes.set_position(gs[0:4].get_position(graph.fig))
            graph.axes.set_subplotspec(gs[0:4])
            graph.ssp_axes = graph.fig.add_subplot(gs[4],sharey=graph.axes)
            ssp = infile[sep[2]+4:sep[3]]
            for i in range(len(ssp)):
                ssp[i] = np.array(ssp[i].split(' ')).astype(np.float64)
            ssp = np.array(ssp)
            graph.ssp_axes.plot(ssp[:,1],ssp[:,0],'b')
            
            # Axes config
            title = 'Sound speed\nprofile'
            _axes_config(graph.ssp_axes,title)
            graph.ssp_axes.set_xlabel('Sound speed (m/s)')
            graph.ssp_axes.grid(True)
            graph.ssp_axes.axes.get_yaxis().set_visible(False)
            graph.fig.tight_layout()
        
        # Case for sound speed field (range-dependent)
        elif infile[sep[2]+1] == '\'c(r,z)\'\n':
            r = infile[sep[2]+4]
            r = np.array(r.split(' '))
            r = r[0:-1].astype(np.float64)
            z = infile[sep[2]+5]
            z = np.array(z.split(' '))
            z = z[0:-1].astype(np.float64)
            c = infile[sep[2]+6:sep[3]]
            C = np.zeros([len(r),len(z)])
            for i in range(len(c)):
                temp = np.array(c[i].split(' '))
                C[i,:] = temp[0:-1].astype(np.float64)
            r, z = np.meshgrid(r,z)
            img = graph.axes.pcolormesh(r, z, C, cmap='jet',shading='auto',zorder=0)
            graph._set_image(img)
            cbar = graph.fig.colorbar(graph.image)
            graph._set_colorbar(cbar)
            
            # Axes config
            title = 'Sound speed field'
            _axes_config(graph.axes,title,rbox=rbox)
            graph.colorbar.set_label('Sound speed (m/s)')
            graph.colorbar.ax.tick_params(labelsize=10)
    
    # Graph config
    zmax = graph.maxdepth
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title,zmax=zmax,rbox=rbox)
    _fig_config(graph.fig)
    
    return graph

def plotray(fname,color_default='k',colorRSR='g',colorRBR='b',colorSRBR='r'):
    '''
    Plots ray paths from RCO, ERF, EPR or ARI runs.
    Also plots altimetry, bathymetry and objects.
    For ARI runs, colors rays according to path (reflections).
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    color_default : ray color for RCO, ERF and EPR runs.
                    refracted refracted ray color for ARI runs.
                   type : string
    colorRSR : refracted surface reflected ray color for for ARI runs.
              type : string
    colorRBR : refracted bottom reflected ray color for for ARI runs.
              type : string
    colorSRBR : surface reflected bottom reflecte ray color for for ARI runs.
              type : string

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency
    
    # Adding graph elements
    graph = TraceoPlot()
    graph._add_surface(infile,sep)
    graph._add_bottom(infile,sep)
    graph._add_objects(infile,sep)
    graph._add_source(infile)
    graph._create_rays_list()
    
    # Ray tracing run
    if infile[-2] == '\'RCO\'\n':
        rays = loadmat('rco.mat')
        nrays = np.size(rays['ray_elev'])
    # All ray information run
    elif infile[-2] == '\'ARI\'\n':
        rays = loadmat('ari.mat')
        nrays = np.size(rays['ray_elev'])
    # Eigenray run
    elif infile[-2] == '\'ERF\'\n' or infile[-2] == '\'EPR\'\n':
        graph._add_receivers(infile,sep)
        rays = loadmat('eig.mat')
        try:
            nrays = int(rays['neigrays'][0,0])
            if nrays == 0:
                print('No eigenrays found.')
        except KeyError:
            print('Eigenray calculation failed.')
            return
            
    # Plotting rays
    rayname = 'ray00000'
    infname = 'inf00000'
    for i in range(1,nrays+1):
        try:
            color = color_default # Refracted-only rays
            ri = str(i)
            ray = rays[rayname[0:-len(ri)] + ri]
            r = ray[0,:]
            z = ray[1,:]
            if infile[-2] == '\'ARI\'\n':
                info = rays[infname[0:-len(ri)] + ri]
                if info[0,1] > 0 and info[0,2] > 0: color = colorSRBR # Hits both surface and bottom
                elif info[0,1] > 0 and info[0,2] == 0: color = colorRSR # Hits only surface
                elif info[0,1] == 0 and info[0,2] > 0: color = colorRBR # Hits only bottom
            ray_trace = graph.axes.plot(r,z,color)
            graph._add_ray(ray_trace[0])
        except KeyError:
            print(str(rayname[0:-len(ri)] + ri) + ' calculation failed.')

    # Graph config
    zmax = graph.maxdepth
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title,zmax=zmax,rbox=rbox)
    _fig_config(graph.fig)
    
    return graph

def plotcpr(fname,cmap='viridis_r'):
    '''
    Plots coherent pressure transmission loss from CPR, CTL or PAV runs.
    Also plots altimetry, bathymetry and objects.
    Colormap range is estimated statistically, but may need to be changed.
    Input file needs to be in the same folder, since the function uses data
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    cmap : transmission loss colormap, defaults to 'viridis_r'.
          type : string

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    # Loading parameters from .in file
    infile, sep = _setup(fname)
    
    # Loading pressure data and calculating TL
    if infile[-2] == '\'CTL\'\n':
        tl = loadmat('ctl.mat')['cohtloss']
        tl = np.where(tl !=float ('+inf'), tl,-20*np.log10(0.0000001)) # Removes infinities
    else:
        if infile[-2] == '\'CPR\'\n':
            P = loadmat('cpr.mat')['pressure']
        elif infile[-2] == '\'PAV\'\n':
            P = loadmat('pav.mat')['pressure']
        P = np.where(P !=0, P, 0.0000001) # Removes zeros
        P = np.where(np.isnan(P), np.nanmax(P), P) # Removes NaN
        tl = -20*np.log10(abs(P))
    
    # Setting up range of colorbar
    med = np.median(tl[tl < -20*np.log10(0.0000001)])
    std = np.std(tl[tl < -20*np.log10(0.0000001)])
    cbarmax = med + std
    cbarmax = 10 * round(cbarmax/10)
    cbarmin = med - 2*std
    cbarmin = 10 * round(cbarmin/10)
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency
    
    # Adding graph elements
    graph = TraceoPlot()
    graph._add_surface(infile,sep)
    graph._add_bottom(infile,sep)
    graph._add_objects(infile,sep)
    graph._add_source(infile)
    zmax = graph.maxdepth

    # Plotting TL
    extent = _get_extent(infile, sep)
    img = graph.axes.imshow(np.flipud(tl), extent=extent,
                            cmap=cmap, aspect='auto')
    img.set_clim(cbarmin,cbarmax)
    graph._set_image(img)
    cbar = graph.fig.colorbar(graph.image)
    graph._set_colorbar(cbar)

    # Graph config
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title,zmax=zmax,rbox=rbox)
    graph.colorbar.set_label('Transmission loss (dB)')
    graph.colorbar.ax.tick_params(labelsize=10)
    graph.colorbar.ax.invert_yaxis()
    _fig_config(graph.fig)
    
    return graph

def plottlz(fname,z):
    '''
    Plots coherent pressure transmission loss along a chosen depth
    from CPR, CTL or PAV runs.
    Interpolates from a 2D TL field, so receiver geometry must be set as "RRY".
    Will not work for other geometries.
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    z : depths along which to plot TL
       type: float or list of floats

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    
    # Loading pressure data and calculating TL
    if infile[-2] == '\'CTL\'\n':
        tl = loadmat('ctl.mat')['cohtloss']
        tl = np.where(tl !=float ('+inf'), tl,-20*np.log10(0.0000001)) # Removes infinities
    else:
        if infile[-2] == '\'CPR\'\n':
            P = loadmat('cpr.mat')['pressure']
        elif infile[-2] == '\'PAV\'\n':
            P = loadmat('pav.mat')['pressure']
        P = np.where(P !=0, P, 0.0000001) # Removes zeros
        P = np.where(np.isnan(P), np.nanmax(P), P) # Removes NaN
        tl = -20*np.log10(abs(P))
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency
    
    graph = TraceoPlot()
    graph._add_bottom(infile,sep,tlsection=True)
    zmax = graph.maxdepth
    
    # Interpolation
    extent = _get_extent(infile,sep)
    ranges = np.linspace(extent[0],extent[1],tl.shape[1])
    depths = np.linspace(extent[2],extent[3],tl.shape[0])
    try:
        interpolate = interp2d(ranges,depths,tl)
    except Exception:
        print('Array geometry is not rectangular')
        return
    
    # Plotting TL
    if np.ndim(z) == 0:
        z = [z]
    for zi in z:
        if zi < 0 or zi > zmax:
            print('{} m out of bounds.'.format(zi))
        else:
            tl_z = interpolate(ranges,zi)
            graph.axes.plot(ranges,tl_z,label='TL @ {} m'.format(zi))

    # Graph config
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title,rbox=rbox)
    graph.axes.set_ylabel('Transmission Loss (dB)',fontsize=10)
    graph.axes.legend()
    _fig_config(graph.fig)
    
    return graph

def plottlr(fname,r):
    '''
    Plots coherent pressure transmission loss along a chosen range
    from CPR, CTL or PAV runs.
    Interpolates from a 2D TL field, so receiver geometry must be set as "RRY".
    Will not work for other geometries.
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    r : ranges along which to plot TL
       type: float or list of floats

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    
    # Loading pressure data and calculating TL
    if infile[-2] == '\'CTL\'\n':
        tl = loadmat('ctl.mat')['cohtloss']
        tl = np.where(tl !=float ('+inf'), tl,-20*np.log10(0.0000001)) # Removes infinities
    else:
        if infile[-2] == '\'CPR\'\n':
            P = loadmat('cpr.mat')['pressure']
        elif infile[-2] == '\'PAV\'\n':
            P = loadmat('pav.mat')['pressure']
        P = np.where(P !=0, P, 0.0000001) # Removes zeros
        P = np.where(np.isnan(P), np.nanmax(P), P) # Removes NaN
        tl = -20*np.log10(abs(P))
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency
    
    graph = TraceoPlot()
    graph._add_bottom(infile,sep,tlsection=True)
    zmax = graph.maxdepth

    # Interpolation
    extent = _get_extent(infile,sep)
    ranges = np.linspace(extent[0],extent[1],tl.shape[1])
    depths = np.linspace(extent[2],extent[3],tl.shape[0])
    try:
        interpolate = interp2d(ranges,depths,tl)
    except Exception:
        print('Array geometry is not rectangular')
        return

    # Plotting TL
    if np.ndim(r) == 0:
        r = [r]
    for ri in r:
        if ri < 0 or ri > rbox[1]:
            print('{} km out of bounds.'.format(ri/1000))
        else:
            tl_r = interpolate(ri,depths)
            graph.axes.plot(depths,tl_r,label='TL @ {} km'.format(ri/1000))
            
    # Graph config
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title)
    graph.axes.set_ylabel('Transmission Loss (dB)',fontsize=10)
    graph.axes.set_xlim(0,zmax)
    graph.axes.set_xlabel('Depth (m)',fontsize=10)
    graph.axes.legend()
    _fig_config(graph.fig)
    
    return graph

def plotpvl(fname,cmap='viridis_r'):
    '''
    Plots coherent transmission loss for horizontal and vertical components
    of particle velocity for PVL or PAV runs as subplots on the same figure.
    Also plots altimetry, bathymetry and objects.
    Colormap range is estimated statistically, but may need to be changed. In
    this case, both images must have the new range set with set_clim to keep
    them in the same scale.
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    cmap : transmission loss colormap, defaults to 'viridis_r'.
          type : string

    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.

    '''
    infile, sep = _setup(fname)
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    rbox = np.array(infile[4].split(' ')).astype(np.float64) # bounding box x coordinates
    f = float(infile[5]) # frequency

    # Adding graph elements
    graph = TraceoPlot(2)
    graph._add_surface(infile,sep)
    graph._add_bottom(infile,sep)
    graph._add_objects(infile,sep)
    graph._add_source(infile)
    zmax = graph.maxdepth
    
    # Loading particle velocity data and calculating TL
    if infile[-2] == '\'PVL\'\n':
            pvlr = loadmat('pvl.mat')['gradpr_r']
            pvlz = loadmat('pvl.mat')['gradpr_z']
    elif infile[-2] == '\'PAV\'\n':
            pvlr = loadmat('pav.mat')['gradpr_r']
            pvlz = loadmat('pav.mat')['gradpr_z']
    pvlr = np.where(pvlr !=0, pvlr, 0.0000001) # Removes zeros
    pvlz = np.where(pvlz !=0, pvlz, 0.0000001) # Removes zeros
    pvlr = np.where(np.isnan(pvlr), np.nanmax(pvlr), pvlr) # Removes NaN
    pvlz = np.where(np.isnan(pvlz), np.nanmax(pvlz), pvlz) # Removes NaN
    tlr = -20*np.log10(abs(pvlr))
    tlz = -20*np.log10(abs(pvlz))
    tl = np.array([tlr,tlz])
    
    # Setting up range of colorbar
    med = np.median(tl[tl < -20*np.log10(0.0000001)])
    std = np.std(tl[tl < -20*np.log10(0.0000001)])
    cbarmax = med + 0.75*std
    cbarmax = 10 * round(cbarmax/10)
    cbarmin = med - 2*std
    cbarmin = 10 * round(cbarmin/10)
    
    # Plotting TL
    extent = _get_extent(infile, sep)
    imges = []
    for k, ax in enumerate(graph.axes):
        imges.append(ax.imshow(np.flipud(tl[k,:,:]), extent=extent,
                               cmap=cmap, aspect='auto'))
        imges[k].set_clim(cbarmin,cbarmax)
        
        # Graph config
        if k == 0:
            title = 'Horizontal particle velocity'
        elif k == 1:
            title = 'Vertical particle velocity'
        _axes_config(ax,title,zmax=zmax,rbox=rbox)

    graph.axes[0].xaxis.label.set_visible(False)
    graph._set_image(imges)
    cbar = graph.fig.colorbar(imges[k],ax=graph.axes)
    graph._set_colorbar(cbar)
    graph.colorbar.set_label('Transmission loss (dB)')
    graph.colorbar.ax.invert_yaxis()
    graph.colorbar.ax.tick_params(labelsize=10)
    graph.fig.suptitle(infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
                       ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz')
    _fig_config(graph.fig)
    
    return graph

def plotaad(fname,which=[]):
    '''
    Plots ray amplitude against travel time for ADR and ADP runs.
    Series of points are grouped by receiver, if there are more than one.
    Which receivers' response are to be plotted need to be specified by the
    "which" parameter. If not specified, all receivers will be plotted on the
    same axes, which can get crowded for too many receivers.
    Imports data from TRACEO input file.

    Parameters
    ----------
    fname : path of the input file with filename, but without extension.
           type: string
    which : indexes of receivers to be plotted.
           type: float or list of floats
    
    Returns
    -------
    graph : instance of TraceoPlot object containing plot elements.
    
    '''
    infile, sep = _setup(fname)
    
    # General information
    src = np.array(infile[3].split(' ')).astype(np.float64) # source coordinates
    f = float(infile[5]) # frequency
    
    graph = TraceoPlot()
    graph._create_aad_list()
    
    # Getting receiver coordinates from .mat file
    ampdel = loadmat('aad.mat')
    rrecs = list(zip(*ampdel['rarray2d']))
    zrecs = list(zip(*ampdel['zarray2d']))
    recs = list(product(rrecs,zrecs))
    nrays = int(ampdel['neigrays'][0,0])
    if nrays == 0:
        print('No eigenrays found.')
        return
    name = 'aad00000'
    
    if which == []:
        which = range(len(recs))
    if np.ndim(which) == 0:
        which = [which]
    for n, rec in enumerate(recs):
        if n in which:
            leg = list(zip(*rec))
            t = []
            a = []
            for i in range(1,nrays+1):
                ri = str(i)
                aad = ampdel[name[0:-len(ri)] + ri]
                if tuple((aad[0,0],aad[0,1])) == rec: # Grouping by receiver
                    t.append(aad[0,2])
                    a.append(abs((aad[0,3]+1j*aad[0,4])*np.exp(1j*aad[0,5])))
            m = graph.axes.plot(t,a,marker='o',markersize=10,markeredgewidth=2,markerfacecolor="None",linestyle='none',label=str(leg[0]))
            color = m[0].get_color()
            
            # Plotting vertical lines
            s = []
            for i in range(len(t)):
                s.append(graph.axes.plot([t[i],t[i]],[0,a[i]],color=color,linewidth=1)[0])
                
            graph._add_aad([m[0],s])
            
    # Graph config
    title = infile[0] + 'r\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[0]/1000) + \
            ' km, z\N{LATIN SUBSCRIPT SMALL LETTER S} = ' + str(src[1]) + ' m, f = ' + str(f) + ' Hz'
    _axes_config(graph.axes,title)
    graph.axes.grid(True)
    graph.axes.invert_yaxis()
    graph.axes.set_ylim(0)
    graph.axes.set_xlabel('Travel time (s)')
    graph.axes.set_ylabel('Ray amplitude')
    graph.axes.legend(title='Receiver @:')
    _fig_config(graph.fig)
    
    return graph

### INPUT FILE WRITING FUNCTION ###
    
def wtraceoinfil(filename=None, thetitle=None, source_info=None, surface_info=None, ssp_info=None, object_info=None, bathymetry_info=None, output_info=None):
    '''
    Writes Traceo input (waveguide) file. 
    
    SYNTAX: wtraceoinfil( filename, title, source, surface, ssp, object, bottom, output )
    
    *******************************************************************************
    Arraial do Cabo, RJ, 13/10/2016
    Written by Orlando Camargo Rodriguez and Simone Pacheco
    *******************************************************************************    
    '''
    filename = filename + '.in'
       
    separation_line = "--------------------------------------------------------------------------------"
    
    #*******************************************************************************
    # Get source data: 

    ds = source_info["ds"]
    xs = source_info["position"]
    rbox = source_info["rbox"]
    freq = source_info["f"]
    thetas = source_info["thetas"]
    nthetas = thetas.size

    theta1 = thetas[0]
    thetan = thetas[nthetas-1]

    #*******************************************************************************
    # Get surface data: 

    atype = surface_info["type"]
    aptype = surface_info["ptype"]
    aitype = surface_info["itype"]
    xati = surface_info["x"]
    nati = xati[0,].size
    #nati = length(xati(1, mslice[:])); print nati

    
    atiu = surface_info["units"]
    aproperties = surface_info["properties"]

    #*******************************************************************************
    # Get sound speed data: 

    cdist  = ssp_info["cdist"]
    cclass = ssp_info["cclass"]

    c = ssp_info["c"]
    z = ssp_info["z"]
    r = ssp_info["r"]
        
    #*******************************************************************************
    # Get object data:

    nobj = object_info["nobjects"]

    if nobj > 0:
        npobj = object_info["npobjects"]
        otype = object_info["type"]
        oitype = object_info["itype"]
        xobj = object_info["x"]
        obju = object_info["units"]
        oproperties = object_info["properties"]
    #*******************************************************************************  
    # Get bathymetry data:

    btype = bathymetry_info["type"]
    bptype = bathymetry_info["ptype"]
    bitype = bathymetry_info["itype"]
    xbty = bathymetry_info["x"]
    nbty = xbty[0,].size
 
    #nbty = length(xbty(1, mslice[:]))    #; print nbty"]
    
    btyu = bathymetry_info["units"]
    bproperties = bathymetry_info["properties"]

    #*******************************************************************************  
    # Get output options: 

    calc_type = output_info["ctype"]
    array_shape = output_info["array_shape"]
    array_r = output_info["r"]
    array_z = output_info["z"]
    array_miss = output_info["miss"]
   
    m = array_r.size  #m = length(array_r)
    n = array_z.size  #n = length(array_z)

    #*******************************************************************************  
    # Write the INFIL: 

    # fid = fopen(filename, mstring('w'))
    fid = open(filename, 'w')
    fid.write(thetitle);fid.write('\n')
    fid.write(separation_line);fid.write("\n")
    
    fid.write(str(ds))
    fid.write("\n")
    fid.write(str(xs[0]))
    fid.write(" ")
    fid.write(str(xs[1]))
    fid.write("\n")
    fid.write(str(rbox[0]))
    fid.write(" ")
    fid.write(str(rbox[1]))
    fid.write("\n")
    fid.write(str(freq))
    fid.write("\n")
    fid.write(str(nthetas))
    fid.write("\n")
    fid.write(str(theta1))
    fid.write(" ")
    fid.write(str(thetan))     
    fid.write("\n")
    fid.write(separation_line);fid.write("\n")
    fid.write('\'');fid.write(atype) ;fid.write('\'\n')
    fid.write('\'');fid.write(aptype);fid.write('\'\n')
    fid.write('\'');fid.write(aitype);fid.write('\'\n')
    fid.write('\'');fid.write(atiu)  ;fid.write('\'\n')
    fid.write(str(nati))
    fid.write("\n")
    
    if aptype == 'H':
       fid.write(str(aproperties[0]));fid.write(" ")
       fid.write(str(aproperties[1]));fid.write(" ")
       fid.write(str(aproperties[2]));fid.write(" ")
       fid.write(str(aproperties[3]));fid.write(" ")
       fid.write(str(aproperties[4]));fid.write('\n')
       for i in range(nati):
          fid.write(str(xati[0][i]));fid.write(" ")
          fid.write(str(xati[1][i]));fid.write('\n')
    elif aptype == 'N':
       for i in range(nati):
          fid.write(str(xati[0][i]));fid.write(" ")
          fid.write(str(xati[1][i]));fid.write(" ")
          fid.write(str(aproperties[0][i]));fid.write(" ")
          fid.write(str(aproperties[1][i]));fid.write(" ")
          fid.write(str(aproperties[2][i]));fid.write(" ")
          fid.write(str(aproperties[3][i]));fid.write(" ")
          fid.write(str(aproperties[4][i]));fid.write('\n')            
    else:
       print('Unknown surface properties...')

    fid.write(separation_line);fid.write("\n")
    fid.write('\'');fid.write(cdist) ;fid.write('\'\n')
    fid.write('\'');fid.write(cclass);fid.write('\'\n')
    
    if cdist == 'c(z,z)':
       nz = z.size
       fid.write('1 ');fid.write(str(nz));fid.write('\n')
       for i in range(nz):
          fid.write(str(z[i]));fid.write(" ")
          fid.write(str(c[i]));fid.write('\n')

    elif cdist == 'c(r,z)':
       
       nz = z.size#m = length( r );
       nr = r.size#n = length( z ); 
       
       #fprintf(fid,   '%d %d\n' ,[m n]);
       
       fid.write(str(nr));fid.write(" ")#fprintf(fid,'%e ',r);fprintf(fid,'\n');
       fid.write(str(nz));fid.write('\n')#fprintf(fid,'%e ',z);fprintf(fid,'\n');
       #gravando distância
       for i in range(nr):
          fid.write(str(r[i]));fid.write(" ")
       fid.write('\n')
       #gravando profundidades
       for i in range(nz):
          fid.write(str(z[i]));fid.write(" ")       
       fid.write('\n')
       #gravando velocsom
       for i in range(nz):
           for j in range(nr):
               fid.write( str( c[i,j] ) );fid.write(" ")
           fid.write('\n')    
    else:
       print('Unknown sound speed distribution...')
    fid.write(separation_line);fid.write("\n")

    #----------------object--------------------  
    fid.write(str(nobj));fid.write('\n')#object_data['nobjects'] = 1 #; % No objects
    if nobj > 0:
        fid.write(str(oitype));fid.write('\n') #object_data['itype']     = "2P" #; % Objects interpolation type
        for i in range(nobj):
            fid.write(str(otype[i]));fid.write('\n') #object_data['type']      = "R"
            fid.write(str(obju[i]));fid.write('\n') #object_data['units']     = "W"#; % (Attenuation Units) Wavelenght 
            fid.write(str(npobj[i]));fid.write('\n') #; % Number of points in each object
            
            ##exit()
            
            #oproperties
            fid.write(str(oproperties[i][0]));fid.write(" ")
            fid.write(str(oproperties[i][1]));fid.write(" ")
            fid.write(str(oproperties[i][2]));fid.write(" ")
            fid.write(str(oproperties[i][3]));fid.write(" ")
            fid.write(str(oproperties[i][4]));fid.write('\n')   
            #object_data['x'] = xboj
            j=0
            for j in range(npobj[i]): 
                fid.write (str(xobj[i][0,j]));fid.write(" ")
                fid.write(str(xobj[i][1,j]));fid.write(" ")
                fid.write(str(xobj[i][2,j]));fid.write('\n')
        
    #----------------------------------------------------------------------------
    fid.write(separation_line);fid.write("\n")
    
    fid.write('\'');fid.write(btype) ;fid.write('\'\n')
    fid.write('\'');fid.write(bptype);fid.write('\'\n')
    fid.write('\'');fid.write(bitype);fid.write('\'\n')
    fid.write('\'');fid.write(btyu)  ;fid.write('\'\n')
    fid.write(str(nbty))
    fid.write("\n")
    
    if aptype == 'H':
       fid.write(str(bproperties[0]));fid.write(" ")
       fid.write(str(bproperties[1]));fid.write(" ")
       fid.write(str(bproperties[2]));fid.write(" ")
       fid.write(str(bproperties[3]));fid.write(" ")
       fid.write(str(bproperties[4]));fid.write('\n')
       for i in range(nbty):
          fid.write(str(xbty[0][i]));fid.write(" ")
          fid.write(str(xbty[1][i]));fid.write('\n')
    elif aptype == 'N':
       for i in range(nbty):
          fid.write(str(xbty[0][i]));fid.write(" ")
          fid.write(str(xbty[1][i]));fid.write(" ")
          fid.write(str(bproperties[0][i]));fid.write(" ")
          fid.write(str(bproperties[1][i]));fid.write(" ")
          fid.write(str(bproperties[2][i]));fid.write(" ")
          fid.write(str(bproperties[3][i]));fid.write(" ")
          fid.write(str(bproperties[4][i]));fid.write('\n')            
    else:
       print('Unknown bottom properties...')
    
    fid.write(separation_line);fid.write("\n")
    fid.write('\'');fid.write(array_shape);fid.write('\'\n')
    fid.write(str(m));fid.write(" ")
    fid.write(str(n));fid.write('\n')
    
    for i in range(m):
        fid.write(str(array_r[i]));fid.write(" ")
    fid.write('\n')
    
    for i in range(n):
        fid.write(str(array_z[i]));fid.write(" ")
    fid.write('\n')
    
    fid.write(separation_line);fid.write("\n")
    
    fid.write('\'');fid.write(calc_type);fid.write('\'\n')
    fid.write(str(array_miss));fid.write("\n")

    fid.close()
