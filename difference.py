from main import openParametersFile, openPressure, openTemp, openVMS, openXgenetareWn


# Supplementary file for tests

import math
import os
import sys
#from hapi2 import *
from hapi import *
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1,'/work/WORK/WORK/scripts/regular_scripts')
#sys.path.insert(2,'/work/WORK/WORK/scripts/hapi2')
sys.path.insert(3,'/work/WORK/WORK/scripts/hapiN')
import json
from getpass import getpass
from scipy.interpolate import interp1d




import pylab as pl
import matplotlib as mpl
import time
import h5py

from main import openParametersFile, openPressure, openTemp, openVMS, openXgenetareWn, OpenHDF5, CloseHDF5, ParallelPart

import random


# flag to print values in functions to debug 
FLAG_DEBUG_PRINT = True
# flag to store prints in OUTPUT.LOG file
FLAG_LOG_FILE = True
# flag to open\close HDF5 file 
FLAG_OPENED_HDF5 = False
# flag to remove HAPI files
FLAG_REMOVE_HAPI = True


INPUT_FILENAME = 'params.inp'

PRES_FILENAME = 'pres.inp'

TEMP_FILENAME = 'temps.inp'

VMS_FILENAME = 'vms.inp'

WN_FILENAME = 'wn.inp'

ParametersCalculation = openParametersFile(INPUT_FILENAME)

Pressures, Np = openPressure(PRES_FILENAME)

(Temps, Npp,Ntt) = openTemp(TEMP_FILENAME, Np)

(VMSs, Nvms) = openVMS(VMS_FILENAME)

WNs, Nwn = openXgenetareWn(WN_FILENAME,ParametersCalculation)


#f = h5py.File('D:\O3_HDF5.hdf5', mode='r')
f = h5py.File('CO_HDF5_3.0_to_5.3_num.hdf5', mode='r')
f2 = h5py.File('CO_HDF5_3.0_to_5.3_num_H2.hdf5', mode='r')

p_arr = f['Pressure'][()]
t_arr = f['Temperature'][()][18]

nu_my = f['Wavenumber'][()]
co_my = f['Gas_05_Absorption'][()][18][12][0]

nu_my2 = f2['Wavenumber'][()]
co_my2 = f2['Gas_05_Absorption'][()][18][12][0]


press_val = p_arr[18]
temp_val = t_arr[12]

print(press_val,temp_val,'********')

#print(f['Gas_05_Absorption'][()][11][11][0].shape)


pres=1.0
Temp=300


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title

resolution_pnnl = 0.015

nu_start =  2020.0
nu_end   =  2029.0

y_start = 1.0e-36
y_end   = 1.0e-15

axYlog = True

pict_open = False

title_band=''
  
figure1 = plt.figure(figsize=(24,12),dpi=700)
ax1 = figure1.add_subplot(211)
ax2 = figure1.add_subplot(212, sharex=ax1)

ax1.set_title(title_band, y=1.05)
#ax.set_xscale('log')
if (axYlog):
    ax1.set_yscale('log')
    ax2.set_yscale('log')

ax1.set_xlim(nu_start, nu_end)
ax1.set_ylim(y_start,y_end)
ax2.set_ylim(y_start,y_end)

ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
ax1.set_ylabel('Cross-section, cm$^2$/molecule')
#ax1.text(1900,1e-19,'%2d %2d %2d'%(18,5,1.0))
#ax2.set_ylim(-1e-25,1e-25)
#ax1.plot(nu_hdf5, coef_h5, label=r'CO HITRAN spectra, 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.1,0.1,0.9),linewidth=1.0)
#ax1.plot(nu_hdf5, coef_hdf5, label=r'CO HITRAN spectra (HDF5), 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.9,0.1,0.1),linewidth=1.0)

ax1.plot(nu_my, co_my,ms=0.05, alpha=0.3, label='H20-broad')
ax1.plot(nu_my2, co_my2, ms=0.05, alpha=0.1,label='H2-broad')


ax2.plot(nu_my, co_my-co_my2,ms=0.2, alpha=0.5,label='H2O-H2')



ax1.legend()
name_img = "./images/diff_broad_h2o_vs_h2.jpg"
#plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
plt.savefig(name_img,bbox_inches='tight')
plt.close()

