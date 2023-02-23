


# Supplementary file for tests

import math
import os
import sys
#from hapi2 import *
from hapi import *
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1,'/work/WORK/WORK/scripts/regular_scripts')
sys.path.insert(2,'/work/WORK/WORK/scripts/hapi2')
import json
from getpass import getpass

import pylab as pl
import matplotlib as mpl
import time
import h5py




f = h5py.File('CO_HDF5.hdf5', mode='r')

print(f.keys())

coef_hdf5 = f['Gas_05_Absorption'][()][1][2][0]
nu_hdf5 = f['Wavenumber'][()]

print(nu_hdf5, coef_hdf5)

f.close()

db_begin('05_hit20')

wn_begin = 0.0
wn_end = 15000.0

Nwn = 1500001

pres = 1.0
Temp = 296.15
VMS = 0.00


wn_step = (wn_end-wn_begin)/(Nwn-1)


nu_co,coef_co = absorptionCoefficient_Voigt(SourceTables='COall',
                                             HITRAN_units=True, OmegaRange=[wn_begin,wn_end],
                                             WavenumberStep=wn_step,
                                             WavenumberWing=25.0,
                                             Diluent={'self':1.00-VMS, 'H2O':VMS},
                                             Environment={'T':Temp,'p':pres},
                                             File = './datafiles/test_4_pict.dat')
 


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

nu_start =      0.0
nu_end   =  15000.0

y_start = 1.0e-26
y_end   = 1.0e-17

axYlog = True

pict_open = False


title_band = r'CO test spectra'

figure1 = plt.figure(figsize=(24,12),dpi=420)
ax1 = figure1.add_subplot(211)
ax2 = figure1.add_subplot(212, sharex=ax1)

ax1.set_title(title_band, y=1.05)
#ax.set_xscale('log')
if (axYlog):
    ax1.set_yscale('log')

ax1.set_xlim(nu_start, nu_end)
ax1.set_ylim(y_start,y_end)
ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
ax1.set_ylabel('Cross-section, cm$^2$/molecule')

#ax2.set_ylim(-1e-25,1e-25)
ax1.plot(nu_co, coef_co, label=r'CO HITRAN spectra, 0.30 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.1,0.1,0.9),linewidth=1.0)
ax1.plot(nu_hdf5, coef_hdf5, label=r'CO HITRAN spectra (HDF5), 0.30 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.9,0.1,0.1),linewidth=1.0)

ax2.plot(nu_co, coef_co-coef_hdf5)

ax1.legend()

#plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
plt.savefig('./images/cross_test_calc_vs_HDF5_XSEC_CO.jpg',bbox_inches='tight')
plt.close()


dtype1 = np.dtype([('nu','float'),('coef','float')])

if (pict_open):
    os.system("./images/cross_test_calc_vs_HDF5_XSEC_CO.jpg")
