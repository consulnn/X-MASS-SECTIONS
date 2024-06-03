import numpy as np
import os, sys
import exo_k as xk
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(1,'/Work_remote/WORK/scripts')
#sys.path.insert(2,'/work/WORK/WORK/scripts/hapi2')
sys.path.insert(3,'/Work_remote/WORK/scripts/hapiN')
import json
from getpass import getpass
from scipy.interpolate import interp1d
print(os.getcwd())
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# ## for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

 
import pylab as pl
import time
import glob
import multiprocessing
from multiprocessing import Pool
import argparse
import h5py

from main import openParametersFile, openPressure, openTemp, openVMS, openXgenetareWn, OpenHDF5, CloseHDF5, ParallelPart

import random

def cm2mum(x):
    return 10000./x







#######################################
### OPEN petitRADTRANS R=1e+6 files ###
#######################################

ID_molec = 5

path_to_hires_spectra='D:\Work_remote\WORK\X-MASS-SECTIONS\datafiles\CO_main_iso_pRT'
#path_to_hires_spectra='D:\Work_remote\WORK\X-MASS-SECTIONS\datafiles\CO_all_pRT'
#path_to_hires_spectra='D:\Work_remote\WORK\X-MASS-SECTIONS\datafiles\H2O_main_pRT'

press_grid_str=['0.000001','0.000010','0.000100','0.001000','0.010000','0.100000',
                '1.000000','10.000000','100.000000','1000.000000']  # notice the use of strings
logp_grid=[np.log10(float(p)) for p in press_grid_str]
t_grid=[81,110,148,200,270,365,
        493,666,900,1215,1641,
        2217,2995]

file_grid=xk.create_fname_grid('sigma_%02d_{temp}.K_{press}bar.dat'%(ID_molec), logpgrid=press_grid_str, tgrid=t_grid,
        p_kw='press', t_kw='temp')
print(file_grid)

Hires_spectra=xk.hires_to_xtable(path=path_to_hires_spectra, filename_grid=file_grid, logpgrid=logp_grid, tgrid=t_grid,
                mol='CO', grid_p_unit='bar', binary=True, mass_amu=28.)

print(Hires_spectra)

f = h5py.File('12C16OHITEMP_JWST_air.pRT.600k.100wing.hdf5', mode='r')


plot_pict = False

axYlog = True

pict_open = False

relative_diff = True



for ind_pRT_pres in np.arange(10):
    for ind_pRT_temp in np.arange(13):
    
        ind_pres = ind_pRT_pres 
        ind_temp = ind_pRT_temp 
    
    
        co_pRT = Hires_spectra.kdata[ind_pRT_pres,ind_pRT_temp,:]
        nu_pRT = (Hires_spectra.wnedges[:-1]+Hires_spectra.wnedges[1:])/2. # - 0.0025
        
        
        ################################
        ### OPEN HAPI X-SEC ############
        ################################
        
        
        # f = h5py.File('CO_all_HITEMP_3.0_to_5.3_air.pRT.450k.hdf5', mode='r')
        # f = h5py.File('H2O_HITRAN_3.0_to_5.3_num_air.hdf5', mode='r')
        
        
        
        p_arr = f['Pressure'][()]
        t_arr = f['Temperature'][()][ind_pres]
        
        
        print(p_arr[ind_pres], t_arr[ind_temp])
        
        
        nu_hapi = f['Wavenumber'][()]
        co_hapi = (f['Gas_05_Absorption'][()])[ind_pres,ind_temp,0]
        

        
        
        
        # ######################################
        # ### OPEN HAPI X-SEC (HITRAN) #########
        # ######################################
        
        
        # f_hit = h5py.File('CO_all_HITRAN_3.0_to_5.3_air.pRT.450k.hdf5', mode='r')
        # # f = h5py.File('H2O_HITRAN_3.0_to_5.3_num_air.hdf5', mode='r')
        
        # print(f_hit.keys())
        # print(f_hit['Pressure'][()]*1.01325)
        
        # ind_pres_hit = 0
        # ind_temp_hit = 8
        
        
        # p_arr_hit = f_hit['Pressure'][()]
        # t_arr_hit = f_hit['Temperature'][()][ind_pres]
        
        
        # print(p_arr_hit[ind_pres], t_arr_hit[ind_temp])
        
        
        # nu_hapi_hit = f_hit['Wavenumber'][()]
        # co_hapi_hit = (f_hit['Gas_05_Absorption'][()])[ind_pres_hit,ind_temp_hit,0]
        
        
        
        # print(co_hapi)
        
        # f_hit.close()
        
        
        
        
        
        
        
        
        
        
        
        co_pRT = co_pRT[nu_pRT>1850.]
        nu_pRT = nu_pRT[nu_pRT>1850.]
        
        co_pRT = co_pRT[nu_pRT<3350.]
        nu_pRT = nu_pRT[nu_pRT<3350.]
        
        
        print(len(nu_pRT), len(nu_hapi))
       
        
        hapi_int = interp1d(nu_hapi, co_hapi,kind='linear')
        co_hapi_int = hapi_int(nu_pRT)
        
        
        nu_ratio = nu_pRT
        co_ratio = (co_pRT-co_hapi_int)/co_pRT*100.
        
        strong_delimit = 1e-19
        
        co_ratio_strong = co_ratio[co_pRT > strong_delimit]
        nu_ratio_strong = nu_pRT[co_pRT > strong_delimit]
        
        co_ratio_weak = co_ratio[co_pRT < strong_delimit]
        nu_ratio_weak = nu_pRT[co_pRT < strong_delimit]
        
        
        
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
        plt.rc('axes', titlesize=25)     # fontsize of the axes title
        plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
        plt.rc('legend', fontsize=15)    # legend fontsize
        plt.rc('figure', titlesize=25)  # fontsize of the figure title
        
        resolution_pnnl = 0.015
        
        nu_start =  1850.0
        nu_end   =  2350.0
        
        y_start = 1.0e-24
        y_end   = 1.0e-16
        
        title_band=''
        
        if (plot_pict):
        
            for ind_plot in np.arange(2):
                figure1 = plt.figure(figsize=(24,12),dpi=700)
                ax1 = figure1.add_subplot(211)
                ax2 = figure1.add_subplot(212, sharex=ax1)
                
                ax1.set_title(title_band, y=1.05)
                #ax.set_xscale('log')
                if (axYlog):
                    ax1.set_yscale('log')
                    # ax2.set_yscale('log')
                    Scale = 'log'
                #    ax2.set_yscale('log')
                else:
                    Scale = 'lin'
                
                if (ind_plot==1):
                    nu_start = 2107.2
                    nu_end   = 2107.6
                    ax2.set_ylim(-10,10)
                    # y_start = 1e-30
                else:
                    ax2.set_ylim(-100,100)                                          # 
            
            
            
            
            
                ax1.set_xlim(nu_start, nu_end)
                ax1.set_ylim(y_start,y_end)
            #    ax2.set_ylim(y_start,y_end)
                
                
                
                secax = ax1.secondary_xaxis('top', functions=(cm2mum,cm2mum))
                secax.set_xlabel(r'$\mu m$')
                
                ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
                ax1.set_ylabel('Cross-section, cm$^2$/molecule')
                
                ax1.plot(nu_pRT,co_pRT, label='petitRADTRANS')#,linewidth=2.5)
                ax1.plot(nu_hapi,co_hapi, label='HAPI')#linewidth=0.7, alpha=0.5)
                ax1.plot(nu_pRT, co_hapi_int, label='Extrapolated HAPI')
                # ax1.plot(nu_hapi_hit,co_hapi_hit, label='HAPI (HITRAN)',alpha=0.4)#linewidth=0.7, alpha=0.5)
            
                
                ax2.plot(nu_ratio_strong, co_ratio_strong, label='(pRT-HAPI)/pRT*100\%')
                ax2.plot(nu_ratio_weak, co_ratio_weak, label='(pRT-HAPI)/pRT*100\%',alpha=0.3)
            
                # ax2.plot(nu_hapi_hit,(co_hapi_hit-co_hapi)/co_hapi_hit*100., label='(HITRAN-HAPI)/HITRAN*100\%',alpha=0.5)
                
                # ax2.plot(nu_pRT,abs(co_pRT-co_hapi_int), label='$|$pRT-HAPI$|$')
            
                # ax2.plot(nu_hapi_hit,abs(co_hapi_hit-co_hapi), label='$|$HITRAN-HAPI$|$',alpha=0.5)
                
                ax1.text(nu_start,(y_start*y_end)**0.5,'p=%4.2e bar, T=%6.2f K'%(p_arr[ind_pres]*1.01325, t_arr[ind_temp]),fontsize=25)
                ax2.text(nu_start,0.0,'strong/weak threshold is %5.2e'%(strong_delimit),fontsize=25)
            
                ax1.legend()
                
                ax2.legend()
                
                name_img = "./images/CO_pRT_vs_HAPI_%d_%4.2e_%8.2f.voigt.%3s.600k.100wing.jpg"%(ind_plot,p_arr[ind_pres]*1.01325,t_arr[ind_temp],Scale)
                #plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
                plt.savefig(name_img,bbox_inches='tight')
                plt.close()
        
        N_a = 6.022140857e+23
        mass_amu = 2.799491E+01
        co_hapi_pRT = co_hapi_int * N_a / mass_amu
        co_hapi_pRT.tofile('./datafiles/HAPI_pRT/sigma_05_%3.0f.K_%8.6fbar.dat'%(t_arr[ind_temp], p_arr[ind_pres]*1.01325))
    
f.close()

lambda_pRT = 1.0e0 / nu_pRT
lambda_pRT.tofile('./datafiles/HAPI_pRT/wlen.dat')

wlens = np.fromfile(path_to_hires_spectra + '\wlen.dat')
blins = np.fromfile(path_to_hires_spectra + '\sigma_05_493.K_0.010000bar.dat')
wlens2 = np.fromfile('./datafiles/HAPI_pRT/wlen.dat')
blins2 = np.fromfile('./datafiles/HAPI_pRT/sigma_05_493.K_0.010000bar.dat')
wlens = 1.0e0 / wlens
wlens2 = 1.0e0 / wlens2


print(len(wlens))
print(len(blins))

fig = plt.figure(figsize=(24,12),dpi=700)
ax = fig.add_subplot(111)

ax.set_xlim(1950.0,2300.0)
ax.set_yscale('log')
ax.set_ylim(1.0e-8,1.0e+9)



ax.plot(wlens,blins,label='Mercedes file')
ax.plot(wlens2,blins2,label='Generated file')
ax.legend()

plt.savefig('./images/wlen-blins.jpg')
plt.close()



