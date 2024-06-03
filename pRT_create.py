import numpy as np
import os, sys
# import exo_k as xk
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
    return 1./x

N_a = 6.022140857e+23
mass_amu =    63.961901   #44.993185#44.013E+00#3.1998E+01
abundance =   9.45678e-1	#1.10574E-02

# 7	    1	12C16O2	    626	9.84204 × 10-1	43.989830	2.8609 × 102	q7.txt	1
# 8	    2	13C16O2 	636	1.10574 × 10-2	44.993185	5.7664 × 102	q8.txt	2
# 9 	3	16O12C18O	628	3.94707 × 10-3	45.994076	6.0781 × 102	q9.txt	1
# 10	4	16O12C17O	627	7.33989 × 10-4	44.994045	3.5426 × 103	q10.txt	6
# 11	5	16O13C18O	638	4.43446 × 10-5	46.997431	1.2255 × 103	q11.txt	2
# 12	6	16O13C17O	637	8.24623 × 10-6	45.997400	7.1413 × 103	q12.txt	12





f = h5py.File('./hydra_ver/SO2.HITRAN2020.25wing.0-35k.hdf5', mode='r')

# print(f['Pressure'][()])
# exit()
lambda_pRT = np.fromfile('./hydra_ver/datafiles/wlen.dat')


lambda_pRT = lambda_pRT 

# print(nu_pRT[0]*10000, nu_pRT[-1]*10000)
# print(len(nu_pRT))

#nu_pRT = 1.0e0/nu_pRT

# quit()

# nu_pRT = nu_pRT[nu_pRT< 18000.0]

# nu_pRT = nu_pRT[nu_pRT>  6000.0]

# print(len(nu_pRT))

#nu_pRT = np.flip(nu_pRT)

plot_print = False

co_sent = np.fromfile('./datafiles/O2_HAPI_pRT/test_0.76/sigma_07_270.K_1.000000bar.dat')
co_sent = np.flip(co_sent)


#print(co_sent[lambda_pRT>0.000076])


for ind_pRT_pres in np.arange(10):
    for ind_pRT_temp in np.arange(13):
    
        ind_pres = ind_pRT_pres 
        ind_temp = ind_pRT_temp 
    
    
        
        
        ################################
        ### OPEN HAPI X-SEC ############
        ################################
        
        
        # f = h5py.File('CO_all_HITEMP_3.0_to_5.3_air.pRT.450k.hdf5', mode='r')
        # f = h5py.File('H2O_HITRAN_3.0_to_5.3_num_air.hdf5', mode='r')
        
        
        
        p_arr = f['Pressure'][()]
        t_arr = f['Temperature'][()][ind_pres]
        
        
        print(p_arr[ind_pres], t_arr[ind_temp])
        print(ind_pres,ind_temp)
        
        print(f.keys())
        nu_hapi = f['Wavenumber'][()]
        co_hapi = (f['Gas_09_Absorption'][()])[ind_pres,ind_temp,0]
        print(co_hapi)
        
        #REVERSED BECAUSE 0,1,2,3 --> 0.4,0.3,0.2,0.1 in LAMBDA_SPACE
        lambda_hapi = np.flip(1.0e0 / nu_hapi)
        co_hapi = np.flip(co_hapi)
        
        lambda_hapi = lambda_hapi 
        
        
        hapi_int = interp1d(lambda_hapi, co_hapi,kind='slinear')
        co_hapi_int = hapi_int(lambda_pRT)
        

        print(len(co_hapi[co_hapi>0.0]))
        print(len(co_hapi_int[co_hapi_int>0.0]))

        
        
        
#         # ######################################
#         # ### OPEN HAPI X-SEC (HITRAN) #########
#         # ######################################
        
         
        co_hapi_dint = hapi_int(lambda_hapi)
        ratio_co = (co_hapi_dint-co_hapi)/co_hapi * 100
        
     
        if (plot_print):
            
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
            plt.rcParams['agg.path.chunksize']=101
            
            # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=25)     # fontsize of the axes title
            plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
            plt.rc('legend', fontsize=15)    # legend fontsize
            plt.rc('figure', titlesize=25)  # fontsize of the figure title
            
            resolution_pnnl = 0.015
            
            nu_start =  12000.0
            nu_end   =  16000.0
            
            y_start = 1.0e-33
            y_end   = 1.0e-19
            
            title_band=''
            figure1 = plt.figure(figsize=(24,12),dpi=500)
            ax1 = figure1.add_subplot(111) #(211)
#            ax2 = figure1.add_subplot(212, sharex=ax1)
            
            ax1.set_xlim(9e-5,4e-3)
            ax1.set_ylim(y_start,y_end)
            secax = ax1.secondary_xaxis('top', functions=(cm2mum,cm2mum))
            secax.set_xlabel(r'cm$^{-1}$')
            ax1.set_xlabel(r'$cm$')
            
            ax1.set_title(title_band, y=1.05)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
            ax1.scatter(lambda_hapi,co_hapi,label='HAPI',s=4.0,marker='x',color=(0.9,0.1,0.1))
            ax1.plot(lambda_pRT,co_hapi_int,label='Interpolated',alpha=0.5)
            ax1.scatter(lambda_pRT, np.linspace((y_start*y_end)**0.5,(y_start*y_end)**0.5,len(lambda_pRT) ), s=5.0,marker='+', color=(0.1,0.9,0.1), label='pRT grid')
#            ax1.scatter(lambda_pRT,co_sent/N_a*mass_amu,label='Sent',marker='+',s=150.)
            ax1.legend()
            
#            ax2.scatter(lambda_hapi, ratio_co)

            name_img = "./hydra_ver/images/N2O_pRT_vs_HAPI_%4.2e_%8.2f.voigt.7000k.25wing.png"%(p_arr[ind_pres]*1.01325,t_arr[ind_temp])
            plt.savefig(name_img,bbox_inches='tight', transparent=False)
            plt.close()
            
            
            
            
#         if (plot_pict):
        
#             for ind_plot in np.arange(2):
#                 figure1 = plt.figure(figsize=(24,12),dpi=700)
#                 ax1 = figure1.add_subplot(211)
#                 ax2 = figure1.add_subplot(212, sharex=ax1)
                
#                 ax1.set_title(title_band, y=1.05)
#                 #ax.set_xscale('log')
#                 if (axYlog):
#                     ax1.set_yscale('log')
#                     # ax2.set_yscale('log')
#                     Scale = 'log'
#                 #    ax2.set_yscale('log')
#                 else:
#                     Scale = 'lin'
                
#                 if (ind_plot==1):
#                     nu_start = 2107.2
#                     nu_end   = 2107.6
#                     ax2.set_ylim(-10,10)
#                     # y_start = 1e-30
#                 else:
#                     ax2.set_ylim(-100,100)                                          # 
            
            
            
            
            
#                 ax1.set_xlim(nu_start, nu_end)
#                 ax1.set_ylim(y_start,y_end)
#             #    ax2.set_ylim(y_start,y_end)
                
                
                
#                 secax = ax1.secondary_xaxis('top', functions=(cm2mum,cm2mum))
#                 secax.set_xlabel(r'$\mu m$')
                
#                 ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
#                 ax1.set_ylabel('Cross-section, cm$^2$/molecule')
                
#                 ax1.plot(nu_pRT,co_pRT, label='petitRADTRANS')#,linewidth=2.5)
#                 ax1.plot(nu_hapi,co_hapi, label='HAPI')#linewidth=0.7, alpha=0.5)
#                 ax1.plot(nu_pRT, co_hapi_int, label='Extrapolated HAPI')
#                 # ax1.plot(nu_hapi_hit,co_hapi_hit, label='HAPI (HITRAN)',alpha=0.4)#linewidth=0.7, alpha=0.5)
            
                
#                 ax2.plot(nu_ratio_strong, co_ratio_strong, label='(pRT-HAPI)/pRT*100\%')
#                 ax2.plot(nu_ratio_weak, co_ratio_weak, label='(pRT-HAPI)/pRT*100\%',alpha=0.3)
            
#                 # ax2.plot(nu_hapi_hit,(co_hapi_hit-co_hapi)/co_hapi_hit*100., label='(HITRAN-HAPI)/HITRAN*100\%',alpha=0.5)
                
#                 # ax2.plot(nu_pRT,abs(co_pRT-co_hapi_int), label='$|$pRT-HAPI$|$')
            
#                 # ax2.plot(nu_hapi_hit,abs(co_hapi_hit-co_hapi), label='$|$HITRAN-HAPI$|$',alpha=0.5)
                
#                 ax1.text(nu_start,(y_start*y_end)**0.5,'p=%4.2e bar, T=%6.2f K'%(p_arr[ind_pres]*1.01325, t_arr[ind_temp]),fontsize=25)
#                 ax2.text(nu_start,0.0,'strong/weak threshold is %5.2e'%(strong_delimit),fontsize=25)
            
#                 ax1.legend()
                
#                 ax2.legend()
                
#                 name_img = "./images/CO_pRT_vs_HAPI_%d_%4.2e_%8.2f.voigt.%3s.600k.100wing.jpg"%(ind_plot,p_arr[ind_pres]*1.01325,t_arr[ind_temp],Scale)
#                 #plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
#                 plt.savefig(name_img,bbox_inches='tight')
#                 plt.close()
        
        co_hapi_pRT = co_hapi_int * N_a / mass_amu / abundance
        
        # OPACITY   = X_SECT      * AVO / MOL MASS / 100% OF ABUNDNCE
        
        # co_hapi_pRT = np.flip(co_hapi_pRT)
        
        co_hapi_pRT.tofile('./hydra_ver/datafiles/SO2_HITRAN20_pRT/sigma_09_%3.0f.K_%8.6fbar.dat'%(t_arr[ind_temp], p_arr[ind_pres]*1.01325))

        hapi_int = []

# nu_pRT = np.flip(nu_pRT)

# nu_pRT = 1.0e0/nu_pRT

# nu_pRT.tofile('./datafiles/O2_HAPI_pRT/wlen.dat')
















    
f.close()
