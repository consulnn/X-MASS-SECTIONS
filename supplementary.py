from main import openParametersFile, openPressure, openTemp, openVMS, openXgenetareWn


# Supplementary file for tests

import math
import os
import sys
#from hapi2 import *
from hapi import *
import matplotlib.pyplot as plt
sys.path.insert(1,'/Work_remote/WORK/scripts')
#sys.path.insert(2,'/work/WORK/WORK/scripts/hapi2')
sys.path.insert(3,'/Work_remote/WORK/scripts/hapiN')
import json
from getpass import getpass
from scipy.interpolate import interp1d



import pylab as pl
import matplotlib as mpl
import time
import glob
import multiprocessing
from multiprocessing import Pool
import argparse
import h5py
import numpy as np

from main import openParametersFile, openPressure, openTemp, openVMS, openXgenetareWn, OpenHDF5, CloseHDF5, ParallelPart

import random

def get_specgrid( R=15000, lambda_min=0.1, lambda_max=20.0):
    #generating wavelength grid with uniform binning in log(lambda)
    #lambda min and max are in microns, R is the spectral resolution
    #R = lambda/delta_lamdba
    specgrid = []
    delta_lambda =[]
    specgrid.append(lambda_min)
    run = True
    i=1
    while run:
        dlam= specgrid[i-1]/R
        specgrid.append(specgrid[i-1]+dlam)
        delta_lambda.append(dlam)

        if specgrid[i] >= lambda_max:
            run=False
        i+=1
    return np.asarray(specgrid),np.asarray(delta_lambda)



samples = np.array([-0.964260871,
-0.81699983,
-0.572979169,
-0.265091178,
0.065091178,
0.372979169,
0.61699983,
0.764260871,
0.803971014,
0.820333352,
0.847446759,
0.881656536,
0.918343464,
0.952553241,
0.979666648,
0.996028986])

ngauss=16

weights = np.array([0.091105683,
0.200142931,
0.282335981,
0.326415405,
0.326415405,
0.282335981,
0.200142931,
0.091105683,
0.010122854,
0.022238103,
0.031370665,
0.036268378,
0.036268378,
0.031370665,
0.022238103,
0.010122854])

gauss = (samples,weights)
print('Gauss[0]=',gauss[0])

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

resol = 999.5


ParametersCalculation = openParametersFile(INPUT_FILENAME)

Pressures, Np = openPressure(PRES_FILENAME)

(Temps, Npp,Ntt) = openTemp(TEMP_FILENAME, Np)

(VMSs, Nvms) = openVMS(VMS_FILENAME)

WNs, Nwn = openXgenetareWn(WN_FILENAME,ParametersCalculation)

#f = h5py.File('D:\O3_HDF5.hdf5', mode='r')

wl_grid = get_specgrid(resol, 3.00033541, 5.29744147)[0] # build bin grid in wavelength
wn_grid = np.sort(10000./wl_grid) # convert to wavenumber and sort
bincentres = np.sort(10000./(0.5*(wl_grid[1:] + wl_grid[:-1]))) # get the bin centres in wl space
lambdamin = np.min(wl_grid)
lambdamax = np.max(wl_grid)





f = h5py.File('CO_HITEMP_3.0_to_5.3_num_H20.hdf5', mode='r')

print(f.keys())
p_arr = f['Pressure'][()]
t_arr = f['Temperature'][()][11]

# pressure units to bar:
    
p_arr = p_arr * 1.01325


nu_my = f['Wavenumber'][()]

# num_dens = press_val * 1e6 / (1.38e-23 * temp_val) 

lambda_my = 1./nu_my * 1.0e4
lambda_my = np.sort(lambda_my)

#print(lambda_my)


lambda_new, delta_new = get_specgrid(resol,3.00033541,5.29744147)


#print('***',np.min(lambda_my),np.max(lambda_my)-0.00)

nu_new = 1.0e+4/lambda_new

nu_new = np.sort(nu_new)

kcoeff = np.zeros((len(p_arr), len(t_arr), len(bincentres), ngauss))


for ipress, press_val in np.ndenumerate(p_arr):
    for itemp, temp_val in np.ndenumerate(t_arr):
# press_val = p_arr[11]
# temp_val = t_arr[11]

#        print(press_val,temp_val,'********')

#        print(f['Gas_05_Absorption'][()][ipress][itemp][0].shape)

        co_my = f['Gas_05_Absorption'][()][ipress][itemp][0]


        # print(nu_my)
        
        # print(nu_new)
        
        
        f_co_new = interp1d(nu_my, co_my, kind='linear')
        
        co_new = f_co_new(nu_new)
        
        #print(lambda_new)
        
        ktable_file = '%s_T%i_P%.4e_%s.ktable.h5' % ('CO-12-14', temp_val, press_val,
                                                             'testing_k')
        
        
        
        
        xsec_in = np.zeros((len(nu_my),2))
        xsec_in[:,0] = nu_my
        xsec_in[:,1] = co_my
        

        bingrid_idx = np.digitize(xsec_in[:,0], wn_grid) # get the indexes for bins

        ktable = np.zeros((len(wn_grid)-1, ngauss))
        for i in range(1, len(wn_grid)):
        
            x = xsec_in[:,0][bingrid_idx == i]
            y = xsec_in[:,1][bingrid_idx == i]
            # print('x=',len(x))
            if len(x) > 0:
        
                # reinterpolate to finer grid
                x_new = np.linspace(np.min(x), np.max(x), len(x)*10)
                y_new = np.interp(x_new, x, y)
        
                sort_bin = np.sort(y_new)
                norm_x = ((x_new-np.min(x_new))/(np.max(x_new) - np.min(x_new)))*2 - 1
                tkcoeff = np.interp(gauss[0], norm_x, sort_bin)
                ktable[i-1,:]  = tkcoeff
#            print(ktable)
        kcoeff[ipress, itemp, :, :] = ktable[:,:]
        print(ipress, itemp)



print(kcoeff[kcoeff>1e-20])




hdf5_out = h5py.File('D:\Work_remote\WORK\X-MASS-SECTIONS\datafiles\%s.R%d_3.0-5.6mu.ktable.petitRADTRANS.h5' %('05',resol),'w')

mass_mol  = 28




# str_type = h5py.new_vlen(str)
str_type = h5py.string_dtype(encoding='utf-8')


dset = hdf5_out.create_dataset("method",(1,), dtype=str_type)
dset2 = hdf5_out.create_dataset("mol_mass",(1,), dtype=int)
dset3 = hdf5_out.create_dataset("DOI",(1,), dtype=str_type)
dset4 = hdf5_out.create_dataset("Date_ID",(1,), dtype=str_type)
dset5 = hdf5_out.create_dataset("mol_name",(1,), dtype=str_type)

hdf5_out['bin_centers'] = bincentres
hdf5_out['bin_edges'] = wn_grid
hdf5_out['wlrange'] = (lambdamin, lambdamax)
hdf5_out['wnrange'] = (10000./lambdamax, 10000./lambdamin)
hdf5_out['weights'] = weights/2.
hdf5_out['samples'] = (samples+1.)/2.
#ngauss and method don't output
hdf5_out['ngauss'] = 16
#hdf5_out['method'] = 'polynomial.legendre.leggauss'
dset[0] = 'petit_samples'

### petitRT
hdf5_out['kcoeff'] = kcoeff
hdf5_out['kcoeff'].attrs['units'] = 'cm^2/molecule'
### TauREX?
# hdf5_out['xsecarr'] = kcoeff
# hdf5_out['xsecarr'].attrs['units'] = 'cm^2/molecule'


hdf5_out['t'] = np.sort(t_arr).astype(float)
hdf5_out['p'] = np.sort(p_arr).astype(float)
hdf5_out['p'].attrs['units'] = 'bar'
dset2[0] = 28
dset3[0] = '10.1010'
dset4[0] = 'v1_200520'
dset5[0] = 'CO'


hdf5_out.close()












f_exo = h5py.File('CO_HDF5_3.0_to_5.3_num_EXO.h5', mode='w')

ds_p = f_exo.create_dataset('p',data=p_arr)
ds_t = f_exo.create_dataset('t',data=t_arr)



f_hcn = h5py.File('./datafiles/12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5', mode='r')

#print(f_hcn['kcoeff'][()][18][5])
kcoeff_temp = f_hcn['kcoeff'][()][8][5]
# kcoeff_temp1 = kcoeff_temp[:,1]
# kcoeff_temp2 = kcoeff_temp[:,2]
# kcoeff_temp3 = kcoeff_temp[:,3]
# kcoeff_temp4 = kcoeff_temp[:,4]
# kcoeff_temp5 = kcoeff_temp[:,5]
# kcoeff_temp6 = kcoeff_temp[:,6]
# kcoeff_temp7 = kcoeff_temp[:,7]
# kcoeff_temp8 = kcoeff_temp[:,8]
# kcoeff_temp9 = kcoeff_temp[:,9]
# kcoeff_temp10 = kcoeff_temp[:,10]
# kcoeff_temp11 = kcoeff_temp[:,11]
# kcoeff_temp12 = kcoeff_temp[:,12]
# kcoeff_temp13 = kcoeff_temp[:,13]
# kcoeff_temp14 = kcoeff_temp[:,14]
# kcoeff_temp15 = kcoeff_temp[:,15]




print('p=',f_hcn['p'][8][()],', t=',f_hcn['t'][5][()])
# 11 11


#print(kcoeff_temp.shape)
nu_h5 = f_hcn['bin_centers'][()]


pRT_CS = np.fromfile("./datafiles/sigma_05_81.K_0.000001bar.dat", dtype=np.float16)

# print('***')
# print(pRT_CS[:20:1])
# print('***')    
coef_hdf5 = f['Gas_05_Absorption'][()][18][5][0]
nu_hdf5 = f['Wavenumber'][()]

#coef_h5 = f_hcn['kcoeff'][()][18][5][:][0]

f_exo.close()

print('***')
print(wl_grid[:10])
print(wl_grid[-10:])

nu_h5_e = f_hcn['bin_edges'][()]
lambda_h5 = np.sort(1e+4/nu_h5_e)
lambda_h5 = lambda_h5[lambda_h5>3.0003]
lambda_h5 = lambda_h5[lambda_h5<5.3003]

print(lambda_h5[:10])
print(lambda_h5[-10:])
print('***')





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

nu_start =  2150.0
nu_end   =  2175.0

y_start = 1.0e-30
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

def cm2mum(x):
    return 10000./x



secax = ax1.secondary_xaxis('top', functions=(cm2mum,cm2mum))
secax.set_xlabel('mum')



ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
ax1.set_ylabel('Cross-section, cm$^2$/molecule')
#ax1.text(1900,1e-19,'%2d %2d %2d'%(18,5,1.0))
#ax2.set_ylim(-1e-25,1e-25)
#ax1.plot(nu_hdf5, coef_h5, label=r'CO HITRAN spectra, 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.1,0.1,0.9),linewidth=1.0)
#ax1.plot(nu_hdf5, coef_hdf5, label=r'CO HITRAN spectra (HDF5), 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.9,0.1,0.1),linewidth=1.0)

# ax1.scatter(nu_h5,kcoeff_temp1,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp2,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp3,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp4,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp5,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp6,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp7,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp8,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp9,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp10,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp11,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp12,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp13,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp14,s=1.4)
# ax1.scatter(nu_h5,kcoeff_temp15,s=1.4)

ax1.plot(nu_my, co_my,ms=0.05, alpha=0.3, label='Cross-section of calcs')
ax1.scatter(nu_new,np.linspace(1e-17,1e-17,len(nu_new)),label='Bins with R=%d'%resol)      
ax1.scatter(nu_my,np.linspace(1e-16,1e-16,len(nu_my)),label='Bins of calculations')      
   

ax2.plot(nu_my, co_my,ms=0.2, alpha=0.5, label='Cross-section of calcs')
ax2.plot(nu_new, co_new,ms=0.5, alpha=0.5,color=(0.9,0.1,0.1), label='Cross-section with R=%d'%resol)


ax1.legend()

ax2.legend()
#####
# K-tables

for i in np.arange(16):
    ax1.scatter(nu_h5,kcoeff_temp[:,i])

for i in np.arange(16):
    ax2.scatter(bincentres, ktable[:,i])
 
#####


       
#ax2.plot(nu_hdf5, coef_h5-coef_hdf5)

name_img = "./images/ExoMolOp_vs_HAPI.jpg"
#plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
plt.savefig(name_img,bbox_inches='tight')
plt.close()




# f_mersedes = open('./datafiles/sigma_05_81.K_0.000001bar.dat', mode='rb')
# text = f_mersedes.read().decode('utf8',errors='ignore')
# print(text[:100])

# f_dace = open('./datafiles/1H-13C-14N__Larner_e2b/Out_00000_18000_00050_n033.bin', mode='rb')
# text = f_dace.read().decode('ascii',errors='ignore')
# print(text[:100])

# f_hcn = h5py.File('./datafiles/1H-13C-14N__Larner_e2b/Out_00000_18000_00050_n033.bin', mode='r')
# print(f_hcn.keys())

# import pandas as pd
# dataset = pd.read_csv(r'./datafiles/sigma_05_81.K_0.000001bar.dat', sep='delimiter', header=None, on_bad_lines='skip',encoding= 'cp1252')

# import pandas as pd
# data=pd.read_csv('./datafiles/sigma_05_81.K_0.000001bar.dat', engine='c',encoding='latin1')
# print(data.head())





#print(f.keys())

# for i in np.arange(15):
#     p_rand = random.randrange(0,Npp)
#     T_rand = random.randrange(0,Ntt)
#     VMS_rand = random.randrange(0,Nvms)
    
    
#     coef_hdf5 = f['Gas_05_Absorption'][()][p_rand][T_rand][VMS_rand]
#     nu_hdf5 = f['Wavenumber'][()]
    
#     print(nu_hdf5, coef_hdf5)
    
    
    
#     db_begin('05_hit20')
    
#     wn_begin = 1886.3
#     wn_end = 3333.3
    
#     Nwn = 144701
    
#     print(p_rand,T_rand,VMS_rand)
#     print(Pressures[p_rand],Temps[p_rand,T_rand],VMSs[VMS_rand])
    
#     p_item = Pressures[p_rand]
#     T_item = Temps[p_rand,T_rand]
#     VMS_item = VMSs[VMS_rand]
    
#     pres = p_item
#     Temp = T_item
#     VMS = VMS_item
    
    
#     wn_step = (wn_end-wn_begin)/(Nwn-1)
    
    
#     nu_co,coef_co = absorptionCoefficient_Voigt(SourceTables='COall',
#                                                   HITRAN_units=True, OmegaRange=[wn_begin,wn_end],
#                                                   WavenumberStep=wn_step,
#                                                   WavenumberWing=25.0,
#                                                   Diluent={'self':1.00-VMS, 'H2O':VMS},
#                                                   Environment={'T':Temp,'p':pres},
#                                                   File = './datafiles/test_4_pict.dat')
     

    
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "font.sans-serif": ["Helvetica"]})
#     ## for Palatino and other serif fonts use:
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Palatino"],
#     })
    
    
#     # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=30)     # fontsize of the axes title
#     plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=15)    # legend fontsize
#     plt.rc('figure', titlesize=30)  # fontsize of the figure title
    
#     resolution_pnnl = 0.015
    
#     nu_start =  1830.0
#     nu_end   =  3338.0
    
#     y_start = 1.0e-30
#     y_end   = 1.0e-17
    
#     axYlog = True
    
#     pict_open = False
    
    
#     title_band = r'CO test spectra'
    
#     figure1 = plt.figure(figsize=(24,12),dpi=700)
#     ax1 = figure1.add_subplot(211)
#     ax2 = figure1.add_subplot(212, sharex=ax1)
    
#     ax1.set_title(title_band, y=1.05)
#     #ax.set_xscale('log')
#     if (axYlog):
#         ax1.set_yscale('log')
#         ax2.set_yscale('log')
    
#     ax1.set_xlim(nu_start, nu_end)
#     ax1.set_ylim(y_start,y_end)
#     ax2.set_ylim(y_start,y_end)

#     ax1.set_xlabel(r'Wavenumber, cm$^{-1}$')
#     ax1.set_ylabel('Cross-section, cm$^2$/molecule')
#     ax1.text(1900,1e-19,'%2d %2d %2d'%(p_rand,T_rand,VMS_rand))
#     #ax2.set_ylim(-1e-25,1e-25)
#     ax1.plot(nu_co, coef_co, label=r'CO HITRAN spectra, 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.1,0.1,0.9),linewidth=1.0)
#     ax1.plot(nu_hdf5, coef_hdf5, label=r'CO HITRAN spectra (HDF5), 1.0 water, T=%6.2f K, %4.2f atm)'%(Temp,pres),alpha=0.5,color=(0.9,0.1,0.1),linewidth=1.0)
    
#     ax2.plot(nu_co, coef_co-coef_hdf5)
    
#     ax1.legend()
#     name_img = "./images/cross_test_calc_vs_HDF5_XSEC_CO_%02d.jpg"%(i)
#     #plt.savefig('./images/absorb_report_2nu8_hcn.jpg',bbox_inches='tight')
#     plt.savefig(name_img,bbox_inches='tight')
#     plt.close()
    
    
#     dtype1 = np.dtype([('nu','float'),('coef','float')])
    
#     if (pict_open):
#         os.system(name_img)


f.close()