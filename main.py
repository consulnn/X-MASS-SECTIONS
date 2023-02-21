if __name__ == "__main__":
    print('\nXMASSSECTION: program to calculate and store cross-sections in HDF5 files.\n')

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
from joblib import Parallel, delayed

import pylab as pl
import matplotlib as mpl
import time
import h5py
#############################################################
# TO DO #####################################################
#############################################################
# TRY/EXCEPT IN XSEC ########################################
#############################################################




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

#############################################################
### FUNCTIONS ###############################################
#############################################################

# opens file with necessary parameters for calculations
def openParametersFile(fname):
    class ParamsError(Exception):
        pass
    try:
        with open(fname,'r') as finp:
            global FLAG_DEBUG_PRINT
            # input as list of lines in 'lines'
            lines = finp.readlines()
            # check if input has enough parameters
            if (len(lines)<17):
                raise ParamsError
            # removing '\n'
            lines = [item.rstrip('\n') for item in lines]
            # splitting names and values
            params = [item.split(':') for item in lines]
            if (FLAG_DEBUG_PRINT):
                print('*** DEBUG: Parameters input ***')
                [print(item[0],'\t',item[1]) for item in params]
                print('*** END: Parameters input ***\n')
        print('*********\nParameters file %s is opened well\n*********'%(fname))
        return params
    except FileNotFoundError:
        print('%s file is not found!'%fname)
        sys.exit()
        # raise FileNotFoundError
    except ParamsError:
        print("Not enought parameters in %s!"%fname)
        sys.exit()
    except Exception as err:
        print('WOW, UNKNOWN ERROR: %s'%(err))
        sys.exit()

# opens pressure array file
def openPressure(fname):
    try:
        apres = np.genfromtxt(fname,dtype='float')
        global FLAG_DEBUG_PRINT
        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: Pressure input ***')
            [print('%12.6f'%(item)) for item in apres]
            print('*** END: Pressure input ***\n')
        print('*********\nPressure file %s is opened well\nTotal number of lines: %d\n*********'%(fname,len(apres)))
        return apres, len(apres)
    except FileNotFoundError:
        print('%s file is not found!'%fname)
        sys.exit()
    
# opens temperature array file
def openTemp(fname,Np):
    class PxTError(Exception):
        'Corrupted relations between Np and NpxNt array'
        pass
    try:
        atemp = np.genfromtxt(fname,dtype='float', missing_values='296.15')
        (Npp, Ntt) = atemp.shape
        if (Npp!=Np):
            raise PxTError
        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: Temperature input ***')
            for item in atemp:
                [print(item1, end='\t') for item1 in item]
                print('')
            print('*** END: Temperature input ***\n')
        print('*********\nTemperature file %s is opened well\nTotal number of lines: %d\nNo of temperatures: %d\n*********'%(fname,Npp,Ntt))
        return atemp, Npp, Ntt
    except FileNotFoundError:
        print('%s file is not found!'%fname)
        sys.exit()
    except PxTError:
        print('Corrupted relations between Np and NpxNt array')
        sys.exit()

# opens pressure array file
def openVMS(fname):
    try:
        avms = np.genfromtxt(fname,dtype='float')
        global FLAG_DEBUG_PRINT
        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: VMS input ***')
            [print('%12.6f'%(item)) for item in avms]
            print('*** END: VMS input ***\n')
        print('*********\nVMS file %s is opened well\nTotal number of lines: %d\n*********'%(fname,len(avms)))
        return avms, len(avms)
    except FileNotFoundError:
        print('%s file is not found!'%fname)
        sys.exit()

def openXgenetareWn(fname,params):
    try:
        with open(fname) as f:
            Nwn = int(f.readline())
            wn_begin = float(params[3][1])
            wn_end = float(params[4][1])
            WN_range = np.linspace(wn_begin,wn_end,Nwn)
#             print(WN_range[:15])
            if (FLAG_DEBUG_PRINT):
                print('*** DEBUG: WN input ***')
                [print('%12.6f'%item1) for item1 in WN_range[:10]]
                print('...')
                [print('%12.6f'%item2) for item2 in WN_range[-10:]]
                print(Nwn)
                print('*** END: WN input ***\n')
            print('*********\nWN file %s is opened well\nTotal number of wn-points: %d\n*********'%(fname,Nwn))
                
            return WN_range, Nwn
    except FileNotFoundError:
        print('%s file is not found!'%fname)
        sys.exit()


# creates a file HDF5 and saturate it with attributes
def OpenHDF5(fname,params,pres, temp, vms, wns, Np, Nt, Nvms, Nwn):
    global FLAG_DEBUG_PRINT
    INDEX_TEMP = '%02d'%(5)
    INDEX_Qbrd_TEMP = '%02d'%(1)
#    print(INDEX_TEMP)
    try:
        f = h5py.File(fname, mode='w')
        global FLAG_OPENED_HDF5 
        FLAG_OPENED_HDF5 = True
        
        print('*********\nHDF5 file %s is opened well\n*********'%(fname))
        
# saturating the attributes        
        [f.attrs.__setitem__(item[0],item[1]) for item in params[:6] ]

        Index_abs = '%02d'%(int(params[10][1]))
        Index_broad = '%02d'%(int(params[15][1]))
        dataset_name = 'Gas_'+Index_abs+'_Absorption'
        dataset_broadname = 'Broadener_'+Index_broad+'_VMS'
        ds_coef = f.create_dataset(dataset_name,shape=(Np,Nt,Nvms,Nwn))#, compression="gzip", compression_opts=4)
        ds_coef.attrs.__setitem__('addl_ident', '')
        ds_coef.attrs.__setitem__('gas_name', 'CO')
        ds_coef.attrs.__setitem__('comment', '')

        ds_index = f.create_dataset('Gas_Index',data=INDEX_TEMP)
        ds_pres = f.create_dataset('Pressure',data=pres)
        ds_temp = f.create_dataset('Temperature',data=temp)
        ds_vms = f.create_dataset(dataset_broadname,data=vms)
        ds_vms.attrs.__setitem__('broadener_name','h2o')
        ds_Qbrd = f.create_dataset('Broadener_Index',data=INDEX_Qbrd_TEMP)
        
        ds_wns = f.create_dataset('Wavenumber',data=wns)
        
#        print(ds_pres[()])
        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: Attributes ***')
            [print(item, f.attrs[item]) for item in f.attrs.keys()]
            print(f.keys())
            [print(f[item]) for item in f.keys()]
            print('*** END: Attributes ***\n')
        return f
    except FileExistsError:
        print('Attempt to re-write file!')
        sys.exit()
    else:
        err = Exception
        print("Unexpected %s"%(err))
        sys.exit()

# closes the HDF5 file 
def CloseHDF5(ftype):
    try:
        global FLAG_OPENED_HDF5 
        if ((FLAG_OPENED_HDF5 != True) or (ftype.__repr__()=='<Closed HDF5 file>')):
            raise FileNotFoundError
        else:
            ftype.close()
            FLAG_OPENED_HDF5 = False
            return
    except FileNotFoundError:
        print('File to close is not found or already closed')
        sys.exit()
    else:
        err = Exception
        print("Unexpected %s"%(err))
        sys.exit()

def SaveHDF5(ftype, p,t,vms,coef_):
    try:
        global FLAG_OPENED_HDF5 
        if ((FLAG_OPENED_HDF5 != True) or (ftype.__repr__()=='<Closed HDF5 file>')):
            raise FileNotFoundError
        else:
            

            return
    except FileNotFoundError:
        print('File to work with is not found or already closed')
        sys.exit()
    else:
        err = Exception
        print("Unexpected %s"%(err))
        sys.exit()
    

# calculate x-sec for exact P, T, VMS of exact molecule
def CalculateXsec(pres, Temp, VMS, WN_range, IndexMol, IndexBroad, param, Nwn):
    class NaNError(Exception):
        'Corrupted p, T or VMS value'
        pass
    
    try:
        if ((pres!=pres) or (Temp!=Temp) or (VMS!=VMS)):
            raise NaNError
        
        db_begin('05_hit20')
        
        wn_begin = float(param[3][1])
        wn_end = float(param[4][1])
    
        wn_step = (wn_end-wn_begin)/(Nwn-1)
    
        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: X-sec ***')
            print('VMS=%4.2f, type='%VMS, type(VMS))
    #        print(tableList())
            print('Range from %8.2f to %8.2f, step %6.2f'%(wn_begin, wn_end,wn_step))
            print('Pressure=%6.2f, temperature=%7.2f'%(pres,Temp))
            print('*** END: X-sec ***\n')
    
        CoefFileName = './datafiles/%06.2fT_Id%02d_%06.4fatm_IdBroad%02d_%06.4fVMS_hit20.dat'%(Temp,IndexMol,pres,IndexBroad,VMS)
    
        nu_co,coef_co = absorptionCoefficient_Voigt(SourceTables='COall',
                                                     HITRAN_units=True, OmegaRange=[wn_begin,wn_end],
                                                     WavenumberStep=wn_step,
                                                     WavenumberWing=25.0,
                                                     Diluent={'self':1.00-VMS, 'H2O':VMS},
                                                     Environment={'T':Temp,'p':pres},
                                                     File = CoefFileName)
        if (FLAG_REMOVE_HAPI):
            os.remove(CoefFileName)
        
        return coef_co
    except NaNError:
        print('Corrupted p, T or VMS value')
        return np.linspace(-1.0,-1.0,Nwn)
    else:
        err = Exception
        print("Unexpected %s"%(err))
        sys.exit()
        
        

def UpdateHDF5(ftype, co_array, i_p, i_t, i_vms):
    Index_abs = '%02d'%(int(ftype['Gas_Index'][()]))
    dataset_name = 'Gas_'+Index_abs+'_Absorption'
    set_abs = ftype[dataset_name][()]
    Nwn = len(co_array)
    set_abs[i_p][i_t][i_vms][:] = co_array
    ftype[dataset_name][()] = set_abs
    return ftype

def ParallelPart(Pressures, Temps, VMSs,WNs,ParametersCalculation,Nwn, ip, it, ivms,co_hdf5):
    ptemp = Pressures[ip]
    ttemp = Temps[ip,it]
    vmstemp = VMSs[ivms]
    print(ptemp,ttemp,vmstemp)
    
    coeffs = CalculateXsec(ptemp, ttemp,vmstemp,WNs,5,1,ParametersCalculation,Nwn)
    co_hdf5 = UpdateHDF5(co_hdf5, coeffs, ip, it, ivms)
    return co_hdf5


# flag to print values in functions to debug 
FLAG_DEBUG_PRINT = True
# flag to store prints in OUTPUT.LOG file
FLAG_LOG_FILE = True
# flag to open\close HDF5 file 
FLAG_OPENED_HDF5 = False
# flag to remove HAPI files
FLAG_REMOVE_HAPI = False

if not os.path.exists("./images"):
    os.mkdir("./images")

if not os.path.exists("./datafiles"):
    os.mkdir("./datafiles")
fLog = open('output.log', 'w')
fLog.close()
if (FLAG_LOG_FILE):
    orig_stdout = sys.stdout
    fLog = open('output.log', 'a')
    sys.stdout = fLog


#############################################################
### BEGIN OF MAIN PART ######################################
#############################################################
XMASSSEC_VERSION = '0.4'; __version__ = XMASSSEC_VERSION
XMASSSEC_HISTORY = [
'INITIATION OF INPUT FILE WITH PARAMETERS 31.01.23 (ver. 0.1)',
'CREATION OF HDF5 FILE + SOME EXCEPTIONS HANDLING (ver. 0.2)',
'CLOSING HDF5 FILE AND SATURATION OF ATTRIBUTES (ver. 0.2.1)',
'SATURATION OF ATTRIBUTE (ROOT) (ver. 0.2.2)',
'INPUTS FOR P,T AND SATURATION OF ATTRUBUTES (ver. 0.2.3)',
'INPUTS FOR VMS, WN AND SATURATION OF ATTRUBUTES + OUTPUT LOG FILE (ver. 0.2.4)',
'CALCULATING X-SEC (ver. 0.3)',
'CALCULATING X-SEC, DEBUG INFO (ver.0.3.1)',
'FIX: MISSING WN RANGE IN HDF5 FILE (ver. 0.3.2)',
'FIX: NEW EXCEPTION IN X-SEC RELATED TO NAN VALUES OF P,T,VMS (ver. 0.3.3)',
'FIRST ITERATION OF PARALLEL ATTEMPT (ver. 0.4)'
]

# version header
print('X-MASS-SEC version: %s'%(XMASSSEC_VERSION))

#####################################################

print("Timer started")
t_begin = time.time()




INPUT_FILENAME = 'params.inp'

PRES_FILENAME = 'pres.inp'

TEMP_FILENAME = 'temps.inp'

VMS_FILENAME = 'vms.inp'

WN_FILENAME = 'wn.inp'

HDF5FileName = 'CO_HDF5.hdf5'

ParametersCalculation = openParametersFile(INPUT_FILENAME)

Pressures, Np = openPressure(PRES_FILENAME)

(Temps, Npp,Ntt) = openTemp(TEMP_FILENAME, Np)

(VMSs, Nvms) = openVMS(VMS_FILENAME)

WNs, Nwn = openXgenetareWn(WN_FILENAME,ParametersCalculation)

co_hdf5 = OpenHDF5(HDF5FileName, ParametersCalculation, Pressures, Temps, VMSs, WNs, Npp, Ntt, Nvms, Nwn)

for ip in np.arange(Npp):
    for it in np.arange(Ntt):
        for ivms in np.arange(Nvms):
            ptemp = Pressures[ip]
            ttemp = Temps[ip,it]
            vmstemp = VMSs[ivms]
            print(ptemp,ttemp,vmstemp)
            
            
            
            
            
            coeffs = CalculateXsec(ptemp, ttemp,vmstemp,WNs,5,1,ParametersCalculation,Nwn)

            co_hdf5 = UpdateHDF5(co_hdf5, coeffs, ip, it, ivms)
            
            

CloseHDF5(co_hdf5)




t_end = time.time()
print('Time taken: %d seconds'%(t_end-t_begin))

print('\nDone.')

if (FLAG_LOG_FILE):
    sys.stdout = orig_stdout
    fLog.close()







