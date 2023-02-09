if __name__ == "__main__":
    print('XMASSSECTION: program to calculate and store cross-sections in HDF5 files.')

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
#############################################################
# TO DO #####################################################
#############################################################
# FULL NUMBER OF PARAMS #####################################
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
def OpenHDF5(fname,params,pres, temp, vms, Np, Nt, Nvms):
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

        if (FLAG_DEBUG_PRINT):
            print('*** DEBUG: Attributes ***')
            [print(item, f.attrs[item]) for item in f.attrs.keys()]
            print('*** END: Attributes ***\n')
        # f.create_dataset('Gas_05_Absorption',shape=)

        ds_index = f.create_dataset('Gas_Index',data=INDEX_TEMP)
        ds_pres = f.create_dataset('Pressure',data=pres)
        ds_temp = f.create_dataset('Temperature',data=temp)
        ds_vms = f.create_dataset('Broadener_01_VMS',data=vms)
        ds_vms.attrs.__setitem__('broadener_name','h2o')
        ds_Qbrd = f.create_dataset('Broadener_Index',data=INDEX_Qbrd_TEMP)
#        print(ds_pres[()])
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

# calculate x-sec for exact P, T, VMS of exact molecule
def CalculateXsec():
    
    
    
    
    
    
    
    
    
    return








# flag to print values in functions to debug 
FLAG_DEBUG_PRINT = True
# flag to store prints in OUTPUT.LOG file
FLAG_LOG_FILE = True
# flag to open\close HDF5 file 
FLAG_OPENED_HDF5 = False


if not os.path.exists("./images"):
    os.mkdir("./images")

if not os.path.exists("./datafiles"):
    os.mkdir("./datafiles")

#############################################################
### BEGIN OF MAIN PART ######################################
#############################################################

print("Timer started")
t_begin = time.time()

XMASSSEC_VERSION = '0.2.4'; __version__ = XMASSSEC_VERSION
XMASSSEC_HISTORY = [
'INITIATION OF INPUT FILE WITH PARAMETERS 31.01.23 (ver. 0.1)',
'CREATION OF HDF5 FILE + SOME EXCEPTIONS HANDLING (ver. 0.2)',
'CLOSING HDF5 FILE AND SATURATION OF ATTRIBUTES (ver. 0.2.1)',
'SATURATION OF ATTRIBUTE (ROOT) (ver. 0.2.2)',
'INPUTS FOR P,T AND SATURATION OF ATTRUBUTES (ver. 0.2.3)',
'INPUTS FOR VMS, WN AND SATURATION OF ATTRUBUTES + OUTPUT LOG FILE (ver. 0.2.4)'
]

# version header
print('X-MASS-SEC version: %s'%(XMASSSEC_VERSION))

#####################################################

if (FLAG_LOG_FILE):
    orig_stdout = sys.stdout
    fLog = open('output.log', 'w')
    sys.stdout = fLog
else:
    fLog = open('output.log', 'w')
    fLog.close()



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

db_begin('05_hit20')

print(tableList())

HDF5FileName = 'CO_HDF5.hdf5'

co_hdf5 = OpenHDF5(HDF5FileName, ParametersCalculation, Pressures, Temps, VMSs, Npp, Ntt, Nvms)

#print(co_hdf5['Broadener_01_VMS'].attrs['broadener_name'])



CalculateXsec()


CloseHDF5(co_hdf5)














































if (FLAG_LOG_FILE):
    sys.stdout = orig_stdout
    fLog.close()



t_end = time.time()
print('Time taken: %d seconds'%(t_end-t_begin))

print('Done.')



