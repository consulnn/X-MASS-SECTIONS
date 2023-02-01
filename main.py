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

XMASSSEC_VERSION = '0.0.1'; __version__ = XMASSSEC_VERSION
XMASSSEC_HISTORY = [
'INITIATION OF INPUT FILE WITH PARAMETERS 31.01.23 (ver. 0.1)'
]

# version header
print('X-MASS-SEC version: %s'%(XMASSSEC_VERSION))

INPUT_FILENAME = 'params.inp'

# opens file with necessary parameters for calculations
def openParametersFile(fname):
    class ParamsError(Exception):
        pass
    try:
        with open(fname,'r') as finp:
            # input as list of lines in 'lines'
            lines = finp.readlines()
            # check if input has enough parameters
            if (len(lines)<6):
                raise ParamsError
            # removing '\n'
            lines = [item.rstrip('\n') for item in lines]
            # splitting names and values
            params = [item.split(':') for item in lines]
            print(params)
    except FileNotFoundError:
        print('%s file is not found!'%fname)
    except ParamsError:
        print("Not enought parameters in %s!"%fname)
    return


openParametersFile(INPUT_FILENAME)



#############################################################
# TO DO #####################################################
#############################################################
# FULL NUMBER OF PARAMS #####################################
#############################################################



















if not os.path.exists("./images"):
    os.mkdir("./images")

if not os.path.exists("./datafiles"):
    os.mkdir("./datafiles")

t_begin = time.time()






t_end = time.time()

print('Time taken: %d seconds'%(t_end-t_begin))
print('Done.')



