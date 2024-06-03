import numpy as np
from hapi import *


wn_begin =      0.0
wn_end =    10000.0

wn_step = 0.01

VMS = 1.0

Temp = 296.15
p = 1.0

CoefFileName = 'CO_HITEMP2019_all.txt'

db_begin('05_hit20')

nu_co,coef_co = absorptionCoefficient_Voigt(SourceTables='05_HITEMP2019',
                                              HITRAN_units=True, OmegaRange=[wn_begin,wn_end],
                                              WavenumberStep=wn_step,
                                              WavenumberWing=25.0,
                                              Diluent={'self':1.00-VMS, 'air':VMS},
                                              Environment={'T':Temp,'p':pres},
                                              File = CoefFileName)




print('Done.')