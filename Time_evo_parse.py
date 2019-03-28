#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:27:31 2017

@author: wcole
"""

import numpy as np
import TD_analysis as TDA
import time_evo_tunham as TE


#Want a script that can run batches
#the input will be txt files with all the parameters to run
#a seperate text file will contain the values for a given hamiltonian

#this should be of length 4
namepara=np.loadtxt('batch_names.txt',dtype=bytes,delimiter='\t').astype(str)
#remember to cast the parameter to number as necessary

#this should be of length 1
valuepara=np.loadtxt('batch_value.txt').astype(float)

for i in range(len(namepara)):
    TE.extMain(namepara[i,0],valuepara[i,:],int(namepara[i,1]),int(namepara[i,2]),namepara[i,3])
    TDA.extMain(namepara[i,3],int(namepara[i,2]))