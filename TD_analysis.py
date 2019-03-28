#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:59:03 2016

@author: wcole
Script to plot the results of the time dep pert calculation for clusters
also fft of result for the timescales
"""
import numpy as np
import scipy.signal as sp
import scipy.fftpack as fft
import matplotlib.pyplot as plt

if __name__ == "__main__":
    name='test'
    timestep='ps'
    stepsize=100/(10**15)
    label='|1>'
    data=np.genfromtxt(name+'.txt',skip_header=1)
    
    #probabaly some grouping up of the data needed for analysis
    #ie for the dimer there are different groups used to reduce the problem to a 
    #2 level system
    
    test=data[:,2]
    #test=data[:,1]+data[:,2]+data[:,3]+data[:,4]
    
    plt.figure()
    plt.plot(data[:,0],test[:],label=label)
    #plt.xlim([0,100])
    plt.xlabel('Time (fs)')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.savefig('Sim_'+name+'.png')
    plt.show()
    #now you will do a fft on the columns, only need to do it on one column
    
    ftData=fft.fft(test)
    #print(ftData)
    freqData=fft.fftfreq(len(ftData), d=stepsize)
    #print(freqData)
    
    indMax=sp.argrelmax(ftData)
    maxList=np.zeros([0,2])
    for i in range(len(ftData)):
        if i in indMax[0] and 0 <= freqData[i] and 10 < ftData[i]:
            dummy=np.zeros([1,2])
            dummy[0,0]=ftData[i]
            #this is in units of fs, we can divide by 1000 to get to ps 
            dummy[0,1]=(1/(freqData[i]))*10**12
            maxList=np.vstack([maxList,dummy])
        
    #maxList=np.sort(maxList)
    print(len(maxList))
    #find the xlim for the plot
    endP=max(maxList[:,1])*2
    #now plot the fft analysis
    plt.figure()
    plt.plot(((1/(freqData)))*10**12,np.abs(ftData))
    plt.xlabel('Hydrogen Bond Lifetime ('+timestep+')')
    plt.xlim([0,endP])
    plt.savefig('FTplot_'+name+'.png')
    plt.show()
    f=open('FT_results_for_'+name+'.txt','w')
    f.write('Amplitude'+'\t'+'Time ('+timestep+')'+'\n')
    for i in range(len(maxList)):
        f.write('%s'%maxList[i,0]+'\t'+'%s'%maxList[i,1]+'\n')
    f.close()
    

#now a function just for running batch calcs
def extMain(fname,timePS):
    """Method to run the main method externally, useful for running batches of calculations in series
        
    Parameters
    ----------
    fname : the name of the file representing the results of the time evolution calculation
    
    timePS : the length of the calculation in picoseconds
    
    Outputs
    ------
    A plot showing the time evolution of the minima and a second plot showing the FFT
        of the calculation
        
    """
    timestep='ps'
    stepsize=timePS/(10**15)
    label='|1>'
    data=np.genfromtxt(fname+'.txt',skip_header=1)

#probabaly some grouping up of the data needed for analysis
#ie for the dimer there are different groups used to reduce the problem to a 
#2 level system

    test=data[:,1]
#test=data[:,1]+data[:,2]+data[:,3]+data[:,4]

    plt.figure()
    plt.plot(data[:,0],test[:],label=label)
#plt.xlim([0,100])
    plt.xlabel('Time (fs)')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.savefig('Sim_'+fname+'.png')
    plt.close()
    #plt.show()
#now you will do a fft on the columns, only need to do it on one column

    ftData=fft.fft(test)
#print(ftData)
    freqData=fft.fftfreq(len(ftData), d=stepsize)
#print(freqData)

    indMax=sp.argrelmax(ftData)
    maxList=np.zeros([0,2])
    for i in range(len(ftData)):
        if i in indMax[0] and 0 <= freqData[i] and 10 < ftData[i]:
            dummy=np.zeros([1,2])
            dummy[0,0]=ftData[i]
        #this is in units of fs, we can divide by 1000 to get to ps 
            dummy[0,1]=(1/(freqData[i]))*10**12
            maxList=np.vstack([maxList,dummy])
    
#maxList=np.sort(maxList)
    print(len(maxList))
#find the xlim for the plot
    endP=max(maxList[:,1])*2
#now plot the fft analysis
    plt.figure()
    plt.plot(((1/(freqData)))*10**12,np.abs(ftData))
    plt.xlabel('Hydrogen Bond Lifetime ('+timestep+')')
    plt.xlim([0,endP])
    plt.savefig('FTplot_'+fname+'.png')
    plt.close()
    #plt.show()
    f=open('FT_results_for_'+fname+'.txt','w')
    f.write('Amplitude'+'\t'+'Time ('+timestep+')'+'\n')
    for i in range(len(maxList)):
        f.write('%s'%maxList[i,0]+'\t'+'%s'%maxList[i,1]+'\n')
    f.close()
