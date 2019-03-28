#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:19:55 2016

@author: wcole
"""
import sys
import spectral as spec
import numpy as np
import matplotlib.pyplot as plt
import os

#choose the appropriate version of hbar for your timestep
global h_bar
#hbar is in units of fs*cm^-1
h_bar=5309.184971

#this hbar is in units of ps*cm^-1
#h_bar=5.309184971
#this in terms of ns*cm^-1
#h_bar=0.005309184971
#This script will take a input hamiltonian and simulate it in time
#It will be used for the dimer, trimer, and pentamer water cluster systems


def getMat():
    """Get a tunneling hamiltonian matrix from file.  Prompt user for the file
        
    Returns
    -------
    inMat: numpy array with the the tunneling elements represented as letters,
        the user will later be prompted to provide values for these elements
        
    """
    while True:
        fname=input('Please enter input file (include extension) ')
        if os.path.isfile(fname) is True:
            inMat=np.loadtxt(fname,dtype=bytes,delimiter='\t').astype(str)
            return inMat
        elif fname=='exit':
            sys.exit()
        else:
            print('Input invalid, please make sure file name includes extension')
            continue
        

def getVar(inHam):
    """Parse the input matrix for the elements symbols, this is in preparation for the user
        entering values
    Parameters
    ----------
    inHam : numpy array obtained from getMat(), with string elements representing tunneling motions
        of the corresponding cluster
        
    Returns
    -------
    outList : List of the different elements present in the matrix, this will be presented to
        the user for definition.
        
    """
    outList=[]
    for i in range(len(inHam)):
        for j in range(len(inHam)):
            if i<=j and inHam[i,j] not in outList[:]:
                outList.append(inHam[i,j])
    return outList

def getVal(inList):
    """Get values from the user for each element of the tunneling hamiltonian
    Parameters
    ----------
    inList : list of the matrix elements obtained from getVar()
    
    Returns
    -------
    outVar : list of floats representing the numeric values for each element in the
        user given tunneling matrix
        
    """
    outVal=np.zeros([len(inList),1])
    for i in range(len(inList)):
        var=inList[i].astype(str)
        while True:
            try:
                #print('Please enter the value for '+'%s'%inList[i])
                inVal=float(input('Please enter the value for '+var+'\n'))
                outVal[i]=inVal
                break
            except ValueError:
                print('Please enter a valid number')
                continue
            
    return outVal
                
    

#Now we need something that will loop thorugh the Ham and replace variables 
#With their actual values
def repHam(inHam, varlist,value):
    """Print the tunneling hamiltonian entered, with string elements, and then the same matrix
        with the user defined values
    Parameters
    ----------
    inHam : numpy array from the user inputed tunneling matrix obtained from getMat()
    
    varList : list of strings representing the different elements of the tunneling matrix obtained from
        getVar()
    
    value : list of float values defined by the user obtained from getVal()
    
    Returns
    -------
    outHam : the tunneling matrix inputed as inHam with string elements replaced by the values defined in
        value[]
        
    """
    #The input will be string matrix, str variable to replace, value
    print('Current Ham matrix:')
    print(inHam)
    outHam=np.zeros([len(inHam),len(inHam)])
    for i in range(len(inHam)):
        for j in range (len(inHam)):
            for k in range(len(varlist)):
                if inHam[i,j] in varlist[k]:
                    outHam[i,j]=value[k]
    print('Converted to...')
    print(outHam)
    return outHam

#Now need to introduce timedependence to the eigenvectors, basically  
#multiplying each by the evo operator exp(-jEt/h_bar)
def timeEvo(eigenVec,eigenVal,t):
    """Apply the time evolution operator, exp((i * eigenvalue * time) / h_bar), to each eigenvector
    Parameters
    ----------
    eigenVec : numpy array representing the matrix of eigenvectors (in each column) of the user defined
        tunneling matrix
    
    eigenVal : list of eigenvalues of the tunneling matrix.  Eigenvalue[i] corresponds to column i
        of eigenVec
        
        t : time step of the calculation
        
    Returns
    -------
    tdMat : the eigenvector matrix, eigenVec, with each column operated on by the time evolution operator
        corresponding the appropriate eigenvector
        
    """
    #The eigenVec should be sq matrix and eigenVal a list of eigenvalues
    #the ith COLUMN of eigenVec corresponds to eigenVal[i]
    #t is a particular time point
    tdMat=np.zeros([len(eigenVec),len(eigenVec)],dtype=complex)
    for i in range(len(eigenVec)):
        op=np.exp((-1j*eigenVal[i]*t)/h_bar)
        tdMat[:,i]=op*eigenVec[:,i]
    return tdMat
#This return a time dependent eigen matrix of w.f.   

#Now we need to solve for the integration constants subject to the initial 
#conditions

def solveIC(eigenVec, eigenVal):
    """Solve the linear matrix equation (Mat)*x = y, subject to the initial conditions defined as
        placing all of the population in the first minima
        
    Parameters
    ---------
    eigenVec : the eigenvector matrix at time = 0
    
    eigenVal : the eigenvalue matrix, this parameter is not used in the calculationand should be removed
        from the function (Mar 2019)
        
    Returns
    -------
        intconMat = an array represent the integration constants for the eigenvector matrix at time = 0
        
    """
    zeroMat=eigenVec
    icMat=np.zeros([len(eigenVec),1])
    icMat[0]=1
    intconMat=np.linalg.solve(zeroMat,icMat)
    return intconMat

def sqCoeff(coeffVec):
    """Square the elements of the given vector
    
    Parameters
    ----------
    coeffVec : vector whose elements are to be squared
    
    Returns
    -------
    outVec : a vector with the squared elements of coeffVec
        
    """
    outVec=np.zeros([len(coeffVec),1])
    for i in range(len(coeffVec)):
        outVec[i]=np.conj(coeffVec[i])*coeffVec[i]
    return outVec

def simCalc(eigenVec,eigenVal,intConst, time, timeStep):
    """Perform the time evolution on a given matrix for given time length
        
    Parameters
    ----------
    eigenVec : the eigenvector matrix used to evolve
    
    eigenVal : the eigenvalues of eigenVec; eigenVal[i] corresponds to column i of eigenVec
    
    time : the length of time to run the calculation for
    
    timeStep : the increments to perform the calculation.  The total number of calculations
        is time / timeStep
        
    Returns
    -------
    
    totMat: the matrix representing the population of the wavefunction in each minima
        in columns > 0, at every time point calculated in column 0
        
    """
    print('Beginning the calculation..')
    print('To USER: Ignore the casting error call, the imaginary part is zero')
    results=np.zeros([0,len(eigenVal)])
    timeVec=np.zeros([0,1])
    #time should be in units of 10s of femtoseconds
    for t in range(0,time, timeStep):
        #get the time dependent w.f.
        tdVec=timeEvo(eigenVec,eigenVal,t)
        #Now add in the integration constants
        coeffVec=np.dot(tdVec,intConst)
        #Now sq the matrix
        sqVec=sqCoeff(coeffVec)
        #and finally collect that for each point
        timeVec=np.vstack([timeVec,t])
        results=np.vstack([results,np.transpose(sqVec)])
    #want to add a time index as well
    totMat=np.hstack((timeVec,results))
    return totMat
        
#This will be a set of function to plot the results for the calculation
#Want to have the ability to "group" up the coeffecients so we can plot them 
#together.
def plotResults(inMat):
    """Plot the population of each minima wrt to time
        
    Parameters
    ---------
    inMat : the matrix to plot with time in column 0 and populations in column > 0
        
    """
    plt.figure()
    #Want a seperate plot for each coeff
    #Plot in units of picoseconds
    for i in range(1,len(np.transpose(inMat))):
        plt.plot((inMat[:,0]/1000),inMat[:,i],label=i)
    plt.xlabel('Time (fs)')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.show()
            
def writeOut(fName,inMat):
    """Write the results of the calculation to a file
        
    Parameters
    ---------
    
    fName : the name of the file to be written
    
    inMat : the matrix resulting from the calculation
        
    """
    f=open(fName+'.txt','w')
    f.write('Time(fs)'+'\t')
    for k in range(len(np.transpose(inMat))):
        f.write('State '+'%s'%k+'\t')
    f.write('\n')
    for i in range(len(inMat)):
        for j in range(len(np.transpose(inMat))):
            f.write('%s'% inMat[i,j]+'\t')
        f.write('\n')
    f.close()

def readME(fname,inHam,eigenVal,orthEVec,intConst,inTime, step):
    """Generate a readme file describing the calculation performed
        
    Parameters
    ----------
    fName : the header of the file to be written
    
    inHam : the user inputed hamiltonian
    
    eigenVal : the eigenvalues of the input hamiltonian
    
    orthEVec : the orthonormal eigenvectors of inHam
    
    intConst: the intgration constants obtained from the given initial conditions
    
    inTime : the range of time simulated
    
    step : the timeStep used
        
    """
    f=open('ReadME_'+fname+'.txt','w')
    f.write('Input conditions for run are as follows:\n\n')
    f.write('Calculation ran for: '+'%s'%inTime+' ps\n')
    f.write('Femtoseconds per step: '+'%s'%step+' fs\n\n')
    f.write('Hamiltonian used: \n\n')
    for i in range(len(inHam)):
        for j in range(len(inHam)):
            f.write('%s'%inHam[i,j]+'\t')
        f.write('\n')
    f.write('\n')
    f.write('Eigenvalues: \n\n')
    for i in range(len(eigenVal)):
        f.write('%s'%eigenVal[i]+'\t')
    f.write('\n\n')
    f.write('Orthonormal eigenvectors are (in columns): \n\n')
    for i in range(len(orthEVec)):
        for j in range(len(orthEVec)):
            if orthEVec[i,j]<0:
                f.write('%s'% round(orthEVec[i,j],3)+'\t')
            else:
                f.write('%s'% round(orthEVec[i,j],4)+'\t')
        f.write('\n')
    f.write('\n')
    f.write('Integration Constants are: \n\n')
    for i in range(len(intConst)):
        f.write('%s'% intConst[i]+'\t')
    f.close()
    
#going to try and put an external main function here that can be called externally and 
#and be parsed arguements
def extMain(hamFname, inValue, timePS, timestepFS,outFname):
    """Method to run the entire calculation externally, useful for running multiple
            calculations in series
    
    Parameters
    ----------
    hamFname : the header for the input matrix file
    
    inValue : the values to replace in the matrix
    
    timePS : the time in picosecond to run the calculation for
    
    timestepFS : the timestep in femtoseconds
    
    outFname : the header for the output files
        
    """
    #First need to get the matrix
    ham=np.loadtxt(hamFname,dtype=bytes,delimiter='\t').astype(str)
    #now get the variables
    varList=getVar(ham)
    #Now need to add in value to the Ham
    ham=repHam(ham,varList,inValue)
    eigenVal,eigenVec=np.linalg.eigh(ham)
    print('System eigenvalues are: ')
    print(eigenVal)
    orthEVec=spec.orthogonalize(eigenVec)
    #the value for time in in units of ps
    scaledTime=1000*timePS
    #Calculate the integration constants
    intConst=solveIC(orthEVec,eigenVal)
    #Now do the calculation, timestep is in units femtoseconds
    godMat=simCalc(orthEVec,eigenVal,intConst, scaledTime, timestepFS)
    writeOut(outFname,godMat)
    #Should also print out a readme file
    readME(outFname,ham,eigenVal,orthEVec,intConst,timePS, timestepFS)
    print('Have a nice day :)')
#Thus starts the actual program

if __name__=="__main__":
    #Prompt the user for a valid input file
    ham=getMat()
    #Then ask for the variables
    varList=getVar(ham)
    #Now need to get values for the variables
    valVec=getVal(varList)
    #Now use these to replace the gen Ham with the actual values
    ham=repHam(ham,varList,valVec)
    
    #Now begins the heavy lifting
    #First get the eigensystem and orthonormal eigenvectors
    eigenVal,eigenVec=np.linalg.eigh(ham)
    print('System eigenvalues are: ')
    print(eigenVal)
    orthEVec=spec.orthogonalize(eigenVec)
    #Figure out how many points to calculate
    inTime=int(input('How many picoseconds do you want to calculate? '))
    #Always do 10 femtosecond steps
    scaledTime=1000*inTime
    timeStep=int(input('How many femtoseconds per step? '))
    #Calculate the integration constants
    intConst=solveIC(orthEVec,eigenVal)
    #Now do the calculation
    godMat=simCalc(orthEVec,eigenVal,intConst, scaledTime, timeStep)
    #The output matrix has time as the row index and the coeff as the column 
    #index
    while True:
        test=input('Calculation finished, would you like to plot the results? (y/n)')
        if test is 'y':
            #plot things
            plotResults(godMat)
            break
        elif test is 'n':
            #Just exit, maybe we should have a print out function here
            break
        else:
            print('Input invalid, please enter y/n...')
            continue
    
    #Should now write out to a file
    fname=input('Please enter a name for the output file: ')
    writeOut(fname,godMat)
    #Should also print out a readme file
    readME(fname,ham,eigenVal,orthEVec,intConst,inTime, timeStep)
    print('Have a nice day :)')
   
        
