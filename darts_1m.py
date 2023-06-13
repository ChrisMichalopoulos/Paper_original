# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:18:02 2023

@author: xrism
"""

from darts.models import AutoARIMA, RNNModel
import numpy as np
from tqdm import tqdm 
import model_acc as m
from darts import TimeSeries



class CustomError(Exception):
    pass



def autoarima1m(train_set,test_set):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    if y%3!=0: raise CustomError("Den einai gia 3mhna")
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(0,y,3):
            data=TimeSeries.from_values(np.array(train))
            model=AutoARIMA()
            model.fit(data)
            
            results[i,ii:ii+3]=model.predict(n=3).values().ravel()
            train.append(test_set[i,ii])
        
    #3month base    
    
    v3=int(y/3)
    
    mvalues = np.zeros((x,v3))
    actual_values = np.zeros((x,v3))
    for i in range(v3):
        mvalues[:,i]= np.sum(results[:,3*i:3*i+3],axis=1)   
        actual_values[:,i]=np.sum(test_set[:,3*i:3*i+3],axis=1)
        
        
    er=m.errors(actual_values,mvalues)
    
    return er.experrors(), results



def lstm1m(train_set,test_set):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    if y%3!=0: raise CustomError("Den einai gia 3mhna")
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(0,y,3):
            data=TimeSeries.from_values(np.array(train))
            model=RNNModel(12,model="LSTM"   )
            model.fit(data,verbose=False)
            
            results[i,ii:ii+3]=model.predict(n=3).values().ravel()
            train.append(test_set[i,ii])
            train.append(test_set[i,ii+1])
            train.append(test_set[i,ii+2])  #DO NOT KNOW IF ITS TRUE
    
    #3month base    
    
    v3=int(y/3)
    
    mvalues = np.zeros((x,v3))
    actual_values = np.zeros((x,v3))
    for i in range(v3):
        mvalues[:,i]= np.sum(results[:,3*i:3*i+3],axis=1)   
        actual_values[:,i]=np.sum(test_set[:,3*i:3*i+3],axis=1)
        
        
    er=m.errors(actual_values,mvalues)
    
    return er.experrors(), results




