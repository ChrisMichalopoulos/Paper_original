import numpy as np
from numba import jit,prange
from tqdm import tqdm
import model_acc as m
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import math as ma


#same as 3 months but regulated for 1 month predict
#seasonality = 12 months

class CustomError(Exception):
    pass



def  KNNtimeseries(mesured,train_set,test_set,k=15,mean=True):
    
    
    #@jit(nopython=True,parallel=True)
    def KNNregressor(train_set,test_set,k=15,mean=True,n=1):
        
        #data separtation
        
        compare=train_set[:,:9]
        knowned_result=train_set[:,9:]
        compared=test_set[:,:9]
        
        #size
        x=compared.shape[0]
        y=compare.shape[0]
    
        
        
        rangematrix  = np.zeros((y, x))
        nearest      = np.full((y, x), -1)
        predict      = np.zeros((x,3))
        
        
        
        if mean == True:
            for i in prange(x):
                for ii in range(y):
                    rangematrix[ii,i]=np.linalg.norm(compare[ii,:]-compared[i,:])
                nearest[:,i]=np.argsort(rangematrix[:,i])
                predict[i,:]=np.sum(knowned_result[nearest[:k,i]],axis=0)/k
            
            
        else:
            for i in prange(x):
                corell       = np.zeros((y,))
                for ii in range(y):
                    rangematrix[ii,i]=np.linalg.norm(compare[ii,:]-compared[i,:])
                    
                    corell[ii]=np.corrcoef(compare[ii,:], compared[i,:])[0, 1]
                    if corell[ii]==np.nan:
                        corell[ii]=0
                nearest[:,i]=np.argsort(rangematrix[:,i])
                

                nearest_temp=nearest[:k,i]
                
                predict[i]=np.sum(knowned_result[nearest_temp] * corell[nearest_temp]          )/np.sum(corell[nearest_temp])
 
                                                                                                                      
                
                
                
               
        
        return predict
        
        
        
    x,y=test_set.shape
    u,v=train_set.shape
    results=np.zeros((x,y))
    
    if y%3!=0: raise CustomError("Den einai gia 3mhna")
    
    
    time_series=np.concatenate((train_set,test_set),axis=1)
        
        
    for i in tqdm(range(0,y,3)):
        results[:,i:i+3]= KNNregressor(mesured[:,i+v-12:i+v],time_series[:,i+v-12:i+v],k=k,mean=mean)
        
    #3month base    
    
    v3=int(y/3)

    mvalues = np.zeros((x,v3))
    actual_values = np.zeros((x,v3))
    for i in range(v3):
        mvalues[:,i]= np.sum(results[:,3*i:3*i+3],axis=1)   
        actual_values[:,i]=np.sum(test_set[:,3*i:3*i+3],axis=1)
        
    print(actual_values)
    print(mvalues)
    er=m.errors(actual_values,mvalues)
    
    return er.experrors() , results