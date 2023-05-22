"""MODELS"""
import numpy as np

import model_acc as m



class CustomError(Exception):
    pass


def Naive(train_set,test_set,coef=(1)):
    #HELP FUNCTION
    def Naive_help(tim_ser,coef):
        if sum(coef)>=1.001 or sum(coef)<=0.999: raise CustomError("Πρέπει το sum(coef) να είναι =1")
        result=np.zeros([tim_ser.shape[0],])
        for i in range(len(coef)):
            result += tim_ser[:,-4*(i+1)]* coef[i]
            
        
        return result
    
    x,y=test_set.shape
    u,v=train_set.shape
    

    
    results=np.zeros(test_set.shape)
    
    time_series=np.concatenate((train_set,test_set),axis=1)
    for i in range(y):
        results[:,i]= Naive_help(time_series[:,:i+v],coef)
    
    er=m.errors(test_set,results)
    
    return er.experrors()