"""MODELS"""
import numpy as np
from model_acc import errors

class CustomError(Exception):
    pass


def Naive(train_set,test_set,coef=(0,1)):
    #HELP FUNCTION
    def Naive_help(tim_ser,coef):
        if sum(coef)>=1.001 or sum(coef)<=0.999: raise CustomError("Πρέπει το sum(coef) να είναι =1")
        if len(coef)*4>tim_ser.shape[0]: raise CustomError("Πολυ μικρή χρονοσειρά για τέτοιο coef")
        result=np.zeros([4,tim_ser.shape[1]])
        for i in range(len(coef)):
            result += tim_ser[-4:,:]* coef[i]
            tim_ser=np.delete(tim_ser,-1,axis=0)
            tim_ser=np.delete(tim_ser,-1,axis=0)
            tim_ser=np.delete(tim_ser,-1,axis=0)
            tim_ser=np.delete(tim_ser,-1,axis=0)
        
        return result
    
    x,y=test_set.shape
    u,v=train_set.shape
    

    
    results=np.zeros(test_set.shape)
    
    time_series=np.concatenate((train_set,test_set),axis=1)
    for i in range(x):
        results[:i]= Naive_help(time_series[:,i+u],coef)
    
    er=errors(test_set,results)
    
    return er.experrors()