"""model acuracy meters"""

import numpy as np

class errors():
    
    def __init__(self, actual_data, predictions):

        self.a = actual_data
        self.p = predictions
    
    def RMSE(self):
        N=self.a.shape[1]
        summ= np.sum((self.a-self.p)**2,axis=1)
        return (summ/N)**0.5
        
    def TotalSum(self):
        return np.sum(self.a - self.p,axis=1)

    def sMAPE(self,factor=2):
        N=self.a.shape[1]
        summ=abs(self.a-self.p)*factor/(self.a+self.p)
        return summ
        
    def MAPE(self):
        N=self.a.shape[1]
        summ=np.sum(abs((self.a-self.p)/self.a),axis=1)
        return summ/N
        
    
    def MAE(self):
        N=self.a.shape[1]
        summ=np.sum(abs(self.a-self.p),axis=1)
        return summ/N
        
    def experrors(self):
        return [self.RMSE(),self.TotalSum(),self.MAPE(),self.sMAPE(),self.MAE()]
    
    
    
    
    
    
def normalization(timeseries):
    
    x,y=timeseries.shape
    
    normalized=np.ones((x,y))*-1
    statistical=[]
    for i in range(x):
        average=np.average(timeseries[i,:])
        std=np.std(timeseries[i,:])
        normalized[i,:]=(timeseries[i,:]-average)/std
        statistical.append((average,std))
        
    
    return normalized , statistical
        
        
def denormalization(timeseries,stats):
    x,y=timeseries.shape
    denormalizes=np.ones((x,y))*-1
    
    for i in range(x):
        denormalizes[i,:]=timeseries[i,:]*stats[i][1]+stats[i][0]
        
    
    return denormalizes