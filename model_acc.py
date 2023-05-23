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
        return np.sum(self.a - self.p)

    def sMAPE(self,factor=1):
        N=self.a.shape[1]
        summ=np.sum(abs(self.a-self.p)*factor/(self.a+self.p),axis=1    )
        return summ/N
        
    def MAPE(self):
        N=self.a.shape[1]
        summ=np.sum(abs((self.a-self.p)/self.a))
        return summ/N
        
    def experrors(self):
        return [self.RMSE(),self.TotalSum(),self.MAPE(),self.sMAPE()]