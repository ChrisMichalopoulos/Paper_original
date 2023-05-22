"""model acuracy meters"""

import numpy as np




# def RMSE(actual_data, predictions):
    
#     N=actual_data.shape[1]
#     summ= np.sum((actual_data-predictions)**2,axis=1)
#     return (summ/N)**0.5

    
# def TotalSum(actual_data, predictions):
#     return np.sum(actual_data-predictions)


# def sMAPE(actual_data,predictions,factor=1):
#     N=actual_data.shape[1]
#     summ=np.sum(abs(actual_data-predictions)*factor/(actual_data+predictions),axis=1    )
#     return summ/N


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
        
            
    def experrors(self):
        return [self.RMSE(),self.TotalSum(),self.sMAPE()]