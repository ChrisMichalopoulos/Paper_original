import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import model_acc as m




def arima1m(train_set,test_set,order=(1,1,1)):
    x,y = test_set.shape
    results = np.zeros([x,y])
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(0,y,3):
            model=ARIMA(train,order=order)
            m_f=model.fit()
            
            
            results[i,ii:ii+3]=m_f.forecast(steps=3)
            train.append(test_set[i,ii])
            train.append(test_set[i,ii+1])
            train.append(test_set[i,ii+2])
        
    
    
    v3=int(y/3)
    
    mvalues = np.zeros((x,v3))
    actual_values = np.zeros((x,v3))
    for i in range(v3):
        mvalues[:,i]= np.sum(results[:,3*i:3*i+3],axis=1)   
        actual_values[:,i]=np.sum(test_set[:,3*i:3*i+3],axis=1)
        
        
    er=m.errors(actual_values,mvalues)
    
    return er.experrors(), results





def sarima1m(train_set,test_set,order=(1,1,1),seasonality=(1,1,1,12)):
    x,y = test_set.shape
    results = np.zeros([x,y])
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(0,y,3):
            model=ARIMA(train,order=order,seasonal_order=seasonality)
            try:
                m_f=model.fit()
            
            
                results[i,ii:ii+3]=m_f.forecast(steps=3)
            except:
                results[i,ii:ii+3]=np.array([np.average(train),np.average(train),np.average(train)])
            train.append(test_set[i,ii])
            train.append(test_set[i,ii+1])
            train.append(test_set[i,ii+2])
        
    
    
    v3=int(y/3)
    
    mvalues = np.zeros((x,v3))
    actual_values = np.zeros((x,v3))
    for i in range(v3):
        mvalues[:,i]= np.sum(results[:,3*i:3*i+3],axis=1)   
        actual_values[:,i]=np.sum(test_set[:,3*i:3*i+3],axis=1)
        
        
    er=m.errors(actual_values,mvalues)
    
    return er.experrors(), results