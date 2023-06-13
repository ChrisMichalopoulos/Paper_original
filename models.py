"""MODELS"""
import numpy as np

import model_acc as m

from statsmodels.tsa.arima.model import ARIMA

from tqdm import tqdm


import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM





class CustomError(Exception):
    pass


def Naive(train_set,test_set,coef=(1),seasonality=12):
    #HELP FUNCTION
    def Naive_help(tim_ser,coef,seasonality):
        if sum(coef)>=1.001 or sum(coef)<=0.999: raise CustomError("Πρέπει το sum(coef) να είναι =1")
        result=np.zeros([tim_ser.shape[0],])
        for i in range(len(coef)):
            result += tim_ser[:,-seasonality*(i+1)]* coef[i]
            
        
        return result
    
    x,y=test_set.shape
    u,v=train_set.shape
    

    
    results=np.zeros(test_set.shape)
    
    time_series=np.concatenate((train_set,test_set),axis=1)
    for i in range(y):
        results[:,i]= Naive_help(time_series[:,:i+v],coef,seasonality=12)
    
    er=m.errors(test_set,results)
    
    return er.experrors(),results








def arima(train_set,test_set,order=(1,1,1)):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    # train_set,stat=m.normalization(train_set)
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(y):
            model=ARIMA(train,order = order)
            m_f=model.fit()
            results[i,ii]=m_f.forecast()
            train.append(test_set[i,ii])
    
    # results= m.denormalization(results,stat)
    
    er=m.errors(test_set,results)
    
    return er.experrors(), results     
            


def sarima(train_set,test_set,seasonality=(1,1,1,4),order=(1,1,1)):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(y):
            try:
                model=ARIMA(train,order = order,seasonal_order=seasonality)
                m_f=model.fit()
                results[i,ii]=m_f.forecast()
                
            except:
                results[i,ii]=0
            train.append(test_set[i,ii])
    er=m.errors(test_set,results)
    
    return er.experrors()          
    
    
    

def LSTMM(train_set,test_set,n_pri_steps=8):
    
    x,y=test_set.shape
    u,v=train_set.shape
    
    results=np.zeros([x,y])
    
    for i in tqdm(range(x)):
        train=pd.Series(train_set[i,:]).to_frame()
        #scale data
        scaler=MinMaxScaler()
        scaler.fit(train)       #TODO maybe its beter if we scale the hole timesiries
        scaled_train=scaler.transform(train)
        
        #Batchs
        generator=TimeseriesGenerator(scaled_train,scaled_train,length=n_pri_steps,batch_size=1) #TODO TEST BOTH SCALED
        
        test=pd.Series(test_set[i,:]).to_frame()
        scaled_test=scaler.transform(test)
        time_series=np.concatenate((scaled_train,scaled_test),axis=0)
        
        
        #define model
        
        model=Sequential()
        model.add(LSTM(100,input_shape=(n_pri_steps,1)))
        model.add(Dense(1))
        model.compile(optimizer="adam",loss="mse")
        
        #model fit
        
        model.fit(generator,epochs=20,verbose=0)
        
        for ii in range(y):
            sc_predict=model.predict(time_series[v-n_pri_steps+ii:v+ii].reshape((1,n_pri_steps,1)))
            results[i,ii]=scaler.inverse_transform(sc_predict)
        
        
        
    er=m.errors(test_set,results)
    
    return er.experrors() 



def average(train_set,test_set):
    x,y=test_set.shape
    u,v=train_set.shape
    
    
    results=np.zeros(test_set.shape)
    
    time_series=np.concatenate((train_set,test_set),axis=1)
    for i in range(y):
        results[:,i]= np.average(time_series[:,:i+v],axis=1)
    
    er=m.errors(test_set,results)
    
    return er.experrors(), results 
    
        
        
        
        
        

        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
    

