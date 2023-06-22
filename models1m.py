import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import model_acc as m
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



def Naive(train_set,test_set,coef=(1),seasonality=12):
    #HELP FUNCTION
    def Naive_help(tim_ser,coef,seasonal):
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
        results[:,i]= Naive_help(time_series[:,:i+v],coef,seasonal=seasonality)
    
    er=m.errors(test_set,results)
    
    return er.experrors(),results



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
            try:
                model=ARIMA(train,order=order,seasonal_order=seasonality)
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




def LSTM1m(train_set,test_set,n_pri_steps=12,layers=100):
    
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
        model.add(LSTM(layers,input_shape=(n_pri_steps,1)))
        model.add(Dense(1))
        model.compile(optimizer="adam",loss="mse")
        
        #model fit
        
        model.fit(generator,epochs=50,verbose=0)
        


        for ii in range(0,y,3):
            predictions=[]
            inputt=list(time_series[v-n_pri_steps+ii:v+ii][:,0])
            
            
            prediction = model.predict(np.array(inputt).reshape((1,n_pri_steps,1)),verbose=0)
            predictions.append(prediction[0][0])
            inputt.append(prediction[0][0])
            
            prediction2 = model.predict(np.array(inputt[-12:]).reshape((1,n_pri_steps,1)),verbose=0)
            predictions.append(prediction2[0][0])
            inputt.append(prediction2[0][0])
            
            prediction3=model.predict(np.array(inputt[-12:]).reshape((1,n_pri_steps,1)),verbose=0)
            predictions.append(prediction3[0][0])

            
            
            results[i,ii:ii+3] = scaler.inverse_transform(np.array(predictions).reshape(1, -1))
        
        
    er=m.errors(test_set,results)
    
    return er.experrors() , results
