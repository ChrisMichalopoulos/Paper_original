from darts.models import AutoARIMA
import numpy as np
from tqdm import tqdm 
import model_acc as m
from darts import TimeSeries
from darts.models import AutoARIMA, RNNModel


def autoarima(train_set,test_set):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(y):
            data=TimeSeries.from_values(np.array(train))
            model=AutoARIMA()
            model.fit(data)
            
            results[i,ii]=model.predict(n=1).values()[0][0]
            train.append(test_set[i,ii])
    
    
    er=m.errors(test_set,results)
    
    return er.experrors() , results



def lstm(train_set,test_set):
    
    x,y=test_set.shape
    
    results=np.zeros([x,y])
    
    
    for i in tqdm(range(x)):
        train=list(train_set[i,:])
        for ii in range(y):
            data=TimeSeries.from_values(np.array(train))
            model=RNNModel(8,model="LSTM"   )
            model.fit(data,verbose=False)
            
            results[i,ii]=model.predict(n=1).values()[0][0]
            train.append(test_set[i,ii])

        
        
    er=m.errors(test_set,results)
    
    return er.experrors(), results