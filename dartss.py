from darts.models import AutoARIMA
import numpy as np
from tqdm import tqdm 
import model_acc as m
from darts import TimeSeries



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
    
    return er.experrors()  