import pickle
import numpy as np
import models as m
"""DATA READ"""

#DATA READ
with open("test_data_final.pkl","rb") as f:
     data=pickle.load(f)

# INFO AND DATA SEPERATION
info=data[:,:5]
timeseries=np.float64(data[:,5:])

#SEPERATION BETWEEN TRAIN AND TEST SET
coef=0.1  #expample 0.1 stands for 10% test set
size=timeseries.shape[1]
train_colum=int((1-coef)*size)

train_set,test_set=np.split(timeseries,[train_colum],axis=1)



#MODELS


#NAIVE

# naiv= m.Naive (train_set,test_set,coef=(1,0))

#ARIMA

#arim=m.arima(train_set,test_set,order=(1,1,0)) #todo trial and error

#SARIMA
# seasonality=(1,0,1,4)
# sarim=m.sarima(train_set,test_set,seasonality,order=(0,1,0))


#LSTM

# lstm=m.LSTMM(train_set,test_set,n_pri_steps=8)


#average

aver=m.average(train_set,test_set)
print(np.average(aver[3]))






