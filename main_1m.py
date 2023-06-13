import pickle
import numpy as np
import models as m
import models_cluster_1m as mc
import darts_1m as d
"""DATA READ"""

#DATA READ
with open("test_data_1m.pkl","rb") as f:
     data=pickle.load(f)

# INFO AND DATA SEPERATION
info=data[:,:5]
timeseries=np.float64(data[:,5:])


#SEPERATION BETWEEN TRAIN AND TEST SET
mesured_coef=0.8
split_index = int(timeseries.shape[0] * mesured_coef)

mesured_data=timeseries[:split_index,:]
unmesured_data=timeseries[split_index:,:] 


coef=0.08  #expample 0.1 stands for 10% test set
size=timeseries.shape[1]
train_colum=int((1-coef)*size)

train_set,test_set=np.split(unmesured_data,[train_colum],axis=1)



autoarim= d.autoarima1m(train_set[:10000,:],test_set[:10000,:])





# lstm=d.lstm1m(train_set[:10000,:],test_set[:10000,:])





# knn=  mc.KNNtimeseries(mesured_data,train_set,test_set,k=40,mean=True)


