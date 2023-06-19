import pickle
import numpy as np
import models1m as m
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


arim=m.arima1m(train_set,test_set,order=(1,1,0))
with open("arima1m.pkl","wb") as f:
    pickle.dump([train_set,arim[1],test_set],f)



naiv= m.Naive (train_set,test_set,coef=(1,0),seasonality=12)

seasonality=(1,0,1,4)
sarim=m.sarima1m(train_set,test_set,order=(0,1,0),seasonality=seasonality)
with open("sarima1m.pkl","wb") as f:
    pickle.dump([train_set,sarim[1],test_set],f)




# autoarim= d.autoarima1m(train_set[:10000,:],test_set[:10000,:])
# with open("autoarima1m.pkl","wb") as f:
#     pickle.dump([train_set[:10000,:],autoarim[1],test_set[:10000,:]],f)




# lstm=d.lstm1m(train_set[:10000,:],test_set[:10000,:])
# with open("lstm1m.pkl","wb") as f:
#     pickle.dump([train_set[:10000,:],lstm[1],test_set[:10000,:]],f)




knn=  mc.KNNtimeseries(mesured_data,train_set,test_set,k=40,mean=True)
with open("knn1m.pkl","wb") as f:
    pickle.dump([train_set,knn[1],test_set],f)


