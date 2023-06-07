import pickle
import numpy as np
import models as m
import models_cluster as mc
import dartss as d
"""DATA READ"""

#DATA READ
with open("test_data_final_low_res.pkl","rb") as f:
     data=pickle.load(f)

# INFO AND DATA SEPERATION
info=data[:,:5]
timeseries=np.float64(data[:,5:])


#SEPERATION BETWEEN TRAIN AND TEST SET
mesured_coef=0.5
split_index = int(timeseries.shape[0] * mesured_coef)

mesured_data=timeseries[:split_index,:]
unmesured_data=timeseries[split_index:,:] 


coef=0.05  #expample 0.1 stands for 10% test set
size=timeseries.shape[1]
train_colum=int((1-coef)*size)

train_set,test_set=np.split(unmesured_data,[train_colum],axis=1)



#MODELS


#NAIVE

naiv= m.Naive (train_set,test_set,coef=(1,0))

# with open("naive.pkl","wb") as f:
#     pickle.dump([train_set,naiv[1],test_set],f)
# #ARIMA

# arim=m.arima(train_set,test_set,order=(1,1,0)) #todo trial and error
# with open("arima.pkl","wb") as f:
#     pickle.dump([train_set,arim[1],test_set],f)

# #SARIMA
# seasonality=(1,0,1,4)
# sarim=m.sarima(train_set,test_set,seasonality,order=(0,1,0))


#AutoARIMA

# autoarim= d.autoarima(train_set[:100,:],test_set[:100,:])


#LSTM

# lstm=m.LSTMM(train_set,test_set,n_pri_steps=8)


#average

# aver=m.average(train_set,test_set)
# print(np.average(aver[3]))



#KNN   


# knn=  mc.KNNtimeseries(mesured_data,train_set,test_set,k=30,mean=True)


# print(np.average(knn[3]))
    


#GMM

# gmm=mc.GMM(mesured_data,train_set,test_set,k=30,cov="full")
# print(np.average(gmm[3]))    #DEBUGGING