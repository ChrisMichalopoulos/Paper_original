import pickle
import numpy as np
import models as m
import models_cluster as mc
import dartss as d
"""DATA READ"""

#DATA READ
with open("test_data_3m.pkl","rb") as f:
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

print("start")


#MODELS


#NAIVE

# naiv= m.Naive (train_set,test_set,coef=(1,0),seasonality=4)

# with open("naive3m.pkl","wb") as f:
#     pickle.dump([train_set,naiv[1],test_set],f)
# #ARIMA




# arim=m.arima(train_set,test_set,order=(1,1,0)) #todo trial and error
# with open("arima3m.pkl","wb") as f:
#     pickle.dump([train_set,arim[1],test_set],f)

#SARIMA
# seasonality=(1,0,1,4)
# sarim=m.sarima(train_set,test_set,seasonality,order=(0,1,0))
# with open("sarima3m.pkl","wb") as f:
#     pickle.dump([train_set,sarim[1],test_set],f)

#AutoARIMA

# autoarim= d.autoarima(train_set[:10000,:],test_set[:10000,:])
# with open("autoarim3m.pkl","wb") as f:
#     pickle.dump([train_set[:10000,:],autoarim[1],test_set[:10000,:]],f)

#LSTM

# lstm=m.LSTMM(train_set[:10000,:],test_set[:10000,:],n_pri_steps=8,layers=75)
# # lstm=d.lstm(train_set[:10000,:],test_set[:10000,:])
# with open("lstm3m.pkl","wb") as f:
#     pickle.dump([train_set[:10000,:],lstm[1],test_set[:10000,:]],f)


#average

# aver=m.average(train_set,test_set)
# with open("average3m.pkl","wb") as f:
#     pickle.dump([train_set,aver[1],test_set],f)




#KNN   


knn=  mc.KNNtimeseries(mesured_data[:100000,:],train_set[:30000,:],test_set[:30000,:],k=40,mean=True)

with open("knn3m.pkl","wb") as f:
    pickle.dump([train_set,knn[1],test_set],f)


    


# #GMM

# gmm=mc.GMM(mesured_data,train_set,test_set,k=30,cov="full")
# with open("GMM.pkl","wb") as f:
#       pickle.dump([train_set,gmm[1],test_set],f)
# # print(np.average(gmm[3]))    #DEBUGGING


    