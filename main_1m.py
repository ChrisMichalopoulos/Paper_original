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

# results=np.ones([4,4])
# for i in range(4):
#     for ii in range(4):    
#         arim=m.arima1m(train_set[:10000,:],test_set[:10000,:],order=(i,1,ii))
#     # with open("arima1m.pkl","wb") as f:
#     #     pickle.dump([train_set[:10000,:],arim[1],test_set[:10000,:]],f)
        # results[i,ii]=np.average(arim[0][2])
        # print(results[i,ii])
        # print([i,ii])
# naiv= m.Naive (train_set,test_set,coef=(1,0),seasonality=12)



# seasonality=(1,0,1,12)
# for i in range(4):
#     for ii in range(4):
#         sarim=m.sarima1m(train_set[:10000,:],test_set[:10000,:],order=(0,1,0),seasonality=(i,0,ii,12))
#         results[i,ii]=np.average(sarim[0][2])
#         print(results[i,ii])
#         print([i,ii])

# with open("sarima1m.pkl","wb") as f:
#     pickle.dump([train_set,sarim[1],test_set],f)




# autoarim= d.autoarima1m(train_set[:10000,:],test_set[:10000,:])
# with open("autoarima1m.pkl","wb") as f:
#     pickle.dump([train_set[:10000,:],autoarim[1],test_set[:10000,:]],f)



# for i in [50,100,200,300]:
#     for ii in [25,50,75,100]:
#         lstm=m.LSTM1m(train_set[:100,:],test_set[:100,:],layers=ii,epochs=i)
#         # with open("lstm1m.pkl","wb") as f:
#         #     pickle.dump([train_set[:5000,:],lstm[1],test_set[:5000,:]],f)
#         results[i,ii]=np.average(lstm[0][2])
#         print(results[i,ii])
#         print([i,ii])

results=np.ones(5)
for i in [30,40,60,80,100]:
    knn=  mc.KNNtimeseries(mesured_data[:100000,:],train_set[:20000,:],test_set[:20000,:],k=i,mean=True)

    results[i]=np.average(knn[0][2])
    print([i,results[i]])
# with open("knn1m.pkl","wb") as f:
#     pickle.dump([train_set[:20000,:],knn[1],test_set[:20000,:]],f)


