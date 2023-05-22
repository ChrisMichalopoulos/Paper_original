import pickle
import numpy as np
import models as m
"""DATA READ"""

#DATA READ
with open("test_data_final.pkl","rb") as f:
     data=pickle.load(f)

# INFO AND DATA SEPERATION
info=data[:,:5]
timeseries=data[:,5:]

#SEPERATION BETWEEN TRAIN AND TEST SET
coef=0.1  #expample 0.1 stands for 10% test set
size=timeseries.shape[1]
train_colum=int((1-coef)*size)

train_set,test_set=np.split(timeseries,[train_colum],axis=1)



#MODELS


#NAIVE

b= m.Naive (train_set,test_set,coef=(0,1))






