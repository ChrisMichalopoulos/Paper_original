import numpy as np
import matplotlib.pyplot as plt
import pickle

name="arima"
with open(name+".pkl","rb") as f:
    data=pickle.load(f)
        
lis=np.random.randint(0,data[0].shape[0]+1,size=10)

train=data[0][lis]

test=data[2][lis]

forecast= data[1][lis]

timeseries=np.concatenate((train, test), axis=1)


for i in range(train.shape[0]):
    
    fig, ax = plt.subplots()
    ax.plot(np.append(train[i,:], test[i,0]),color="blue",label="actual_timeseries")
    ax.plot(np.arange(test.shape[1])+ train.shape[1]    ,forecast[i,:],color="red",label="Forecast")
    ax.plot(np.arange(test.shape[1])+train.shape[1], test[i,:],color="green",label="test_set"    )
    ax.set_title('{} {}'.format(name,i))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    plt.show()
    
    
    
    