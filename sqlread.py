import numpy as np
import pickle
import pandas as pd
import sqlite3

### Read sqlite query results into a pandas DataFrame ### 

con = sqlite3.connect("data1.db")
df = pd.read_sql_query("SELECT * from timeseries_ESYS LIMIT 10000", con)
con.close()
test_sample=df.sample(frac=0.25)

with open("test_data.pkl","wb") as f:
    pickle.dump(test_sample,f)
"------FIX THIS IN THE END---------"


with open("test_data.pkl","rb") as f:
     data=pickle.load(f)

n=3 #number of last colums to drop
data=data.iloc[:,n:-n]

"""" Remove rows with Nan"""
data2=data.dropna(subset=[data.columns[-1]])

with open("test_data_filltered_w0.pkl","wb") as f:
     pickle.dump(data2,f)
     
     

"""REMOVE GARBAGE TIMESERIES"""

array=data2.values

info=array[:,:5]
actual_data=array[:,5:]
threshold=5
index=[]
last_val=15 # the number of last values that are considered

for i in range(info.shape[0]):
    if actual_data[i,-last_val:].mean() > threshold:
        if not np.prod(actual_data[i,-last_val:])<=0.01:
            if not np.isnan(np.sum(actual_data[i,:])):
                if not np.any(actual_data[i,-last_val:] == np.nan) :
                    index.append(i)
        
    
    
new_info=info[index,:]
new_actual_data=actual_data[index,:]

"""3month base from 1 month"""
[u,v]= new_actual_data.shape

v3=int(v/3)

mvalues = np.zeros((u,v3))

for i in range(v3):
    mvalues[:,i]= np.sum(new_actual_data[:,3*i:3*i+3],axis=1)
    
    
finished_data=np.concatenate((new_info,mvalues),axis=1)

with open("test_data_final.pkl","wb") as f:
     pickle.dump(finished_data,f)
    
    

