import numpy as np
import pickle
import pandas as pd
import sqlite3

### Read sqlite query results into a pandas DataFrame ### 

# con = sqlite3.connect("data1.db")
# df = pd.read_sql_query("SELECT * from timeseries_ESYS LIMIT 1000", con)
# con.close()
# test_sample=df.sample(frac=0.25)

# with open("test_data.pkl","wb") as f:
#     pickle.dump(test_sample,f)
"------FIX THIS IN THE END---------"


with open("test_data.pkl","rb") as f:
     data=pickle.load(f)

n=3 #number of last colums to drop
data=data.iloc[:,:-n]

"""" Remove rows with Nan"""
data2=data.dropna(subset=[data.columns[-1]])

with open("test_data_filltered_w0.pkl","wb") as f:
     pickle.dump(data2,f)
     
     
"""3month base from 1 month"""


array=data2.values

info=array[:,:5]
actual_data=array[:,5:]

[u,v]= actual_data.shape

v3=int(v/3)

mvalues = np.zeros((u,v3))

for i in range(v3):
    mvalues[:,i]= np.sum(actual_data[:,3*i:3*i+3],axis=1)
    
    
finished_data=np.concatenate((info,mvalues),axis=1)

with open("test_data_final.pkl","wb") as f:
     pickle.dump(finished_data,f)
    
    

