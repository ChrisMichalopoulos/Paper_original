import matplotlib.pyplot as plt
import numpy as np
import pickle
import model_acc as m

# DATA FOR 3 months predictions

#Data read from files

with open("average3m.pkl","rb") as f:
    data1=pickle.load(f)

aver=m.errors(data1[2],data1[1])

average=aver.experrors()

with open("naive3m.pkl","rb") as f:
    data1=pickle.load(f)
    
naiv=m.errors(data1[2],data1[1])
    
naive= naiv.experrors()


with open("arima3m.pkl","rb") as f:
    data1=pickle.load(f)

data1[1][data1[1] < 0]=0
arim=m.errors(data1[2],data1[1])
    
arima= arim.experrors()


with open("sarima3m.pkl","rb") as f:
    data1=pickle.load(f)
    
data1[1][data1[1] < 0]=0    
sarim=m.errors(data1[2],data1[1])
    
sarima= sarim.experrors()



with open("autoarim3m.pkl","rb") as f:
    data1=pickle.load(f)

data1[1][data1[1] < 0]=0
auto=m.errors(data1[2],data1[1])

autoarima=auto.experrors()



with open("lstm3m.pkl","rb") as f:
    data1=pickle.load(f)

data1[1][data1[1] < 0]=0
lst=m.errors(data1[2],data1[1])

lstm=lst.experrors()   #it is not finished yet waiting for the final data

with open("knn3m.pkl", "rb") as f:
    data1=pickle.load(f)

knn=m.errors(data1[2][:30000,:],data1[1])

knner=knn.experrors()


plt.style.use('_mpl-gallery')








# Combine the data into a list
#boxblots
data = [ average[2], naive[2],arima[2],sarima[2],autoarima[2],lstm[2],knner[2]]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 7), dpi=1000)

# Create the boxplot
ax.boxplot(data,widths=0.5, patch_artist=True,
                showmeans=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 1},
                boxprops={"facecolor": "white", "edgecolor": "black",
                          "linewidth": 1},
                whiskerprops={"color": "black", "linewidth": 1.5},
                capprops={"color": "black", "linewidth": 1.5},
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})



# Set the x-axis tick labels
ax.set_xticklabels(["Αverage","Naive","ARIMA","SARIMA","AutoARIMA","LSTM","KNN"],fontsize=12)


# Set the y-axis label
ax.set_ylabel('Value',fontsize=12)
ax.set_ylim([0, 1])
ax.set_yticks(np.arange(0, 1, 0.1))

ax.tick_params(axis='y', labelsize=12)
# Set the title
ax.set_title('MAPE Quarter-year Analysis',fontsize=16)

# Show the plot
plt.show()

"----------------------------------------------------------------------------------------------------------------------"

#Bar plots

x=["Average","Naive","ARIMA","SARIMA", "AutoARIMA","LSTM","KNN"]
y=[average[1],naive[1],arima[1],23392,autoarima[1],lstm[1],knner[1]]

fig1, ax1 = plt.subplots(dpi=1000,figsize=(8, 7))
ax1.bar(x, y, color='gray', edgecolor='white', linewidth=0.5, alpha=1)

for i, value in enumerate(y):
    plt.text(i,value,str(int(value)),ha="center",va="bottom")



ax1.tick_params(axis='y', labelsize=12)

ax1.set_ylabel('Value',fontsize=12)

ax1.set_ylim([0, 2000000])
ax1.set_title('Total Balance quarter-year',fontsize=16)

plt.show()

"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"

"""#Data reads for 1 month predictions"""

with open("sarima1m.pkl","rb") as f:
    data1=pickle.load(f)
data1[1][data1[1] < 0]=0
sarim=m.errors(data1[2],data1[1])
    
sarima1m= sarim.plotperquarter(month=1)



with open("knn1m.pkl","rb") as f:
    data1=pickle.load(f)
data1[1][data1[1] < 0]=0
knn=m.errors(data1[2],data1[1])
    
knn1m= knn.plotperquarter(month=1)


with open("arima1m.pkl","rb") as f:
    data1=pickle.load(f)
data1[1][data1[1] < 0]=0
arim1m=m.errors(data1[2],data1[1])
    
arima1m= arim1m.plotperquarter(month=1)



with open("autoarima1m.pkl","rb") as f:
    data1=pickle.load(f)
data1[1][data1[1] < 0]=0
autoarim1m=m.errors(data1[2],data1[1])
    
autoarima1m= autoarim1m.plotperquarter(month=1)

with open("lstm1m.pkl","rb") as f:
    data1=pickle.load(f)
data1[1][data1[1] < 0]=0
lstmm=m.errors(data1[2],data1[1])
    
lstm1m= lstmm.plotperquarter(month=1)

   

#plot
plt.style.use('_mpl-gallery')


# data3 = np.random.normal(0, 1.5, 100)
# data4 = np.random.normal(0, 1, 100)

# Combine the data into a list
#boxblots
data = [average[2], naive[2],np.average(arima1m[0],axis=1),np.average(sarima1m[0],axis=1),np.average(autoarima1m[0],axis=1), np.average(lstm1m[0],axis=1),np.average(knn1m[0],axis=1) ]



# Create a figure and axis
fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=1000)

# Create the boxplot
ax2.boxplot(data,widths=0.5, patch_artist=True,
                showmeans=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 1},
                boxprops={"facecolor": "white", "edgecolor": "black",
                          "linewidth": 1},
                whiskerprops={"color": "black", "linewidth": 1.5},
                capprops={"color": "black", "linewidth": 1.5},
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})



# Set the x-axis tick labels
ax2.set_xticklabels(['Average', 'Naive',"ARIMA", 'SARIMA',"ΑutoARIMA","LSTM","KNN"],fontsize=12)

# Set the y-axis label
ax2.set_ylabel('Value',fontsize=12)
ax2.set_ylim([0, 1])
ax2.set_yticks(np.arange(0, 1, 0.1))
ax.tick_params(axis='y', labelsize=12)
# Set the title
ax2.set_title('MAPE Μonthly Αnalysis',fontsize=16)

# Show the plot
plt.show()



"----------------------------------------------------------------------------------------------------------------------"

#Bar plots

u=["Average","Naive","ARIMA","SARIMA","AutoARIMA","LSTM","KNN"]
v=[np.sum(average[1]),np.sum(naive[1]),np.sum(arima1m[1]),np.sum(sarima1m[1]),np.sum(autoarima1m[1]),np.sum(lstm1m[1]),np.sum(knn1m[1])]
v=[abs(i) for i in v]
fig3, ax3 = plt.subplots(dpi=1000,figsize=(8, 7))
ax3.bar(u, v, color='grey', edgecolor='white', linewidth=0.5, alpha=1)
for i, value in enumerate(v):
    plt.text(i,value,str(int(value)),ha="center",va="bottom")

ax3.set_ylabel('Values',fontsize=12)
ax3.set_ylim([0, 2000000])
ax3.set_ylabel('Value',fontsize=12)
ax3.set_title('Total Balance monthly',fontsize=16)

plt.show()



"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
"""seasonality performance"""

"""3month period"""




plt.style.use('_mpl-gallery')




colors=["Black","blue","green","Grey"]

# Define the data for the bar plot
categories = ['Average', 'Naive',"ARIMA","SARIMA","AutoARIMA","LSTM","KNN"]
values1 = [average[4][1],naive[4][1], np.average(arima1m[0][:,1]),      np.average(sarima1m[0][:,1]) , np.average(autoarima1m[0][:,1]), np.average(lstm1m[0][:,1]),np.average(knn1m[0][:,1])]

values2 = [average[4][2],naive[4][2], np.average(arima1m[0][:,2]),      np.average(sarima1m[0][:,2]) , np.average(autoarima1m[0][:,2]), np.average(lstm1m[0][:,2]),np.average(knn1m[0][:,2])]

values3 = [average[4][3],naive[4][3], np.average(arima1m[0][:,3]),      np.average(sarima1m[0][:,3]) , np.average(autoarima1m[0][:,3]), np.average(lstm1m[0][:,3]),np.average(knn1m[0][:,3])]

values4 = [average[4][0],naive[4][0], np.average(arima1m[0][:,0]),      np.average(sarima1m[0][:,0]) , np.average(autoarima1m[0][:,0]), np.average(lstm1m[0][:,0]),np.average(knn1m[0][:,0])]


# Set the width of each bar
bar_width = 0.2

# Set the positions of the bars within the cluster
positions = np.arange(len(categories))

# Create a figure and axis
fig, ax = plt.subplots(dpi=1000,figsize=(8, 7))

# Create the bars for each category
ax.bar(positions - bar_width, values1, width=bar_width, label='1st quarter',color=colors[0])
ax.bar(positions, values2, width=bar_width, label='2nd quarter',color=colors[1])
ax.bar(positions + bar_width, values3, width=bar_width, label='3rd quarter',color=colors[2])
ax.bar(positions + (2 * bar_width), values4, width=bar_width, label='4th quarter',color=colors[3])

# Set the x-axis tick positions and labels
ax.set_xticks(positions)
ax.set_xticklabels(categories,fontsize=12)

# Set the y-axis label
ax.set_ylabel('Value',fontsize=12)
ax.tick_params(axis='y', labelsize=12)
# Set the title
ax.set_title('Error per quarter year',fontsize=16)
ax.set_ylim([0, 1])
# Add a legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Show the plot
plt.show()



"----------------------------------------------------------------------------------------------------------------------"









