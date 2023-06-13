import matplotlib.pyplot as plt
import numpy as np
import pickle
import model_acc as m


# DATA FOR 3 months predictions

#Data read from files

with open("arima3m.pkl","rb") as f:
    data1=pickle.load(f)
    
arim=m.errors(data1[2],data1[1])
    
arima= arim.experrors()

with open("naive3m.pkl","rb") as f:
    data1=pickle.load(f)
    
naiv=m.errors(data1[2],data1[1])
    
naive= naiv.experrors()

with open("average3m.pkl","rb") as f:
    data1=pickle.load(f)

aver=m.errors(data1[2],data1[1])

average=aver.experrors()



plt.style.use('_mpl-gallery')






data3 = np.random.normal(0, 1.5, 100)
data4 = np.random.normal(0, 1, 100)

# Combine the data into a list
#boxblots
data = [arima[2], naive[2], average[2], data4]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

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
ax.set_xticklabels(['Arima', 'Naive', 'Average', 'Data 4'])

# Set the y-axis label
ax.set_ylabel('Value')
ax.set_ylim([0, 1])
ax.set_yticks(np.arange(0, 1.5, 0.1))
# Set the title
ax.set_title('MAPE Error')

# Show the plot
plt.show()





x=["Naive","Arima","Average"]
y=[np.sum(naive[1]),np.sum(arima[1]),np.sum(average[1])]

fig1, ax1 = plt.subplots(dpi=100)
ax1.bar(x, y, color='blue', edgecolor='white', linewidth=0.5, alpha=1)

ax.set_xlabel('Categories')
ax.set_ylabel('Values')


ax1.set_title('Bar Plot Example')

plt.show()




#Data reads for 1 month predictions










