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



with open("autoarim3m.pkl","rb") as f:
    data1=pickle.load(f)
#sarima must rerun only total sum here
auto=m.errors(data1[2],data1[1])

autoarima=auto.experrors()



with open("lstm3m.pkl","rb") as f:
    data1=pickle.load(f)

lst=m.errors(data1[2],data1[1])

lstm=lst.experrors()


plt.style.use('_mpl-gallery')






data3 = np.random.normal(0, 1.5, 100)
data4 = np.random.normal(0, 1, 100)

# Combine the data into a list
#boxblots
data = [arima[2], naive[2], average[2],autoarima[2],lstm[2]]

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
ax.set_xticklabels(['Arima', 'Naive', 'Average',"autoarima","lstm"])

# Set the y-axis label
ax.set_ylabel('Value')
ax.set_ylim([0, 1])
ax.set_yticks(np.arange(0, 1.5, 0.1))
# Set the title
ax.set_title('MAPE Error - 3 months')

# Show the plot
plt.show()

"-------------------------------------------------------------------------------------------"

#Bar plots

x=["Naive","Arima","Average","Sarima"]
y=[np.sum(naive[1]),np.sum(arima[1]),np.sum(average[1]),np.sum(data1[1])]

fig1, ax1 = plt.subplots(dpi=100,figsize=(8, 6))
ax1.bar(x, y, color='gray', edgecolor='white', linewidth=0.5, alpha=1)

ax.set_xlabel('Categories')
ax.set_ylabel('Models')


ax1.set_title('Bar Plot Example')

plt.show()

"-------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------"

"""#Data reads for 1 month predictions"""

with open("autoarima1.pkl","rb") as f:
    data1=pickle.load(f)
    
# autoarim1m=m.errors(data1[2],data1[1])
    
# autoarima1m= autoarim1m.experrors()



#plot
plt.style.use('_mpl-gallery')


data3 = np.random.normal(0, 1.5, 100)
data4 = np.random.normal(0, 1, 100)

# Combine the data into a list
#boxblots
data = [data1[0][2], naive[2], average[2]]

# Create a figure and axis
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)

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
ax2.set_xticklabels(['Autoarima', 'Naive', 'Average'])

# Set the y-axis label
ax2.set_ylabel('Value')
ax2.set_ylim([0, 1])
ax2.set_yticks(np.arange(0, 1.5, 0.1))
# Set the title
ax2.set_title('MAPE Error - 1 month')

# Show the plot
plt.show()



"-------------------------------------------------------------------------------------------"

#Bar plots

u=["Naive","Average","Autoarima"]
v=[np.sum(naive[1]),np.sum(average[1]),np.sum(data1[0][1])]

fig3, ax3 = plt.subplots(dpi=100,figsize=(8, 6))
ax3.bar(u, v, color='blue', edgecolor='white', linewidth=0.5, alpha=1)

ax3.set_xlabel('Models')
ax3.set_ylabel('Values')


ax3.set_title('Total sum -1 month-')

plt.show()



"-------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------"
"""seasonality performance"""

"""3month period"""




plt.style.use('_mpl-gallery')






da1 = np.random.normal(0, 1.5, 100)
da2 = np.random.normal(0, 2, 100)
da3= np.random.normal(0, 3, 100)
da4= np.random.normal(0, 1, 100)

da11 = np.random.normal(0, 1.5, 100)
da12 = np.random.normal(0, 3.5, 100)
da13 = np.random.normal(0, 1, 100)
da14 = np.random.normal(0, 2, 100)



colors=["Black","blue","green","Grey"]

# Define the data for the bar plot
categories = ['Model 1', 'Model 2']
values1 = [np.average(da1), np.average(da11)]
values2 = [np.average(da2), np.average(da12)]
values3 = [np.average(da3), np.average(da13)]
values4 = [np.average(da4), np.average(da14)]

# Set the width of each bar
bar_width = 0.2

# Set the positions of the bars within the cluster
positions = np.arange(len(categories))

# Create a figure and axis
fig, ax = plt.subplots(dpi=1000)

# Create the bars for each category
ax.bar(positions - bar_width, values1, width=bar_width, label='Bar 1',color=colors[0])
ax.bar(positions, values2, width=bar_width, label='Bar 2',color=colors[1])
ax.bar(positions + bar_width, values3, width=bar_width, label='Bar 3',color=colors[2])
ax.bar(positions + (2 * bar_width), values4, width=bar_width, label='Bar 4',color=colors[3])

# Set the x-axis tick positions and labels
ax.set_xticks(positions)
ax.set_xticklabels(categories)

# Set the y-axis label
ax.set_ylabel('Value')

# Set the title
ax.set_title('Mape')

# Add a legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Show the plot
plt.show()



"-------------------------------------------------------------------------------------------"









