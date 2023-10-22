import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
from sklearn.model_selection import train_test_split

csv_file = pd.read_csv("/Pattern_Classification_pro/P_01/dataset.txt", delimiter =' ', header = None)
csv_file.columns = ['x1', 'x2', 'y']
csv_file.to_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/dataset_01.csv',index = None)

csv_file_01 = pd.read_csv('/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)


x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

#============================================================
model = KernelDensity()
model.fit(x_train)
log_dens = model.score_samples(x_test)

plt.fill(x_test, np.exp(log_dens), c='green')
plt.show()