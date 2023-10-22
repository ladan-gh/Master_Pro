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

#============================================================
# plt.fill(x_test, np.exp(log_dens), c='green')
# plt.show()
error = []

for i in range(1, 10):
    j = i / 10
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=j)
    model = KernelDensity()
    model.fit(x_train)
    error.append(model.score_samples(x_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plt.plot(range(1, 10), error_02, color='red', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
# plt.plot(range(1, 10), error_03, color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', markersize=10)
plt.title('Error Rate of 3 Class')
plt.xlabel('Data Percentage')
plt.ylabel('Mean Error')
plt.show()
