import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#---------------------------------------
df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/dataset1.csv')
x = df['x']
y = df['y']


mean_x = x.mean()
mean_y = y.mean()

sum_yx = (x * y).sum()
x_sq = (x ** 2).sum()
n = len(x)
ssxy = sum_yx - (n * mean_x * mean_y)
ssxx = ((x - mean_x) ** 2).sum()

theta_1 = ssxy/ssxx
theta_0 = mean_y - (theta_1 * mean_x)

x_new = [6.2, 12.8, 22.1, 30]

for i in range(0, len(x_new)):

    y_pred = theta_0 + theta_1 * x_new[i]
    print("y_pred",i, "is: ",y_pred)


