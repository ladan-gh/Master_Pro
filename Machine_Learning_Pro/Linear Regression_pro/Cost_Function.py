import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#---------------------------------------
df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/dataset1.csv')

df['x0'] = np.ones(97)
x = df[['x', 'x0']]
y = df[['y']]

x1 = x['x']
x1 = np.matrix(x1)


x_01 = np.matrix(x) # (97, 2)
y_01 = np.matrix(y) # (97, 1)


x_transpose = np.transpose(x_01)
x_ = x_transpose * x_01
x_inverse = np.linalg.inv(x_)

y_ = x_transpose * y_01
theta = x_inverse * y_


theta_0 = theta[0][0] # Bias
theta_1 = theta[1][0]

y_pred = theta_0 + theta_1 * x1

j_theta = (1/(2 * len(x))) * np.sum((y_pred - y_01) ** 2) # MSE

print(j_theta)

# plt.scatter(x1, y_pred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()