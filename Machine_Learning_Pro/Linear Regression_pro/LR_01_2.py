import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/dataset1.csv')

y = df['y']
x = df.iloc[:, [0]] # x = df['x']

#--------------------------------------------
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

#----------------------------------------------------
def Gradient_Descent(x ,y):
    # print(x.shape)
    # print(type(x))

    x = np.matrix(x)
    y = np.matrix(y)

    # print(type(x))
    # print(x.shape)

    theta_new = [[0], [0]]
    theta_new = np.matrix(theta_new)


    # num of data point
    n = len(x)

    # initialize the learning rate
    l = 0.0001

    for i in range(0, len(x)):
        theta_0 = theta_new[0][0]  # Bias
        theta_1 = theta_new[1][0]

        y_pred = theta_0 + (theta_1 * x)

        theta_old = theta_new
        theta_new = theta_old + l * np.sum((y - y_pred) * x)

        cost = (1 / 2) * np.sum((y - y_pred) ** 2)  # MSE

        if theta_new == theta_old:
            break




        # find the partial derivativs for w and b
        # D_w = (-2/n) * np.sum(x * (y - y_pred))
        # D_b = (-2/n) * np.sum(y - y_pred)

    #     w = w - l * D_w
    #     b = b - l * D_b
    #     print("w: {}, b: {} cost is: {}".format(w, b, cost))
    #
    # plt.scatter(x, y)
    # plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  # regression line
    # plt.show()

#------------------------------------------------------------
def Gradient_Descent_Stochastic(x,y):
    b = w = 0
    epochs = 1500
    n = len(x)
    # random_x = []
    # random_y = []


    # initialize the learning rate
    l = 0.0001


    for i in range(epochs):

        y_pred = w * x[i]  + b
        cost = (1/n) * np.sum([val ** 2 for val in (y - y_pred)])

        # find the partial derivativs for w and b
        D_w = (-2/n) * np.sum(x1 * (y1 - y_pred))
        D_b = (-2/n) * np.sum(y1 - y_pred)

        w = w - l * D_w
        b = b - l * D_b
        print("w: {}, b: {} cost is: {}, iteration{}".format(w, b, cost, i))

    plt.scatter(x1, y1)
    plt.plot([min(x1), max(x1)], [min(y_pred), max(y_pred)], color='red') # regression line
    plt.show()

#-----------------------------------------------
def Batch_Gradient_Descent(x, y):
    learning_rate = 0.01
    epochs = 1500
    bias = 0

    # print(x.shape[0]) #sample
    # print(len(df.columns)) #column

    x = pd.DataFrame(x)
    num_features = len(x.columns)

    w = np.ones(shape=(num_features)) # w1 and w2 initialize to 1
    b = 0
    total_samples = x.shape[0]

    for i in range(epochs): # every iter use *all* training sample
        y_pred = np.dot(w, x.transpose()) + bias# pred = w1 * x1 + w2 * x2 + bias

        w_gred = -(2/total_samples) * (x.transpose().dot(y - y_pred))
        b_gred = -(2/total_samples) * np.sum(y - y_pred)

    w = w - learning_rate * w_gred
    b = b - learning_rate * b_gred

    cost = np.mean(np.square(y - y_pred)) #MSE
    print(cost)

    # return w, b, cost

#-------------------------------------------------------------------------
# Gradient_Descent(x, y)
# Gradient_Descent_Stochastic(x,y)
# Batch_Gradient_Descent(x, y)

# w,b, cost = Batch_Gradient_Descent(x, y, 100)

# link : https://www.youtube.com/watch?v=5Xz7rgxhjb4