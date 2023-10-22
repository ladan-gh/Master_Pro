import numpy as np
#----------------------2.a-------------------------------------------
class LinearRegressor:
    def __init__(self, learning_rate, n):
        self.learning_rate = learning_rate # Alpha
        self.n = n
        self.w = 1
        self.b = 0

    def hypothesis_fn(self, feature):
        return self.w * feature + self.b # w.x + b

    def MSE(self, y_pred, y): #MSE for linear regression
        mse = np.sum(((y - y_pred) ** 2), axis=0) / self.n
        #mse = np.square(np.subtract(y, y_pred)).mean()
        return mse

    def gd_fn(self, feature, y_pred, y):# The derivative of w and b
        w_prime = (np.sum((y - y_pred) * feature)) / self.n
        b_prime = (np.sum((y - y_pred) * 1)) / self.n
        return w_prime, b_prime

    def fit(self, iteration, feature, y):
        loss_l = []

        for i in range(iteration):
            y_pred = self.hypothesis_fn(feature)
            loss = self.MSE(y_pred, y)
            ###############
            print(loss) ###
            ###############
            w_prime, b_prime = self.gd_fn(feature, y_pred, y)

            # Update furmula
            self.w = self.w - self.learning_rate * w_prime
            self.b = self.b - self.learning_rate * b_prime

            loss_l.append(loss)

        return loss_l, self.w, self.b

#---------------------read dataset------------------------------------------
import pandas as pd

df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/dataset1.csv')

x = df['x']
y = df['y']
#---------------------------------------------------------------------------
# plot loss
import matplotlib.pyplot as plt

# n = 97 # num of samples
# alpha = 0.001
#
# model = LinearRegressor(learning_rate=alpha, n=n)
# loss_lst, w, b = model.fit(10, feature=x, y=y)

# plt.plot(loss_lst, label='Loss Curve')
# plt.legend()
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()

#-------------------------------------2.b-----------------------------------------
#-------------------plot data-------------------------------
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#--------------------------GD(batch mode)---------------------------------------
def compute_cost(feature, y, theta):
    num_sample = len(feature)

    for x,y1 in zip(feature, y):
        y_hat = np.dot(theta, np.array([1.0, feature]))
        # cost_sum = cost_sum + ((y - y_hat) ** 2)
        cost_sum = np.sum((y_hat - y1) ** 2) # loss

    cost = cost_sum / (num_sample * 2.0) # (1/(2m)) * (sum(h(x) - y) ** 2) = J(theta) in GD

    return cost
#*******************************************
def linear_regression_batch_GD(feature, y, theta, alpha, max_iter):
    iteration = 0
    num_sample = len(feature)
    cost = np.zeros(max_iter)
    theta_store = np.zeros([2, max_iter])

    while iteration < max_iter:
        cost[iteration] = compute_cost(feature, y, theta)
        theta_store[:, iteration] = theta

        print('#===================') # print cost per iteration
        print(f' iteration: {iteration}')
        print(f' cost: {cost[iteration]}')

        for x,y1 in zip(feature, y): # batch descent slope --> loop on our dataset
            y_hat = np.dot(theta, np.array([1.0, x]))
            gradient = np.array([1.0, x]) * (y1 - y_hat)
            #theta = np.add(alpha * gradient / num_sample, theta)
            theta += alpha * gradient / num_sample
        iteration += 1
    return theta, cost, theta_store

#========================SGD=====================================
def SGD(feature, y, theta, alpha):
    num_sample = len(feature)
    cost = np.zeros(num_sample)
    theta_store = np.zeros([2, num_sample])
    iteration = 0

    for x,y1 in zip(feature, y):
        cost[iteration] = compute_cost(feature, y1, theta)
        theta_store_batch[:, iteration] = theta

        print('#===================')  # print cost per iteration
        print(f' iteration: {iteration}')
        print(f' cost: {cost[iteration]}')

        y_hat = np.dot(theta, np.array([1.0, x]))
        gradient = np.array([1.0, x]) * (y1 - y_hat)
        theta += alpha * gradient / num_sample

        iteration = iteration + 1
    return theta, cost, theta_store

#*****************Train the regression model*******************
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

theta_0 = np.array([20.0, 80.0])

alpha_batch = 1e-3
max_iter = 1500

theta_hat_batch, cost_batch, theta_store_batch =\
linear_regression_batch_GD(x_train, y_train, theta_0, alpha_batch, max_iter)


# theta_hat_batch, cost_batch, theta_store_batch = SGD(x_train, y_train, theta_0, alpha_batch)