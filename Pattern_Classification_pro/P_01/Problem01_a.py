# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd


csv_file = pd.read_csv("/Pattern_Classification_pro/P_01/dataset.txt", delimiter =' ', header = None)
csv_file.columns = ['x1', 'x2', 'y']
csv_file.to_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/dataset_01.csv',index = None)

csv_file_01 = pd.read_csv('/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)


x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30) #select randomly data for test and train

p_ = 1/3 # Common probability

#------------------------W1 prob-------------------------------
def w1_prob(x):
    mean_01 = np.array([0, 0])
    cov_01 = np.array([[4, 0], [0, 4]])
    prob_01 = multivariate_normal.pdf(x, mean_01, cov_01) # Likelihood

    return prob_01

#-------------------------W2 prob-------------------------------
def w2_prob(x):
    mean_02 = np.array([10, 0])
    cov_02 = np.array([[4, 0], [0, 4]])
    prob_02 = multivariate_normal.pdf(x, mean_02, cov_02)

    return prob_02

#-------------------------W3 prob-------------------------------
def w3_prob(x):
    mean_03 = np.array([5, 5])
    cov_03 = np.array([[5, 0], [0, 5]])
    prob_03 = multivariate_normal.pdf(x, mean_03, cov_03)

    return prob_03

#---------------------------------------------------------------
w1 = [] # class w1
w2 = [] # class w2
w3 = [] # class w3
counter1 = 0
counter2 = 0
counter3 = 0


for i in range(0, len(x_test)):
    sum_ = np.sum([w1_prob(x_test[i]) * p_, w2_prob(x_test[i]) * p_, w3_prob(x_test[i]) * p_])

    B_01 = ( p_ * w1_prob(x_test[i]) ) / sum_ # Posterior
    B_02 = ( p_ * w2_prob(x_test[i]) ) / sum_
    B_03 = ( p_ * w3_prob(x_test[i]) ) / sum_

    main_prob = np.max([B_01, B_02, B_03])

    if main_prob == B_01:
        w1.append(y_test[i])

    elif main_prob == B_02:
        w2.append(y_test[i])

    else:
        w3.append(y_test[i])

#----------------check error on each class-------------------------------------
for i in range(0, len(w1)):
    if w1[i] != 1:
        counter1 += 1

for i in range(0, len(w2)):
    if w2[i] != 2:
        counter2 += 1

for i in range(0, len(w3)):
    if w3[i] != 3:
        counter1 += 1

#---------------devide on zero-------------------------------
if len(w1) != 0:
    error_w1 = (counter1*100) / len(w1)
else:
    error_w1 = 0

if len(w2) != 0:
    error_w2 = (counter2*100) / len(w2)
else:
    error_w2 = 0

if len(w3) != 0:
    error_w3 = (counter3*100) / len(w3)
else:
    error_w3 = 0

#----------------print error-------------------------------
print(f'Error_w1: {int(error_w1)} %')
print(f'Error_w2: {int(error_w2)} %')
print(f'Error_w3: {int(error_w3)} %')