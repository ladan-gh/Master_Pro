import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


csv_file = pd.read_csv("C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/P_01/dataset.txt", delimiter =' ', header = None)
csv_file.columns = ['x1', 'x2', 'y']
csv_file.to_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/P_01/dataset_01.csv',index = None)

csv_file_01 = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)

x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=None)

#-------------------------mean and cov of each class----------------------------------
def mean_cov(x_train):

    sum_ = np.sum(x_train)
    mean_ = sum_ / len(x_train)  # mean of train set

    multiple_ = []
    for i in range(0, len(x_train)):
        substract_1 = np.subtract(x_train[i], mean_)
        substract_2 = np.transpose(substract_1)
        multiple_.append(substract_1 * substract_2)

    cov_ = np.sum(multiple_) / len(x_train)  # covariance of train set

    return cov_, mean_

#-----------------------------------------------------------
def cov_(x_train_01, x_train_02):

    sum_01 = np.sum(x_train_01)
    sum_02 = np.sum(x_train_02)

    mean_01 = sum_01 / len(x_train_01)  # mean of train set
    mean_02 = sum_02 / len(x_train_02)

    multiple_01 = []
    sum_ = []
    for i in range(0, len(x_train_01)):
        substract_1 = np.subtract(x_train_01[i], mean_01)
        substract_2 = np.subtract(x_train_02[i], mean_02)
        sum_.append(substract_1 * substract_2)

    cov_ = np.sum(sum_) / len(x_train_01)

    return cov_

#----------------------------------------------------------------------
# Find the indices of the samples in y_train that are one, two and three
idx_one = np.where(y_train == 1)[0]
idx_two = np.where(y_train == 2)[0]
idx_three = np.where(y_train == 3)[0]

# split to three class
x_train_one = x_train[idx_one]
y_train_one = y_train[idx_one]
x_train_one = pd.DataFrame(x_train_one)
y_train_one = pd.DataFrame(y_train_one)
x1_01 = x_train_one[0] # x1 for class 1
x2_01 = x_train_one[1] # x2 for class 1
cov1_x11, mean1_x11 = mean_cov(np.array(x1_01)) # cov x1, x1
cov1_x21, mean1_x21 = mean_cov(np.array(x2_01)) # cov x2, x2
cov_x11_x21 = cov_(x1_01, x2_01) # cov x1, x2
cov1 = np.array([[cov1_x11, cov_x11_x21],[cov_x11_x21, cov1_x21]])
mean1 = np.array([mean1_x11, mean1_x21])

# print('cov func is:')
# print(np.cov(x1_01, x2_01))

#-------------------------------------------------------------
x_train_two = x_train[idx_two]
y_train_two = y_train[idx_two]
x_train_two = pd.DataFrame(x_train_two)
y_train_two = pd.DataFrame(y_train_two)
x1_02 = x_train_two[0]
x2_02 = x_train_two[1]
cov1_x12, mean1_x12 = mean_cov(np.array(x1_02)) # cov x1, x1
cov1_x22, mean1_x22 = mean_cov(np.array(x2_02)) # cov x2, x2
cov_x1_x22 = cov_(x1_02, x2_02) # cov x1, x2
cov2 = np.array([[cov1_x12, cov_x1_x22],[cov_x1_x22, cov1_x22]])
mean2 = np.array([mean1_x12, mean1_x22])

#------------------------------------------------------------------
x_train_three = x_train[idx_three]
y_train_three = y_train[idx_three]
x_train_three = pd.DataFrame(x_train_three)
y_train_three = pd.DataFrame(y_train_three)
x1_03 = x_train_three[0]
x2_03 = x_train_three[1]
cov1_x13, mean1_x13 = mean_cov(np.array(x1_03)) # cov x1, x1
cov1_x23, mean1_x23 = mean_cov(np.array(x2_03)) # cov x2, x2
cov_x1_x23 = cov_(x1_03, x2_03) # cov x1, x2
cov3 = np.array([[cov1_x13, cov_x1_x23],[cov_x1_x23, cov1_x23]])
mean3 = np.array([mean1_x13, mean1_x23])

#-------------------------------------------------------------
p_ = 1/3 # Common probability

#------------------------W1 prob-------------------------------
def w_prob(x, cov_, mean_):
    prob_ = multivariate_normal.pdf(x, mean_, cov_) # Likelihood
    return prob_

#--------------------------------------------------------------
counter = 0
w1 = [] # class w1
w2 = [] # class w2
w3 = [] # class w3
counter1 = 0
counter2 = 0
counter3 = 0


for i in range(0, len(x_test)):
    sum_ = np.sum([w_prob(x_test[i], cov1, mean1) * p_, w_prob(x_test[i], cov2, mean2) * p_, w_prob(x_test[i], cov3, mean3) * p_])

    B_01 = ( p_ * w_prob(x_test[i], cov1, mean1) ) / sum_ # Posterior
    B_02 = ( p_ * w_prob(x_test[i], cov2, mean2) ) / sum_
    B_03 = ( p_ * w_prob(x_test[i], cov3, mean3) ) / sum_

    main_prob = np.max([B_01, B_02, B_03])

    if main_prob == B_01.max():
        w1.append(y_test[i])

    elif main_prob == B_02.max():
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

#---------------devide zero-------------------------------
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