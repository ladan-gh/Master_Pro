import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier

#----------------------------------------------------------
csv_file_01 = pd.read_csv('/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)

x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.90) # test data random---> 30% from total

#------------percent_-----------------------------------------
# def percent_(class_, num):
#
#     percent = int(len(class_) * num)
#     DF = class_.sample(n=percent)
#
#     return DF

#---------------------------------------------------------------
class_ = df.groupby(['y'])
class_01 = class_.get_group(1)
class_02 = class_.get_group(2)
class_03 = class_.get_group(3)

# lable_01 = class_01['y']
# lable_02 = class_02['y']
# lable_03 = class_03['y']


#-----------10% of each class------------
# _10_percent_01 = percent_(class_01, 0.1)
# _10_percent_02 = percent_(class_02, 0.1)
# _10_percent_03 = percent_(class_03, 0.1)
# random_train_01 = pd.concat((_10_percent_01, _10_percent_02, _10_percent_03), axis=0)
# # random_train_01 = random_train_01[['x1', 'x2']]
#
# lable_01 = class_01['y']
# lable_02 = class_02['y']
# lable_03 = class_03['y']
#
# l_01 = lable_01.iloc[:len(_10_percent_01)]
# l_02 = lable_02.iloc[:len(_10_percent_02)]
# l_03 = lable_03.iloc[:len(_10_percent_03)]
#
# lable_ = pd.concat((l_01, l_02, l_03), axis=0)


#--------------------classification num 1(MLE)---------------------------------
p_ = 1/3 # Common probability

#-----------------------Case b---------------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------
def w_prob(x, cov_01, mean_01):
    prob_01 = multivariate_normal.pdf(x, mean_01, cov_01) # Likelihood
    return prob_01

#--------------------------------------------------------------
w1 = [] # class w1
w2 = [] # class w2
w3 = [] # class w3
counter1 = 0
counter2 = 0
counter3 = 0
#---------------------------------------------------------------
# mean and cov of each class
cov1, mean1 = mean_cov(np.array(class_01)) # w1
cov2, mean2 = mean_cov(np.array(class_02)) # w2
cov3, mean3 = mean_cov(np.array(class_03)) # w3

#---------------------------------------------------------------
for i in range(0, len(x_test)):
    sum_ = np.sum([w_prob(x_test[i], cov1, mean1) * p_, w_prob(x_test[i], cov2, mean2) * p_, w_prob(x_test[i], cov3, mean3) * p_])

    B_01 = ( p_ * w_prob(x_test[i], cov1, mean1) ) / sum_ # Posterior
    B_02 = ( p_ * w_prob(x_test[i], cov2, mean2) ) / sum_
    B_03 = ( p_ * w_prob(x_test[i], cov3, mean3) ) / sum_

    main_prob = np.max([B_01, B_02, B_03])

    if main_prob == B_01.max():
        w1.append(y_test[i])
        if y_test[i] != y[i]:
            counter1 += 1

    elif main_prob == B_02.max():
        w2.append(y_test[i])
        if y_test[i] != y[i]:
            counter2 += 1

    else:
        w3.append(y_test[i])
        if y_test[i] != y[i]:
            counter3 += 1

error_w1 = (counter1*100) / len(x_test)
error_w2 = (counter2*100) / len(x_test)
error_w3 = (counter3*100) / len(x_test)

print(f'Error_w1: {int(error_w1)} %')
print(f'Error_w2: {int(error_w2)} %')
print(f'Error_w3: {int(error_w3)} %')

#-----------------------Case c---------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.neighbors import KernelDensity
#
#
# model = KernelDensity()
# model.fit(x_train)
# log_dens = model.score_samples(x_test)
#
# plt.fill(x_test, np.exp(log_dens), c='green')
# plt.show()
#-----------------------Case d---------------------------------------------------------------------------------------------------
# classifier = KNeighborsClassifier(n_neighbors = 1)
#
# classifier = classifier.fit(random_train_01, lable_)
# y_pred = classifier.predict(x_test)
#
# # ac = accuracy_score(y_test, y_pred) # Accuracy
# error = np.mean(y_pred != y_test) # ME
# print(error)