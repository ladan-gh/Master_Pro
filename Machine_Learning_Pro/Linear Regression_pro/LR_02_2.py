import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics

#-------------------------------------------------------
csv_file_01 = pd.read_csv("C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/train.csv")
df_01 = pd.DataFrame(csv_file_01)

csv_file_02 = pd.read_csv("C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/Linear Regression_pro/test.csv")
df_02 = pd.DataFrame(csv_file_02)

#---------------------
# integer encode-->train data
label_encoder = LabelEncoder()
integer_encoded_01 = label_encoder.fit_transform(df_01['gender'])
integer_encoded_02 = label_encoder.fit_transform(df_01['smoker'])
integer_encoded_03 = label_encoder.fit_transform(df_01['region'])

df_01['gender'] = integer_encoded_01.reshape(len(integer_encoded_01), 1)
df_01['smoker'] = integer_encoded_02.reshape(len(integer_encoded_02), 1)
integer_encoded_03 = integer_encoded_03.reshape(len(integer_encoded_03), 1)

#---------------------
# integer encode-->test data
integer_encoded_04 = label_encoder.fit_transform(df_02['gender'])
integer_encoded_05 = label_encoder.fit_transform(df_02['smoker'])
integer_encoded_06 = label_encoder.fit_transform(df_02['region'])

df_02['gender'] = integer_encoded_04.reshape(len(integer_encoded_04), 1)
df_02['smoker'] = integer_encoded_05.reshape(len(integer_encoded_05), 1)
integer_encoded_06 = integer_encoded_06.reshape(len(integer_encoded_06), 1)
#---------------------
# One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
df_01['region'] = onehot_encoder.fit_transform(integer_encoded_03) # OneHot encode-->train data
df_02['region'] = onehot_encoder.fit_transform(integer_encoded_06)# OneHot encode-->test data

#-----------------------------------------------------------
c = 0
# train
x_01 = df_01[['age','gender','bmi','children','smoker','region']]
y_01 = df_01['charges']

# test
x_test = df_02[['age','gender','bmi','children','smoker','region']]
y_test = df_02['charges']


for i in range(0, 1000):
    x_train = x_01[0:c+10]
    y_train = y_01[0:c+10]
    c += 10
    if c == 1000:
        break

    reg = linear_model.LinearRegression()

    # train the model using the training sets
    reg.fit(x_train, y_train)

    # regression coefficients
    print('Theta: ', reg.coef_)

    ## plotting residual errors in training data
    plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color="green", s=10, label='Train data')

    ## plotting residual errors in test data
    plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color="blue", s=10, label='Test data')

    ## plotting line for zero residual error
    plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

    ## plotting legend
    plt.legend(loc='upper right')

    ## plot title
    plt.title("Residual errors")

    ## method call for showing the plot
    plt.show()