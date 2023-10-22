import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


csv_file = pd.read_csv("/Pattern_Classification_pro/P_01/dataset.txt", delimiter =' ', header = None)
csv_file.columns = ['x1', 'x2', 'y']
csv_file.to_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Pattern_Classification_pro/dataset_01.csv',index = None)

csv_file_01 = pd.read_csv('/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)


x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#ac = accuracy_score(y_test, y_pred) # Accuracy
#------------------------------------------------------------
error = np.mean(y_pred != y_test) # ME
print("Error is:", error)
