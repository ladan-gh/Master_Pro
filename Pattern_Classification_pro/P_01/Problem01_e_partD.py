import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#----------------------------------------------------------
csv_file_01 = pd.read_csv('/Pattern_Classification_pro/P_01/dataset_01.csv')
df = pd.DataFrame(csv_file_01)

x = df.iloc[:, 0:2].values #rows
y = df.iloc[:, 2].values #lable

#-----------------------Case d---------------------------------------------------------------------------------------------------
error = []

for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=1)
    j = i / 10
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=j)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    error.append(np.mean(y_pred != y_test))

# ac = accuracy_score(y_test, y_pred) # Accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate')
plt.xlabel('Data Percentage')
plt.ylabel('Mean Error')
plt.show()