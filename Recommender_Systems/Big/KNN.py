import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#------------------------------------------
def prediction_coverage(predicted):
    if predicted != 0:
        coverage = 1
    else:
        coverage = 0

    return coverage

#------------------------------------------
def Mean_Absolute_Error(true, predicted):
    sum_ = 0

    if type(true) is list:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true[i])

    else:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true.iloc[i])

    return sum_
#------------------------------------------------------------
def predict_user_KNN(user_id, item_id, rating_, k):
    threshold = 0
    # Using similarity matrix....
    user_similarity = pd.read_csv('/home/401156007/Master/pearson_matrix.csv')

    neighbor_indices = np.zeros(k)
    count = 0

    for index, row in user_similarity.iterrows():
        if row[user_id] >= threshold and index != user_id:
            neighbor_indices[count] = index  # col
            count += 1
            if count == k:
                break

    # Get the ratings of the neighbors for the given item
    neighbor_ratings = rating_.loc[
        (rating_['userId'].isin(neighbor_indices)) & (rating_['movieId'] == item_id), 'rating'].tolist()
    neighbor_indices = rating_.loc[
        (rating_['userId'].isin(neighbor_indices)) & (rating_['movieId'] == item_id), 'userId'].tolist()

    sim = []
    for i in neighbor_indices:
        for j in user_similarity.columns:
            if i == int(j):
                sim.append(user_similarity.iloc[user_id, i])

    sum_ = 0
    # Calculate the predicted rating
    if len(neighbor_ratings) > 0:

        for i in range(len(sim)):
            sum_ += (neighbor_ratings[i] * sim[i])

        predicted_rating = sum_ / sum(sim)
    else:
        predicted_rating = 0

    return predicted_rating

#----------------------------------------------
# **Main Program**
ratings_df = pd.read_csv('/home/401156007/Master/ratings.csv')
ratings_df = ratings_df.sort_values('timestamp', ascending=True)

t1 = pd.read_csv('/home/401156007/Master/0.csv')
t2 = pd.read_csv('/home/401156007/Master/1.csv')
t3 = pd.read_csv('/home/401156007/Master/2.csv')
t4 = pd.read_csv('/home/401156007/Master/3.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('/home/401156007/Master/5.csv')


# **Plot MAE for KNN**
#-------------------------------------------------------
MAE_knn = []
for k in range(1, 26):

    # ------------------Calcute MAE---------------------------
    rate_item = []
    item = []
    sum_ = 0

    for i in train['userId'].unique():
        for j in train['movieId'].unique():
            pred = predict_user_KNN(i, j, train, k)  # predicted

            for h in range(0, len(test)):
                if test['userId'][h] == i:
                    if test['movieId'][h] == j:
                        rate_item.append(test['rating'][h])

        item.append(Mean_Absolute_Error(rate_item, pred))

    for i in range(0, len(item)):
        sum_ += item[i]

    value = sum_ / len(test)
    MAE_knn.append(value)

# **Plot Coverage for KNN**
# ------------------Calcute Coverage---------------------------
cov_knn = []

for k in range(1, 26):
    rate_item = []
    item = []
    sum_ = 0

    for i in train['userId'].unique():
        for j in train['movieId'].unique():
            pred = predict_user_KNN(i, j, train, k)  # predicted
            item.append(prediction_coverage(pred))

    for i in range(0, len(item)):
        sum_ += item[i]

    value = sum_ / len(test)
    cov_knn.append(value)

#======================================
k = []
for i in range(1, 26):
    k.append(i)

# --------------Plot MAE---------------------
df = pd.DataFrame(MAE_knn)
df.to_csv('/home/401156007/Master/MAE_knn.csv')
df_01 = pd.read_csv('/home/401156007/Master/MAE_knn.csv')

# Select the second column of the DataFrame
col = df_01.iloc[:, 1]

col.plot(kind='bar')
plt.title('MAE Error(KNN)')

# Set the labels for the x-axis and y-axis
plt.xlabel('Index')
plt.ylabel('Value')

plt.savefig('/home/401156007/Master/MAE_knn.jpg')

plt.show()

# --------------Plot Coverage---------------------
df = pd.DataFrame(cov_knn)
df.to_csv('/home/401156007/Master/cov_knn.csv')
df_01 = pd.read_csv('/home/401156007/Master/cov_knn.csv')

# Select the second column of the DataFrame
col = df_01.iloc[:, 1]

col.plot(kind='bar')

plt.title('Covearge Error(KNN)')
plt.xlabel('Index')
plt.ylabel('Value')

plt.savefig('/home/401156007/Master/cov_knn.jpg')

plt.show()