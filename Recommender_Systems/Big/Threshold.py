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

#------------------------------------------------------------
def Mean_Absolute_Error(true, predicted):
    sum_ = 0

    if type(true) is list:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true[i])

    else:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true.iloc[i])

    return sum_

#---------------------------------------------
def predict_user_threshold(user_id, item_id, rating_, theta):

    # Load the similarity matrix
    user_similarity = pd.read_csv('/home/401156007/Master/pearson_matrix.csv')

    # Get the indices of the users who are similar to the given user
    neighbor_indices = []
    for index, row in user_similarity.iterrows():
        if row[user_id] >= theta and index != user_id:
            neighbor_indices.append(index)#col


    # Get the ratings of the neighbors for the given item
    neighbor_ratings = rating_.loc[(rating_['userId'].isin(neighbor_indices)) & (rating_['movieId'] == item_id), 'rating'].tolist()
    neighbor_indices = rating_.loc[(rating_['userId'].isin(neighbor_indices))& (rating_['movieId'] == item_id), 'userId'].tolist()

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


# **Plot MAE for Threshold**
#-----------------------------------------------------------
MAE_Threshold = []
theta = [i / 10 for i in range(-10, 11)]

for t in theta:

    # ------------------Calcute MAE---------------------------
    rate_item = []
    item = []
    sum_ = 0

    for i in train['userId'].unique():
        for j in train['movieId'].unique():
            pred = redict_user_threshold(i, j, train, theta)  # predicted

            for h in range(0, len(test)):
                if test['userId'][h] == i:
                    if test['movieId'][h] == j:
                        rate_item.append(test['rating'][h])

        item.append(Mean_Absolute_Error(rate_item, pred))

    for i in range(0, len(item)):
        sum_ += item[i]

    value = sum_ / len(test)
    MAE_Threshold.append(value)

# **Plot Coverage for Threshold**
# ------------------Calcute Coverage---------------------------
cov_Threshold = []

for k in range(1, 26):
    rate_item = []
    item = []
    sum_ = 0

    for i in train['userId'].unique():
        for j in train['movieId'].unique():
            pred = predict_user_threshold(i, j, train, theta)  # predicted
            item.append(prediction_coverage(pred))

    for i in range(0, len(item)):
        sum_ += item[i]

    value = sum_ / len(test)
    cov_Threshold.append(value)

# --------------Plot MAE---------------------
df = pd.DataFrame(MAE_knn)
df.to_csv('/home/401156007/Master/MAE_Threshold.csv')
df_01 = pd.read_csv('/home/401156007/Master/MAE_Threshold.csv')

# Select the second column of the DataFrame
col = df_01.iloc[:, 1]

col.plot(kind='bar')
plt.title('MAE Error(Threshold)')

# Set the labels for the x-axis and y-axis
plt.xlabel('Index')
plt.ylabel('Value')

plt.savefig('/home/401156007/Master/MAE_Threshold.jpg')

plt.show()

# --------------Plot Coverage---------------------
df = pd.DataFrame(cov_knn)
df.to_csv('/home/401156007/Master/cov_Threshold.csv')
df_01 = pd.read_csv('/home/401156007/Master/cov_Threshold.csv')

# Select the second column of the DataFrame
col = df_01.iloc[:, 1]

col.plot(kind='bar')

plt.title('Covearge Error(Threshold)')
plt.xlabel('Index')
plt.ylabel('Value')

plt.savefig('/home/401156007/Master/cov_Threshold.jpg')

plt.show()