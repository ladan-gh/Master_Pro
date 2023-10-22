import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import math
import threading

#------------------------------------------
def Average_Total(train):
    average = train['rating'].mean()
    return average

#--------------------------------
def prediction_coverage(predicted):
    if predicted != 0:
        coverage = 1
    else:
        coverage = 0

    return coverage

#----------------------------------
def Per_User_Average(df, user_id):
    user_ratings = df[df['userId'] == user_id]['rating']
    return user_ratings.mean()

#---------------------------------
def Per_Item_Average(train, item):
    item_rows = train[train['movieId'] == item]

    if item_rows.empty:
        return 0

    ratings = item_rows['rating'].values

    avg_rating = np.mean(ratings)

    return avg_rating

#------------------------------------------------------------
def predict_user_rating(user_id, item_id, rating_):
    threshold = 0
    # Load the similarity matrix
    user_similarity = pd.read_csv('/home/401156007/Master/pearson_matrix.csv')

    # Get the indices of the users who are similar to the given user
    neighbor_indices = []
    for index, row in user_similarity.iterrows():
        if row[user_id] >= threshold and index != user_id:
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
def Mean_Absolute_Error(true, predicted):
    sum_ = 0

    if type(true) is list:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true[i])

    else:
        for i in range(0, len(true)):
            sum_ += abs(predicted - true.iloc[i])

    return sum_

#----------------------------------------------
# **Main Program**
t1 = pd.read_csv('/home/401156007/Master/0.csv')
t2 = pd.read_csv('/home/401156007/Master/1.csv')
t3 = pd.read_csv('/home/401156007/Master/2.csv')
t4 = pd.read_csv('/home/401156007/Master/3.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('E:/dataAI/5.csv')

# **Calcute MAE error**
#------------------Calcute MAE---------------------------per user
MAE = [] #y1
rate_user = []
sum_ = 0
user = []

user_ids = train['userId'].unique()
avg_ratings = {user_id: Per_User_Average(train, user_id) for user_id in user_ids}


for user_id in user_ids:
    ratings = test.loc[test['userId'] == user_id, ['movieId', 'rating']]
    ratings = ratings[ratings['movieId'].isin(train['movieId'].unique())]
    ratings = ratings.set_index('movieId')
    user_ratings = ratings['rating']
    avg_user = avg_ratings[user_id]
    user.append(Mean_Absolute_Error(user_ratings, avg_user))


for i in range(0, len(user)):
    sum_ += user[i]

value = sum_ / len(test)
print('per user is: ', value)

MAE.append(value)

#------------------Calcute MAE---------------------------per item
item = []
sum_ = 0

item_ids = train['movieId'].unique()

avg_ratings = {item_id: Per_Item_Average(train, item_id) for item_id in item_ids}


for item_id in item_ids:
    ratings = test.loc[test['movieId'] == item_id, ['userId', 'rating']]
    ratings = ratings[ratings['userId'].isin(train['userId'].unique())]
    ratings = ratings.set_index('userId')
    item_ratings = ratings['rating']
    item.append(Mean_Absolute_Error(item_ratings, avg_ratings[item_id]))


for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
print('per item is: ', value)

MAE.append(value)

# ------------------Calcute MAE---------------------------user based
item = []
sum_ = 0
pred_user = []

for i in train['userId'].unique():
  for j in train['movieId'].unique():
      pred = predict_user_rating(i, j, train)#predicted
      pred_user.append(predict_user_rating(i, j, train))

      # Filter the test DataFrame based on the user ID and item ID
      filtered_df = test.query('userId == @i and movieId == @j')

      # Extract the 'rating' column from the filtered DataFrame
      rate_item = filtered_df['rating'].tolist()
      item.append(Mean_Absolute_Error(rate_item, pred))


for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
MAE.append(value)

#----------------------------------------------------------average total
avg_total = Average_Total(train)

# Compute the average rating and the absolute differences
absolute_diff = np.abs(test['rating'] - avg_total)

# Compute the final value
value = absolute_diff.sum() / len(test)

MAE.append(value)

#---------------------------------------------------
df = pd.DataFrame(MAE)
df.to_csv('/home/401156007/Master/MAE_01.csv')
df = pd.read_csv('/home/401156007/Master/MAE_01.csv')

df = df.rename(columns={'Unnamed: 0': 'Algorithms', '0':'values'})

# Set the index of the second row to 'NewIndex'
df = df.set_index(pd.Index(['per user', 'per item', 'total']))


# Get the values for each bar
x_labels = df.index # Assuming the labels are in a column named 'Label'
values = df['values'] # Assuming the values are in a column named 'Value'

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(x_labels, values)

# Set the y-axis scale to logarithmic to show the big difference in values
ax.set_yscale('log')

# Set the plot title and axis labels
ax.set_title('MAE Error')
ax.set_xlabel('Algorithms')
ax.set_ylabel('Values')

plt.savefig('/home/401156007/Master/MAE_01.jpg')

# Show the plot
plt.show()

#----------------------------------------
# d = pd.DataFrame(pred_user)
# d.to_csv('/home/401156007/Master/pored_user_rating.csv')