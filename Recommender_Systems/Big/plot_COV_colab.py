import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#------------------------------------------
def Average_Total(train):
    average = train['rating'].mean()
    return average

#-------------------------------------------
def prediction_coverage(predicted):
    if predicted != 0:
        coverage = 1
    else:
        coverage = 0

    return coverage

#------------------------------------------
def Per_User_Average(df, user_id):
    user_ratings = df[df['userId'] == user_id]['rating']
    return user_ratings.mean()

#------------------------------------------
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
# **Main Program**
t1 = pd.read_csv('E:/dataAI/0.csv')
t2 = pd.read_csv('E:/dataAI/1.csv')
t3 = pd.read_csv('E:/dataAI/2.csv')
t4 = pd.read_csv('E:/dataAI/5.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('E:/dataAI/3.csv')

# ------------------Calcute Coverage----------------------per user
# **Calcute Coverage error**
cov = []  # y
user = []
sum_ = 0

for i in train['userId'].unique():
    avg_user = Per_User_Average(train, i)  # predicted
    user.append(prediction_coverage(avg_user))

for i in range(0, len(user)):
    sum_ += user[i]

value = sum_ / len(test)

print('per user is: ', value)
cov.append(value)

# ------------------Calcute Coverage----------------------per item
sum_ = 0
item = []

for i in train['movieId'].unique():
    avg_item = Per_Item_Average(train, i)  # predicted
    item.append(prediction_coverage(avg_item))

for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)

print('per item is: ', value)
cov.append(value)

# ------------------Calcute Coverage---------------------------user based
# rate_item = []
# item = []
# sum_ = 0
#
# for i in train['userId'].unique():
#     for j in train['movieId'].unique():
#         pred = predict_user_rating(i, j, train)  # predicted
#         item.append(prediction_coverage(pred))
#
# for i in range(0, len(item)):
#     sum_ += item[i]
#
# value = sum_ / len(test)
# cov.append(value)

#------------------------------------------------average total
avg_total = Average_Total(train)

pred = prediction_coverage(avg_total)


# Compute the final value
value = pred / len(test)

print('global is: ')
print('{:.15f}'.format(value))

cov.append(value)

# --------------Plot Coverage---------------------
df = pd.DataFrame(cov)
df.to_csv('E:/dataAI/cov_05.csv')
df = pd.read_csv('E:/dataAI/cov_05.csv')

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
ax.set_title('Coverage Error')
ax.set_xlabel('Algorithms')
ax.set_ylabel('Values')

plt.savefig('E:/dataAI/Coverage_05.jpg')
plt.show()