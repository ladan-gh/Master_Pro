import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import math

#----------------------------------
class User_Base_CFSystem():

    def plot_rating_Hist(self, rating_):
        rating_['rating'].plot(kind='hist', density=True)
        plt.title('MovieLens Rating')
        plt.xlabel('Rating')
        plt.ylabel('Percent of total rating')
        plt.show()

    #----------------------------------
    def K_fold(self, rating_):
        list_user = ratings_df['userId'].unique()
        CsvPath = 'E:/dataAI/'
        users = ratings_df['userId'].unique()
        k = 0
        index_ = []
        rate = []

        for j in range(0, 5):
            data = pd.DataFrame()
            if j == 0:
                for i in range(0, len(users)):
                    rate = ratings_df[ratings_df['userId'] == users[i]]
                    len_ = len(rate) // 5
                    index_.append(len_ + k)
                    data = pd.concat([data ,rate[k:len_ + k]])

                csvFileName = CsvPath + str(j) + '.csv'
                df = pd.DataFrame(data)
                df.to_csv(csvFileName)

            else:
                for i in range(0, len(users)):
                    rate = ratings_df[ratings_df['userId'] == users[i]]
                    len_ = len(rate) // 5
                    data = pd.concat([data, rate[index_[i]:len_ + index_[i]]])
                    index_[i] = len_ + index_[i]

                csvFileName = CsvPath + str(j) + '.csv'
                df = pd.DataFrame(data)
                df.to_csv(csvFileName)


        remain = pd.DataFrame()
        for i in range(0, len(users)):
            rate = ratings_df[ratings_df['userId'] == users[i]]
            r = len(rate) % 5
            remain = pd.concat([remain,rate.iloc[-r:, :]])

        df_5 = pd.read_csv('E:/dataAI/4.csv')
        remain= pd.concat([remain,df_5])
        remain.to_csv('E:/dataAI/5.csv')

    #---------------------------------------------------
    # custom function to create pearson correlation method from scratch
    def pearson_sim(self, train, user_01, user_02):
        user_01_ratings = train.loc[train['userId'] == user_01, ['movieId', 'rating']]
        user_02_ratings = train.loc[train['userId'] == user_02, ['movieId', 'rating']]
        common_movies = pd.merge(user_01_ratings, user_02_ratings, on=['movieId'], suffixes=('_01', '_02'))

        if common_movies.empty:
            return 0.0

        similar_item_rate_01 = common_movies['rating_01'].tolist()
        similar_item_rate_02 = common_movies['rating_02'].tolist()

        mean_01 = np.mean(similar_item_rate_01)
        mean_02 = np.mean(similar_item_rate_02)

        numerator = np.sum((similar_item_rate_01 - mean_01) * (similar_item_rate_02 - mean_02))
        denominator = np.sqrt(np.sum((similar_item_rate_01 - mean_01) ** 2)) * np.sqrt(
            np.sum((similar_item_rate_02 - mean_02) ** 2))

        if denominator == 0:
            return 0.0

        pearson_ = numerator / denominator
        return pearson_

    #----------------------------------------------------
    # Calculation of offers based on the ratings of similar users
    def predict_user_rating(self, user_id, item_id, rating_):
        threshold = 0
        # Load the similarity matrix
        user_similarity = pd.read_csv('/home/401156007/Master/pearson_matrix.csv')

        # Get the indices of the users who are similar to the given user
        neighbor_indices = []
        for index, row in user_similarity.iterrows():
            if row[user_id] >= threshold and index != user_id:
                neighbor_indices.append(index)  # col

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

    #--------------------------------------------------------
    def predict_user_KNN(self, user_id, item_id, rating_, k):
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

    #--------------------------------------------------------
    def predict_user_threshold(self, user_id, item_id, rating_, theta):

        # Load the similarity matrix
        user_similarity = pd.read_csv('/home/401156007/Master/pearson_matrix.csv')

        # Get the indices of the users who are similar to the given user
        neighbor_indices = []
        for index, row in user_similarity.iterrows():
            if row[user_id] >= theta and index != user_id:
                neighbor_indices.append(index)  # col

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

    #--------------------------------------------------------
    def Per_User_Average(self, df, user_id):
        user_ratings = df[df['userId'] == user_id]['rating']
        return user_ratings.mean()

    # ----------------------------------------------------
    def Per_Item_Average(self, test, train):
        avg = []
        items = train['movieId'].unique()

        for i in range(0, len(items)):
            x = []
            for j in range(0, len(train)):
                if train['movieId'][j] == items[i]:
                    x.append(train['rating'][j])

            avg.append(np.mean(x))

        return avg

    #------------------------------------------
    def Average_Total(self, train):
        average = train['rating'].mean()
        return average

    #-------------------------------------------
    def Mean_Absolute_Error(self, true, predicted):
        sum_ = 0

        if type(true) is list:
            for i in range(0, len(true)):
                sum_ += abs(predicted - true[i])

        else:
            for i in range(0, len(true)):
                sum_ += abs(predicted - true.iloc[i])

        return sum_

    #----------------------------------------------
    def prediction_coverage(self, predicted):
        if predicted != 0:
            coverage = 1
        else:
            coverage = 0

        return coverage

    #-----------------------------------------------
    def save_sim(self, x, train):
        r = np.zeros((x, x))

        for i in range(0, len(r)):
            for j in range(0, len(r)):
                if train['userId'].unique()[i] == train['userId'].unique()[j]:
                    r[i][j] = 1
                else:
                    r[i][j] = pearson_sim(train, train['userId'].unique()[i], train['userId'].unique()[j])

        df = pd.DataFrame(r)
        df.to_csv('E:/dataAI/pearson_matrix.csv')

#---------------read data and create matrix for rating data-----------
ratings_df = pd.read_csv('/home/401156007/Master/ratings.csv')
ratings_df = ratings_df.sort_values('timestamp', ascending=True)

#--------------------------------------------------
train, test = User_Base.K_fold(ratings_df)
User_Base.save_sim(len(train['userId'].unique()), train) #save similarity matrix

#--------------------------------------------------
t1 = pd.read_csv('/home/401156007/Master/0.csv')
t2 = pd.read_csv('/home/401156007/Master/1.csv')
t3 = pd.read_csv('/home/401156007/Master/2.csv')
t4 = pd.read_csv('/home/401156007/Master/3.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('/home/401156007/Master/5.csv')

#----------------------------------------------------
User_Base = User_Base_CFSystem()

#------------------------------------------------------
User_Base.plot_rating_Hist(ratings_df)

# ------------------Calcute Coverage----------------------per user
# **Calcute Coverage error**
cov = []  # y
user = []
sum_ = 0

for i in train['userId'].unique():
    avg_user = User_Base.UserPer_User_Average(train, i)  # predicted
    user.append(User_Base.prediction_coverage(avg_user))

for i in range(0, len(user)):
    sum_ += user[i]

value = sum_ / len(test)
cov.append(value)

# ------------------Calcute Coverage----------------------per item
sum_ = 0
item = []

for i in train['movieId'].unique():
    avg_item = User_Base.Per_Item_Average(train, i)  # predicted
    user.append(User_Base.prediction_coverage(avg_item))

for i in range(0, len(user)):
    sum_ += item[i]

value = sum_ / len(test)
cov.append(value)

# ------------------Calcute Coverage---------------------------user based
rate_item = []
item = []
sum_ = 0

for i in train['userId'].unique():
    for j in train['movieId'].unique():
        pred = User_Base.predict_user_rating(i, j, train)  # predicted
        item.append(User_Base.prediction_coverage(pred))

for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
cov.append(value)

#------------------------------------------------average total
avg_total = User_Base.Average_Total(train)
avg = User_Base.prediction_coverage(avg_total)

value = avg / len(test)
cov.append(value)

# --------------Plot Coverage---------------------
df = pd.DataFrame(cov)
df.to_csv('E:/dataAI/cov.csv')
df = pd.read_csv('E:/dataAI/cov.csv')

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

plt.savefig('E:/dataAI/Coverage.jpg')
plt.show()


# **Calcute MAE error**
#------------------Calcute MAE---------------------------per user
MAE = [] #y1
rate_user = []
sum_ = 0
user = []

user_ids = train['userId'].unique()
avg_ratings = {user_id: User_Base.Per_User_Average(train, user_id) for user_id in user_ids}


for user_id in user_ids:
    ratings = test.loc[test['userId'] == user_id, ['movieId', 'rating']]
    ratings = ratings[ratings['movieId'].isin(train['movieId'].unique())]
    ratings = ratings.set_index('movieId')
    user_ratings = ratings['rating']
    avg_user = avg_ratings[user_id]
    user.append(User_Base.Mean_Absolute_Error(user_ratings, avg_user))


for i in range(0, len(user)):
    sum_ += user[i]

value = sum_ / len(test)
MAE.append(value)

#------------------Calcute MAE---------------------------per item
item = []
sum_ = 0

item_ids = train['movieId'].unique()

avg_ratings = {item_id: User_Base.Per_Item_Average(train, item_id) for item_id in item_ids}


for item_id in item_ids:
    ratings = test.loc[test['movieId'] == item_id, ['userId', 'rating']]
    ratings = ratings[ratings['userId'].isin(train['userId'].unique())]
    ratings = ratings.set_index('userId')
    item_ratings = ratings['rating']
    item.append(User_Base.Mean_Absolute_Error(item_ratings, avg_ratings[item_id]))


for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
MAE.append(value)

# ------------------Calcute MAE---------------------------user based
item = []
sum_ = 0
pred_user = []

for i in train['userId'].unique():
  for j in train['movieId'].unique():
      pred = User_Base.predict_user_rating(i, j, train)#predicted
      pred_user.append(User_Base.predict_user_rating(i, j, train))

      # Filter the test DataFrame based on the user ID and item ID
      filtered_df = test.query('userId == @i and movieId == @j')

      # Extract the 'rating' column from the filtered DataFrame
      rate_item = filtered_df['rating'].tolist()
      item.append(User_Base.Mean_Absolute_Error(rate_item, pred))


for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
MAE.append(value)

#----------------------------------------------------------average total
avg_total = User_Base.Average_Total(train)

sum_ = 0
list_ = []
for i in range(0, len(test['rating'])):
    sum_ += abs(avg_total - test['rating'][i])
    list_.append(sum_)

s = 0
for j in list_:
    s += j

value = s / len(test)
MAE.append(value)

#------------------plot MAE---------------------------------
df = pd.DataFrame(MAE)
df.to_csv('/home/401156007/Master/MAE.csv')
df = pd.read_csv('/home/401156007/Master/MAE.csv')

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

plt.savefig('/home/401156007/Master/MAE.jpg')

# Show the plot
plt.show()


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
#--------------------------------------
k = []
for i in range(1, 26):
    k.append(i)

#plot for Threshold
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