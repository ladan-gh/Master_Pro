import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import math

#----------------------------------
class Item_Based_CFSystem():

    def plot_rating_Hist(self, rating_):
        rating_['rating'].plot(kind='hist', density=True)
        plt.title('MovieLens Rating')
        plt.xlabel('Rating')
        plt.ylabel('Percent of total rating')
        plt.show()

    # ---------------------------------------------------
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
    def pearson_sim(self, train, item_01, item_02):
        item_01_ratings = train.loc[train['movieId'] == item_01, ['userId', 'rating']]
        item_02_ratings = train.loc[train['movieId'] == item_02, ['userId', 'rating']]
        common_items = pd.merge(item_01_ratings, item_02_ratings, on=['userId'], suffixes=('_01', '_02'))

        if common_items.empty:
            return 0.0

        similar_user_rate_01 = common_items['rating_01'].tolist()
        similar_user_rate_02 = common_items['rating_02'].tolist()

        similar_user_rate_01 = np.array(similar_user_rate_01)
        similar_user_rate_02 = np.array(similar_user_rate_02)

        numerator = np.sum(similar_user_rate_01 * similar_user_rate_02)

        denominator = np.sqrt(np.sum(similar_user_rate_01 ** 2)) * np.sqrt(np.sum(similar_user_rate_02 ** 2))

        if denominator == 0:
            return 0.0

        pearson_ = numerator / denominator

        return pearson_

    #----------------------------------------------------
    def predict_item_rating(self, user_id, item_id, rating_):
        threshold = 0
        # Load the similarity matrix
        item_similarity = pd.read_csv('E:/dataAI/pearson_matrix_item.csv', index_col=0)

        # Get the indices of the items that are similar to the given item
        neighbor_indices = []
        for index, row in item_similarity.iterrows():
            if row[item_id] >= threshold and index != item_id:
                neighbor_indices.append(index)  # row

        # Get the ratings of the neighbors by the same user
        neighbor_ratings = rating_.loc[(rating_['movieId'].isin(neighbor_indices)) & (rating_['userId'] == user_id), 'rating'].tolist()
        neighbor_indices = rating_.loc[(rating_['movieId'].isin(neighbor_indices)) & (rating_['userId'] == user_id), 'movieId'].tolist()

        sim = []
        for i in neighbor_indices:
            for j in item_similarity.index:
                if i == int(j):
                    sim.append(item_similarity.loc[item_id, i])

        sum_ = 0
        # Calculate the predicted rating
        if len(neighbor_ratings) > 0:
            for i in range(len(sim)):
                sum_ += (neighbor_ratings[i] * sim[i])

            predicted_rating = sum_ / sum(sim)
        else:
            predicted_rating = 0

        return predicted_rating

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

    # -----------------------------------------------
    def Per_User_Average(self, df, user_id):
        user_ratings = df[df['userId'] == user_id]['rating']
        return user_ratings.mean()

    #-----------------------------------------------
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

    #-----------------------------------------------
    def save_sim(self, x, train):
        r = np.zeros((x, x))

        for i in range(0, len(r)):
            for j in range(0, len(r)):
                if train['userId'].unique()[i] == train['userId'].unique()[j]:
                    r[i][j] = 1
                else:
                    r[i][j] = Item_Based_CFSystem.pearson_sim(train, train['userId'].unique()[i], train['userId'].unique()[j])

        df = pd.DataFrame(r)
        df.to_csv('E:/dataAI/pearson_matrix_item.csv')

#---------------read data and create matrix for rating data-----------
ratings_df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Special_Topics_AI/Big/ratings.csv')
ratings_df = ratings_df.sort_values('timestamp', ascending=True)

item_based = Item_Based_CFSystem()
#--------------------------------------------------
item_based.K_fold(ratings_df)
t1 = pd.read_csv('E:/dataAI/0.csv')
t2 = pd.read_csv('E:/dataAI/1.csv')
t3 = pd.read_csv('E:/dataAI/2.csv')
t4 = pd.read_csv('E:/dataAI/3.csv')
t5 = pd.read_csv('E:/dataAI/5.csv')

train = pd.concat([t1, t2, t3, t4, t5])
item_based.save_sim(len(train['movieId'].unique()), train) #save similarity matrix

#--------------------------------------------------
train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('E:/dataAI/5.csv')

#----------------------------------------------------
Item_Base = Item_Based_CFSystem()

#------------------------------------------------------
Item_Base.plot_rating_Hist(ratings_df)

# ------------------Calcute Coverage----------------------per user
# **Calcute Coverage error**
cov = []  # y
user = []
sum_ = 0

for i in train['userId'].unique():
    avg_user = Item_Base.UserPer_User_Average(train, i)  # predicted
    user.append(Item_Base.prediction_coverage(avg_user))

for i in range(0, len(user)):
    sum_ += user[i]

value = sum_ / len(test)
cov.append(value)

# ------------------Calcute Coverage----------------------per item
sum_ = 0
item = []

for i in train['movieId'].unique():
    avg_item = Item_Base.Per_Item_Average(train, i)  # predicted
    user.append(Item_Base.prediction_coverage(avg_item))

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
        pred = Item_Base.predict_user_rating(i, j, train)  # predicted
        item.append(Item_Base.prediction_coverage(pred))

for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
cov.append(value)

#------------------------------------------------average total
avg_total = Item_Base.Average_Total(train)
avg = Item_Base.prediction_coverage(avg_total)

value = avg / len(test)
cov.append(value)

# --------------Plot Coverage---------------------
df = pd.DataFrame(cov)
df.to_csv('E:/dataAI/cov_item.csv')
df = pd.read_csv('E:/dataAI/cov_item.csv')

df = df.rename(columns={'Unnamed: 0': 'Algorithms', '0':'values'})

# Set the index of the second row to 'NewIndex'
df = df.set_index(pd.Index(['per user', 'per item', 'total', 'user based', 'item based']))


# Get the values for each bar
x_labels = df.index
values = df['values']

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(x_labels, values)

ax.set_yscale('log')

# Set the plot title and axis labels
ax.set_title('Coverage Error')
ax.set_xlabel('Algorithms')
ax.set_ylabel('Values')

plt.savefig('E:/dataAI/Coverage_item.jpg')
plt.show()


# **Calcute MAE error**
# ------------------Calcute MAE---------------------------per user
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

# ------------------Calcute MAE---------------------------user based
item = []
sum_ = 0
pred_user = []

for i in train['userId'].unique():
  for j in train['movieId'].unique():
      pred = Item_Base.predict_user_rating(i, j, train)#predicted
      pred_user.append(Item_Base.predict_user_rating(i, j, train))

      # Filter the test DataFrame based on the user ID and item ID
      filtered_df = test.query('userId == @i and movieId == @j')

      # Extract the 'rating' column from the filtered DataFrame
      rate_item = filtered_df['rating'].tolist()
      item.append(Item_Base.Mean_Absolute_Error(rate_item, pred))


for i in range(0, len(item)):
    sum_ += item[i]

value = sum_ / len(test)
MAE.append(value)

#------------------plot MAE---------------------------------
df = pd.DataFrame(MAE)
df.to_csv('E:/dataAI/MAE_item.csv')
df = pd.read_csv('E:/dataAI/MAE_item.csv')

df = df.rename(columns={'Unnamed: 0': 'Algorithms', '0':'values'})

# Set the index of the second row to 'NewIndex'
df = df.set_index(pd.Index(['per user', 'per item', 'total', 'user based', 'item based']))

# Get the values for each bar
x_labels = df.index
values = df['values']

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(x_labels, values)

ax.set_yscale('log')

# Set the plot title and axis labels
ax.set_title('MAE Error')
ax.set_xlabel('Algorithms')
ax.set_ylabel('Values')

plt.savefig('E:/dataAI/MAE_item.jpg')

plt.show()