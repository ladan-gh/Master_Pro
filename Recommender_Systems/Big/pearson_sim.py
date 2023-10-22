import pandas as pd
import numpy as np
import math

#-------------------------------------------
def pearson_sim(train, user_01, user_02):
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
    denominator = np.sqrt(np.sum((similar_item_rate_01 - mean_01)**2)) * np.sqrt(np.sum((similar_item_rate_02 - mean_02)**2))

    if denominator == 0:
        return 0.0

    pearson_ = numerator / denominator
    return pearson_
#-------------------------------------------
def save_sim(x, train):
    r = np.zeros((x, x))

    for i in range(0, len(r)):
        for j in range(0, len(r)):
            if train['userId'].unique()[i] == train['userId'].unique()[j]:
                r[i][j] = 1
            else:
                r[i][j] = pearson_sim(train, train['userId'].unique()[i], train['userId'].unique()[j])

    df = pd.DataFrame(r)
    df.to_csv('/home/401156007/Master/pearson_matrix.csv')

#----------------------------------------------
t1 = pd.read_csv('/home/401156007/Master/0.csv')
t2 = pd.read_csv('/home/401156007/Master/1.csv')
t3 = pd.read_csv('/home/401156007/Master/2.csv')
t4 = pd.read_csv('/home/401156007/Master/3.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('/home/401156007/Master/5.csv')

#------------Call----------------------------
save_sim(len(train['userId'].unique()), train)