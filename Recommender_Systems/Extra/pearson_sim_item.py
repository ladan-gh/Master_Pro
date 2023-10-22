import pandas as pd
import numpy as np
import math

#-------------------------------------------
def pearson_sim(train, item_01, item_02):

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
#-------------------------------------------
def save_sim(x, train):
    r = np.zeros((x, x))

    for i in range(0, len(r)):
        for j in range(0, len(r)):
            if train['movieId'].unique()[i] == train['movieId'].unique()[j]:
                r[i][j] = 1
            else:
                r[i][j] = pearson_sim(train, train['movieId'].unique()[i], train['movieId'].unique()[j])

    df = pd.DataFrame(r)
    df.to_csv('E:/dataAI/pearson_matrix_item.csv')

#----------------------------------------------
t1 = pd.read_csv('E:/dataAI/0.csv')
t2 = pd.read_csv('E:/dataAI/1.csv')
t3 = pd.read_csv('E:/dataAI/2.csv')
t4 = pd.read_csv('E:/dataAI/3.csv')

train = pd.concat([t1, t2, t3, t4])
test = pd.read_csv('E:/dataAI/5.csv')

#------------Call----------------------------
save_sim(len(train['movieId'].unique()), train)