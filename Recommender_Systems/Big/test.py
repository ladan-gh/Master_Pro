# import pandas as pd
# import numpy as np
# import math
#
# #----------------------------------
# def pearson_sim(rating_, user_id_01, user_id_02):
#     avg = []
#     x1 = []
#     rate_01 = []
#     for j in range(0, len(rating_)):
#         if rating_['userId'][j] == user_id_01:
#             x1.append(rating_['movieId'][j])
#             rate_01.append(rating_['rating'][j])
#     x2 = []
#     rate_02 = []
#     for j in range(0, len(rating_)):
#         if rating_['userId'][j] == user_id_02:
#             x2.append(rating_['movieId'][j])
#             rate_02.append(rating_['rating'][j])
#     set1 = set(x1)
#     set2 = set(x2)
#     similar_item = list(set1.intersection(set2))
#     m1 = []
#     m2 = []
#     for i in range(0, len(rating_)):
#         if rating_['userId'][i] == user_id_01:
#             m1.append(rating_['rating'][i])
#         elif rating_['userId'][i] == user_id_02:
#             m2.append(rating_['rating'][i])
#
#     mean_01 = np.mean(m1)  # average rating of user1 to all item
#     mean_02 = np.mean(m2)  # average rating of user2 to all item
#
#     similar_item_rate_01 = []
#     similar_item_rate_02 = []
#
#     for i in range(0, len(rating_)):
#         for j in range(0, len(similar_item)):
#
#             if user_id_01 == rating_['userId'][i]:
#
#                 if rating_['movieId'][i] == similar_item[j]:
#
#                     similar_item_rate_01.append(rating_['rating'][i])
#
#             elif user_id_02 == rating_['userId'][i]:
#
#                 if rating_['movieId'][i] == similar_item[j]:
#
#                     similar_item_rate_02.append(rating_['rating'][i])
#
#     sum_ = sum((similar_item_rate_01 - np.mean(rate_01)) * (similar_item_rate_02 - np.mean(rate_02)))  # The numerator
#
#     try:
#         pearson_ = sum_ / (math.sqrt( sum( (similar_item_rate_01 - np.mean(rate_01))**2 ) ) * math.sqrt( sum( (similar_item_rate_02 - np.mean(rate_02))**2 ) ))
#     except ZeroDivisionError:
#         pearson_ = 0
#
#     return pearson_

#********************************************************************