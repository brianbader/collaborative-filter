# functions to help with analysis
#################################
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error


# normalize and denormalize a dataframe or vector
def normalize(x, mean_x, std_x):
    z = x.subtract(mean_x, axis='index').div(std_x, axis='index')
    y = x.copy()
    y[y!=0] = z[y!=0]
    return y


def denormalize(x, mean_x, std_x):
    z = x.mul(std_x, axis='index').add(mean_x, axis='index')
    y = x.copy()
    y[y!=0] = z[y!=0]
    return y


def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(actual, pred))


'''
# standardize ratings within user
# need to provide original rating matrix (df), full user list, and full movie list
# that are present in either the train or test set
# returns pandas dfs of ratings matrix, and mean/sd rating within each user
def data_pipeline(rating_raw, userlist, movielist):
    mean_rating_raw = rating_raw.groupby(['userId'])['rating'].mean()
    std_rating_raw = rating_raw.groupby(['userId'])['rating'].std()
    # create user-movie matrix
    # need to make sure all users and movies are present in the pivoted matrix
    rating_raw_tmp = pd.DataFrame(list(itertools.product(userlist, movielist)), columns=['userId', 'movieId'])
    rating_raw_tmp['rating'] = 0
    rating_raw = pd.concat([rating_raw, rating_raw_tmp], axis=0)
    rating_raw.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)
    rating_raw = pd.pivot_table(rating_raw[['userId', 'movieId', 'rating']], values='rating', index='userId', columns='movieId', fill_value=0)
    return rating_raw, mean_rating_raw, std_rating_raw
 '''


# predict using user or item-based similarity matrix
def get_prediction_matrix(rating_clean, mean_rating_clean, similarity, sim_type='user'):
    ind_rating = rating_clean.copy()
    ind_rating[rating_clean!=0] = 1
    if sim_type == 'user':
        pred = pd.DataFrame(similarity.dot(rating_clean) / similarity.dot(ind_rating), index=rating_clean.index, columns=rating_clean.columns)
    elif sim_type == 'item':
        pred = pd.DataFrame(rating_clean.dot(similarity) / ind_rating.dot(similarity), index=rating_clean.index, columns=rating_clean.columns)
    # fill NAs with mean user rating
    mean_rating_fill = pd.DataFrame(0, index=rating_clean.index, columns=rating_clean.columns).add(mean_rating_clean, axis='index')
    pred[np.isnan(pred)] = mean_rating_fill[np.isnan(pred)]
    return pred


# recommend (unrated) movie based on given title and user
def recommend_movies(ratings_full_, user_similarity_, movies_, user_ids, top_num):
    # change column names to movie titles instead of ids
    movie_dict = dict(zip(movies_['title'], movies_['movieId']))
    for user in user_ids:
        for movie in movie_titles:

        get_current_top_movies = ratings_full_.iloc[user, movie_dict[movie]]
        get_pred_top_movies    = pred_.iloc[:, 10]
        print([movie_dict[i] for i in get_current_top_movies])
        print([movie_dict[i] for i in get_pred_top_movies])




pred_.iloc[:, 10]
ratings_full_.iloc[:,10].nlargest(20)

