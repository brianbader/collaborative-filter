# COLLABORATIVE FILTERING IMPLEMENTATION FROM SCRATCH
#
# This code implements memory-based collaborative filtering using cosine similarities
# between users (user-user) and items (item-item). Generally speaking, the input
# is a ratings matrix (users x items), with zeros for unrated items. For demonstration,
# I use the MovieLens 100K dataset as input. Functionality includes generating predictions
# and recommendations.
#
# Inputs and directory structure:
# -input/ratings.csv
# -input/movies.csv
# -code/simple_cf_model.py
# -code/source_fns.py
#
#######################################################################
# imports
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
# set this to the top directory
os.chdir('C:/Users/bbade01/Desktop/movielens')
# read in source functions
exec(open('./code/source_fns.py').read())


# read in data
ratings = pd.read_csv('./input/ratings.csv')
movies = pd.read_csv('./input/movies.csv')
print('All movies have at least {} ratings, so we are good!'.format(ratings.groupby(['userId'])['userId'].count().min()))
# remove movies with small number of ratings (say less than 20)
# and those with small deviaton of ratings
count_movie_rating = ratings.groupby(['movieId'])['rating'].transform(lambda x: len(x))
std_movie_rating = ratings.groupby(['movieId'])['rating'].transform(lambda x: np.std(x))
ratings = ratings[(std_movie_rating > 1) & (count_movie_rating > 20)]
ratings.drop('timestamp', axis=1, inplace=True)
# create full pivot matrix of users and movies
ratings_pivot = pd.pivot_table(ratings[['userId', 'movieId', 'rating']], values='rating', index='userId', columns='movieId', fill_value=0)
mean_user_rating = ratings.groupby(['userId'])['rating'].mean()
std_user_rating = ratings.groupby(['userId'])['rating'].std()


# get user and item similarity matrices (note we probably don't need to normalize item ratings)
user_similarity = 1 - pairwise_distances(ratings_pivot.subtract(mean_user_rating, axis='index'), metric='cosine')
item_similarity = 1 - pairwise_distances(ratings_pivot.T, metric='cosine')


# get predictions (user-user, item-item)
# must use mean and std ratings from train data to not introduce bias
pred_user = get_prediction_matrix(ratings_pivot, mean_user_rating, user_similarity, 'user')
pred_item = get_prediction_matrix(ratings_pivot, mean_user_rating, item_similarity, 'item')


'''
# baseline RMSE
get_rmse(pd.DataFrame(0, columns=ratings_pivot.columns, index=ratings_pivot.index).add(mean_user_rating, axis='index'), ratings_pivot)
# user, then item -- not a huge improvement!
get_rmse(pred_user, ratings_pivot)
get_rmse(pred_item, ratings_pivot)
'''


# get some reccomendations
recommend_movies(ratings_pivot_=ratings_pivot, pred_=pred_user, movies_=movies, user_ids=[88], top_num=5, top_set=100)

