## COLLABORATIVE FILTERING IMPLEMENTATIONS FROM SCRATCH

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
os.chdir('C:/Users/bbade01/Desktop/movielens')
# read in source functions
exec(open('./code/source_fns.py').read())


# read in data
ratings = pd.read_csv('./input/ratings.csv')
movies = pd.read_csv('./input/movies.csv')
# all users have rated at least 20 movies, so we are good!
ratings.groupby(['userId'])['userId'].count().min()
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



recommend_movies(ratings_pivot, pred_user, movies, [30], 20)








# baseline RMSE
get_rmse(train_mean_rating + np.zeros(shape=train.shape), test)
# user, then item -- not a huge improvement!
get_rmse(test_pred_user, test)
get_rmse(test_pred_item, test)




##############################################
# implement alternating weighted least squares
#
# fix parameters first
from scipy import sparse
R = pd.pivot_table(ratings[['userId', 'movieId', 'rating']], values='rating', index='userId', columns='movieId', fill_value=0).values
lambda_ = 0.1
n_factors = 40
n, m = R.shape
n_iter = 20
X = 5 * np.random.rand(n, n_factors) # user factor matrix
Y = 5 * np.random.rand(n_factors, m) # movie factor matrix
# create weights matrix of indicators
W = R.copy()
W[W > 0] = 1


def return_error(R, X, Y, W):
    return np.sum((W * (R - np.dot(X, Y)))**2)


def get_factor_matrices(R, X, Y, W, lambda_, n_factors, n_iter):
    print('Precomputing some quantities...')
    Wu_R = np.empty([n, m])
    Wi_R = np.empty([m, n])
    Wu_diag = []
    Wi_diag = []
    for u, Wu in enumerate(W):
        Wu_diag.append(sparse.csr_matrix(np.diag(Wu)))
        Wu_R[u] = np.dot(Wu_diag[u].todense(), R[u].T)

    for i, Wi in enumerate(W.T):
        Wi_diag.append(sparse.csr_matrix(np.diag(Wi)))
        Wi_R[i] = np.dot(Wi_diag[i].todense(), R[:, i])

    lambda_factors = lambda_ * np.eye(n_factors)
    weighted_errors = []

    print('Begin iterating...')
    for ii in range(n_iter):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu_diag[u].todense(), Y.T)) + lambda_factors,
                                   np.dot(Y, Wu_R[u])).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(Wi_diag[i].todense(), X)) + lambda_factors,
                                     np.dot(X.T, Wi_R[i]))
        weighted_errors.append(return_error(R, X, Y, W))
        print('Iteration {} is completed'.format(ii))


out = get_factor_matrices(R, X, Y, W, lambda_, n_factors, n_iter)





weighted_Q_hat = np.dot(X,Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))


def print_recommendations(W=W, Q=Q, Q_hat=Q_hat, movie_titles=movie_titles):
    #Q_hat -= np.min(Q_hat)
    #Q_hat[Q_hat < 1] *= 5
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')

