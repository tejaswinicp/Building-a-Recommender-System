#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 23:37:19 2019

@author: tejaswinicp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics.pairwise import pairwise_distances 

#Reading users data
u_header = ['user_id', 'age', 'sex', 'occupation', 'zip_code'] 
users = pd.read_csv('ml-100k/u.user', sep = '|', names = u_header, encoding = 'latin-1')

#reading ratings data
r_header = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_header,encoding='latin-1')

#Reading items file
i_header = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_header,
encoding='latin-1')

ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_header, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_header, encoding='latin-1')
print(ratings_train.shape, ratings_test.shape)

#items['movie title'] = items['movie title'].fillna("").astype('str')
#title_corpus = ' '.join(items['movie title'])
#title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
#plt.figure(figsize=(16,8))
#plt.imshow(title_wordcloud)
#plt.axis('off')
#plt.show()

#Number of unique users and movies
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
    
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')
print(user_prediction)
