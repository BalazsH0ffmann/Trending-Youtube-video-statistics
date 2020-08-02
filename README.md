# Trending-Youtube-video-statistics
Clustering with DBSCAN method

import mglearn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics


data=pd.read_csv('C:/Users/User/Downloads/CAvideos.csv')
data.head()

data.drop(['video_id'], axis=1, inplace=True)
data.drop(['trending_date'], axis=1, inplace=True)
data.drop(['category_id'], axis=1, inplace=True)
data.drop(['publish_time'], axis=1, inplace=True)
data.drop(['tags'], axis=1, inplace=True)
data.drop(['title'], axis=1, inplace=True)
data.drop(['channel_title'], axis=1, inplace=True)
data.drop(['dislikes'], axis=1, inplace=True)
data.drop(['comment_count'], axis=1, inplace=True)
data.drop(['thumbnail_link'], axis=1, inplace=True)
data.drop(['comments_disabled'], axis=1, inplace=True)
data.drop(['ratings_disabled'], axis=1, inplace=True)
data.drop(['video_error_or_removed'], axis=1, inplace=True)
data.drop(['description'], axis=1, inplace=True)

data.head()

data=data.sample(frac=1)
data.head()

#DBSCAN Custering

from numpy import unique
from numpy import where

data_X=data.iloc[:,[0,1]].values

model=DBSCAN(eps=0.7, min_samples=70)
yhat=model.fit_predict(data_X)
clusters=unique(yhat)
for cluster in clusters:
    row_ix=where(yhat==cluster)
    plt.scatter(data_X[row_ix, 0], data_X[row_ix, 1])

plt.show()


