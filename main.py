import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

from module import get_unrated_item
from module import get_pred_unrated_item
from module import get_top_highest_unrated


#set movie and rating dataset location
movie_path = "dataset/movies.csv"
rating_path = "dataset/ratings.csv"


#import movie database
movie_data = pd.read_csv(movie_path,
                         index_col = "movieId",
                         delimiter=',')


#import rating database and wrangling
rating_data = pd.read_csv(rating_path,
                         delimiter=',')

rating_data.drop('timestamp', axis = 1, inplace = True)

rating_data.columns=['user_id','item_id','rating']

reader = Reader(rating_scale=(1,5))
utility_data = Dataset.load_from_df(rating_data, reader)


#split dataset
trainset, testset = train_test_split(utility_data, test_size = 0.2, random_state = 123)

#Model training

#Parameters
best_params_svd = {'n_epochs': 20,
                    'n_factors': 50,
                    'lr_all': 0.01,
                    'reg_all': 0.05}

#Class building
SVD_model = SVD(**best_params_svd)

#Training
SVD_model.fit(trainset)

#Put user id that will be predicted
user_id_predict = 111

recommended_movies = get_top_highest_unrated(estimator=SVD_model,
                        k=10,
                        userid= user_id_predict,
                        rating_data=utility_data.df,
                        metadata=movie_data)

print(recommended_movies)
    