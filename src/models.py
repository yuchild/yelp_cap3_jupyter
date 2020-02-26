import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

from surprise import (SVD
                      , SVDpp
                      , SlopeOne
                      , NMF
                      , NormalPredictor
                      , KNNBaseline
                      , KNNBasic
                      , KNNWithMeans
                      , KNNWithZScore
                      , BaselineOnly
                      , CoClustering)
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

from sklearn.decomposition import NMF


def load_data():
    biz_dt = pd.read_json('data/business.json', lines = True)
    rev_dt = pd.read_json('data/review.json', lines = True)
    ckin_dt = pd.read_json('data/checkin.json', lines = True)
    pho_dt = pd.read_json('data/photo.json', lines = True)
    tip_dt = pd.read_json('data/tip.json', lines = True)
    usr_dt = pd.read_json('data/user.json', lines = True)
    return biz_dt, rev_dt, ckin_dt, pho_dt, tip_dt, usr_dt

def data_summary(a, b, c, d, e, f):
    sum_dict = {'name': ['business'
                         , 'review'
                         , 'checkin'
                         , 'photo'
                         , 'tip'
                         , 'user']
                , 'rows': [a.shape[0]
                           , b.shape[0]
                           , c.shape[0]
                           , d.shape[0]
                           , e.shape[0]
                           , f.shape[0]
                          ]
                , 'colums': [a.shape[1]
                             , b.shape[1]
                             , c.shape[1]
                             , d.shape[1]
                             , e.shape[1]
                             , f.shape[1]
                          ]
               }
    return pd.DataFrame(sum_dict).style.hide_index()

def combine_tables(user_df, rev_df, biz_df):
    user_rev = user_df.merge(rev_df
                             , on = 'user_id'
                             , how = 'inner')
    
    user_rev.rename(columns = {'useful_x': 'useful_user_sent'
                               , 'funny_x': 'funny_user_sent'
                               , 'cool_x': 'cool_user_sent'
                               , 'useful_y': 'useful_rev'
                               , 'funny_y': 'funny_rev'
                               , 'cool_y': 'cool_rev'}
                    , inplace = True)
    
    user_rev_biz = user_rev.merge(biz_df
                                  , on = 'business_id'
                                  , how = 'inner')
    
    user_rev_biz.rename(columns = {'text': 'rev_text'
                                   , 'name_x': 'user_name'
                                   , 'review_count_x': 'user_review_count'
                                   , 'stars_x': 'stars_rev'
                                   , 'name_y': 'biz_name'
                                   , 'stars_y': 'biz_star'
                                   , 'review_count_y': 'biz_review_count'}
                        , inplace = True)
    user_rev_biz = user_rev_biz.loc[user_rev_biz['is_open'] == 1]
    user_rev_biz = user_rev_biz.drop(['address'
                                      , 'state'
                                      , 'postal_code'
                                      , 'latitude'
                                      , 'longitude'
                                      , 'is_open'
                                      , 'hours'
                                     ]
                                    , axis = 1)
    return user_rev_biz

def collab_mat(city, df):
    city_df = df[df['city']==city]
    user_biz_collab_mat = city_df.pivot_table(index = 'user_id'
                                              , columns = 'business_id'
                                              , values = 'average_stars')
    user_biz_collab_mat.fillna(0
                               , inplace=True)    
    return user_biz_collab_mat


def svd_mat(df, k=10):
    u, s, vt = svds(df, k)
    sigma = np.diag(s)
    user_biz_predictions = u @ sigma @ vt + df.mean(axis=0).to_numpy()
    return s, user_biz_predictions

def top_biz_pred(name_id, df_all, df_mat, df_pred, n=5):
    
    name = df_all['user_name'].loc[df_all.user_id == name_id].unique()[0]
    
    user_id_list = np.array(df_mat.index)
    user_idx = np.argwhere(user_id_list == '---PLwSf5gKdIoVnyRHgBA')[0][0]
    
    biz = df_pred[user_idx].argsort()[-n:][::-1]
    
    biz_id_list = np.array(df_mat.columns)
    biz_list = biz_id_list[biz]
    
    top_biz_list = []
    
    for bz in biz_list:
        top_biz_list.append(df_all['biz_name'].loc[df_all.business_id == bz].unique()[0])
    return name, top_biz_list


def surprise_validate(df):
    data = df[['user_id'
               , 'business_id'
               , 'average_stars']].loc[df.city == 'Scottsdale']
    
    reader = Reader()
    data = Dataset.load_from_df(data, reader)
    
    trainset, testset = train_test_split(data, test_size = 0.25)
    
    algo = SVD()
    algo.fit(trainset)
    
    predictions = algo.test(testset)
    
    acc = accuracy.rmse(predictions)
    
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [SVD()
                      , NMF()
                      , NormalPredictor()
                      , CoClustering()
                      , BaselineOnly()
                     ]:
    # Perform cross validation
        results = cross_validate(algorithm
                                 , data
                                 , measures=['RMSE']
                                 , cv=5
                                 , verbose=False)

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
    
    df_sum = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
        
    return data, acc, df_sum


def NMF_Mat(df):
    model_nmf = NMF(n_components = 40
               , init = 'random'
               , random_state = 0)
    m = model_nmf.fit_transform(df)
    h = model_nmf.components_
    nmf_mat = m @ h
    
    return nmf_mat



















    

    
    


            
