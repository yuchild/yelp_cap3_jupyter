import pandas as pd
import numpy as np
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

from surprise import Dataset
from surprise import Reader

from surprise.model_selection.validation import cross_validate
from surprise import accuracy

from sklearn.decomposition import NMF

from tensorflow.keras.layers import (Input
                                     , Embedding
                                     , Dot
                                     , Flatten
                                    )
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import Callback

import os
import warnings
warnings.filterwarnings('ignore')

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

def svd_model(df):
    """
    Creates svd model for predcitions and cross validation
    Returns: data 
    """
    from surprise.model_selection.split import train_test_split
    data = df[['user_id'
                    , 'business_id'
                    , 'average_stars']].loc[df.city == 'Scottsdale']
    
    reader = Reader()
    
    data = Dataset.load_from_df(data
                               , reader)
    
    trainset, testset = train_test_split(data
                                        , test_size=0.25)
    
    algo = SVD()
    algo.fit(trainset)
    
    predictions = algo.test(testset)
    
    acc = accuracy.rmse(predictions)
    
    svd_cv = cross_validate(SVD()
                           , data
                           , cv = 5)
    
    return data, acc, svd_cv['test_rmse']

def surprise_bench(df):
    """
    Creates benchmark dataframe of SVD, NMF, NormalPredictor, and Baseline with 
    5 Fold cross validation and returns rmse metrics
    """
    from surprise import (SVD
                      , SVDpp
                      , NMF
                      , NormalPredictor
                      , BaselineOnly)

    from surprise import Dataset
    from surprise import Reader

    from surprise.model_selection.validation import cross_validate
    from surprise import accuracy
    
    data = df[['user_id'
                    , 'business_id'
                    , 'average_stars']].loc[df.city == 'Scottsdale']
    
    reader = Reader()
    
    data = Dataset.load_from_df(data
                               , reader)    
    benchmark = []
    
    # Iterate over all algorithms
    for algorithm in [SVD()
                      , SVDpp()
                      , NMF()
                      , NormalPredictor()
                      , BaselineOnly()
                     ]:
    # Perform cross validation
        results = cross_validate(algorithm
                                 , data
                                 , measures=['RMSE', 'MAE']
                                 , cv=5
                                 , verbose=False
                                )

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    return pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')


def NMF_Mat(df):
    model_nmf = NMF(n_components = 40
               , init = 'random'
               , random_state = 0)
    m = model_nmf.fit_transform(df)
    h = model_nmf.components_
    nmf_mat = m @ h
  
    return nmf_mat

def NN_Model(df, n_factors = 10, ep = 5):
    from sklearn.model_selection import train_test_split
    user_rev_biz_scott = df[['user_id'
                                   , 'user_name'
                                   , 'business_id'
                                   , 'biz_name'
                                   , 'average_stars']].loc[df.city == 'Scottsdale']
      
    user_df = user_rev_biz_scott.groupby(['user_id', 'user_name']).size().reset_index(name="Freq")
    user_df.drop('Freq', axis=1, inplace=True)
           
    user_id_list = list(user_df.user_id)
    user_id_dict = {y: x for (x, y) in enumerate(user_id_list)}
    user_rev_biz_scott['user_num'] = user_rev_biz_scott.user_id.map(user_id_dict)
    
    biz_df = user_rev_biz_scott.groupby(['business_id', 'biz_name']).size().reset_index(name="Freq")
    biz_df.drop('Freq', axis=1, inplace=True)
       
    biz_id_list = list(biz_df.business_id)
    biz_id_dict = {y: x for (x, y) in enumerate(biz_id_list)}
    user_rev_biz_scott['biz_num'] = user_rev_biz_scott.business_id.map(biz_id_dict)
    
    X = user_rev_biz_scott[['user_num'
                        , 'user_name'
                        , 'biz_num'
                        , 'biz_name'
                        , 'average_stars'
                       ]]
    y = user_rev_biz_scott.average_stars
    
    X_train, X_test, y_train, y_test = train_test_split(X
                                                        , y
                                                        , test_size=0.25
                                                        , random_state=42)

    n_users = user_rev_biz_scott.user_id.nunique()
    n_biz = user_rev_biz_scott.business_id.nunique()
    
    biz_input = Input(shape=[1]
                     , name = 'Biz_Input')
    biz_embedding = Embedding(n_biz
                             , n_factors
                             , name='Biz_Embed')(biz_input)
    biz_vac = Flatten(name = 'Flatten_Biz')(biz_embedding)

    user_input = Input(shape=[1]
                      , name = 'User_Input')
    user_embedding = Embedding(n_users
                              , n_factors
                              , name = 'User_Embed')(user_input)
    user_vac = Flatten(name = "Flatten_User")(user_embedding)

    prod = Dot(name = 'Dot_Product'
              , axes = 1)([biz_vac, user_vac])
    model = Model([user_input, biz_input]
                 , prod)
    model.compile(optimizer = 'adam'
                 , loss = 'mse'
                 , metrics = ['accuracy'])
    
    class TestCallback(Callback):
        def __init__(self, test_data):
            self.test_data = test_data

        def on_epoch_end(self, epoch, logs={}):
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
    if os.path.exists('biz_model.h5'):
        model = load_model('biz_model.h5')
    else:
        history = model.fit([X_train.user_num
                             , X_train.biz_num]
                             , y_train
                             , epochs=ep
                             , verbose=False
                             , validation_data = ([X_test.user_num
                                                   , X_test.biz_num]
                                                   , y_test)
                             , callbacks = [TestCallback(([X_test.user_num
                                                           , X_test.biz_num]
                                                           , y_test))]
                           )
    model.save('NN_Embed_Model')
    return user_id_dict, biz_id_dict, X, X_test, model, history


def NN_Results_df(mod, xtest, n=10):
    predictions = mod.predict([xtest.user_num.head(n)
                              , xtest.biz_num.head(n)]
                             )
    pred_df = xtest[['user_name', 'biz_name', 'average_stars']].iloc[0:n]
    pred_df['Prediction'] = predictions
    return pred_df



def con_bas_biz_rec(df, n = 5):
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    df.fillna('', inplace = True)
    
    def combine(rows):
        return rows['rev_text']+' '+rows['categories']
    
    df['text'] = df.apply(combine
                          , axis = 1)
    
    user_rev_biz_scott = df.loc[df.city == 'Scottsdale']
    
    urbs_cond = user_rev_biz_scott.drop_duplicates(subset = 'business_id')
    
    count_matrix = CountVectorizer().fit_transform(urbs_cond['text'])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    biz = cosine_sim[0].argsort()[-(n+1):][::-1][1:]
    
    biz_perc = cosine_sim[0][biz]
    
    biz_dict = {x: y for x in urbs_cond.business_id for y in urbs_cond.biz_name}
    
    biz_df = urbs_cond[['business_id', 'biz_name']]
    
    biz_similar = []
    for idx in biz:
        biz_similar.append(biz_df.biz_name.iloc[idx])
    
    biz_dict = {'name': biz_similar
               , 'rating': biz_perc}
    
    return pd.DataFrame(biz_dict)










    

    
    


            
