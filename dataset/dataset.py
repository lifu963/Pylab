import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder,OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer


class Dataset(object):
    def __init__(self,df,
                 feature_cols,target_cols,
                 test_size=0.2,
                 stratified_features=None,
                 scaler=None,
                 imputer=None,fill_value=None,n_neighbors=None,
                 cat_features=None,isCatted=None,
                 cat_imputer=None,cat_fill_value=None,cat_n_neighbors=None
                ):
        
        if stratified_features is None:
            train_set ,test_set = train_test_split(df,test_size=test_size)
        else:
            split = StratifiedShuffleSplit(n_splits=1,test_size=test_size)
            for train_index,test_index in split.split(df,df[stratified_features]):
                train_set = df.loc[train_index]
                test_set = df.loc[test_index]
        
        self.transformer = Transformer(feature_cols,scaler=scaler,
                                       imputer=imputer,fill_value=fill_value,n_neighbors=n_neighbors,
                                       cat_features=cat_features,isCatted=isCatted,
                                       cat_imputer=cat_imputer,cat_fill_value=cat_fill_value,cat_n_neighbors=cat_n_neighbors)
        
        self.trainX = self.transformer.fit_transform(train_set[feature_cols])
        self.trainY = train_set[target_cols]
        self.testX = self.transformer.transform(test_set[feature_cols])
        self.testY = test_set[target_cols]
        
    def Get_transformer(self):
        return self.transformer
    
    def GetTrainSet(self):
        return self.trainX,self.trainY
    
    def GetTestSet(self):
        return self.testX,self.testY



class DefaultFormer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X

    
    
class DropImputer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return pd.DataFrame(X).dropna(axis=0).values
    

    
class Catmapper(BaseEstimator,TransformerMixin):
    def __init__(self,cat_features):
        self.cat_features = list(cat_features)
        self.cat_dict = {}
    
    def fit(self,X,y=None):
        for cat_feature in self.cat_features:
            cat_value = list(X[cat_feature].value_counts().index)
            label_value = list(range(len(cat_value)))
            map_dict = {cat:label for cat,label in zip(cat_value,label_value)}
            self.cat_dict[cat_feature] = map_dict
        return self
    
    def transform(self,X):
        X_ = X.copy()
        for cat_feature in self.cat_features:
            map_dict = self.cat_dict[cat_feature]
            X_[cat_feature] = X_[cat_feature].map(map_dict)
        return X_

    
class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self,features,scaler=None,
                 imputer=None,fill_value=None,n_neighbors=None,
                 cat_features=None,isCatted=None,
                 cat_imputer=None,cat_fill_value=None,cat_n_neighbors=None,
                 ):
        
        num_cols = features.copy()
        if cat_features is not None:
            for cat in list(cat_features):
                num_cols.remove(cat)
        
        try:
            if scaler is None:
                Scaler = DefaultFormer()
            elif scaler is 'standard':
                Scaler = StandardScaler()
            elif scaler is 'minmax':
                Scaler = MinMaxScaler()
            else:
                raise Exception('scaler设置错误')
        except Exception as e:
            print(e)
        
        Imupter = self.get_imputer(imputer,fill_value,n_neighbors)
        
        num_pipeline = Pipeline([
            ('imputer',Imupter),
            ('scaler',Scaler),
        ])
        
        cat_Imupter = self.get_imputer(cat_imputer,cat_fill_value,cat_n_neighbors)
        
        try:
            if isCatted is None:
                Catter = DefaultFormer()
            elif isCatted == 'ordinal':
                Catter = OrdinalEncoder()
            elif isCatted == 'onehot':
                Catter = OneHotEncoder()
            else:
                raise Exception('isCatted设置错误')
        except Exception as e:
            print(e)
        
        if cat_imputer is not None or isCatted is not None:
            assert cat_features is not None
            catmapper = Catmapper(cat_features)
        else:
            catmapper = DefaultFormer()
        
        
        cat_pipeline = Pipeline([
            ('cat_imputer',cat_Imupter),
            ('catter',Catter),
        ])
        
        if cat_features is not None:
            cols_pipeline = ColumnTransformer([
                ('num',num_pipeline,num_cols),
                ('cat',cat_pipeline,cat_features)])
        else:
            cols_pipeline = num_pipeline
            
        self.full_pipeline = Pipeline([
            ('mapper',catmapper),
            ('cols',cols_pipeline),
        ])
            
    def get_imputer(self,imputer,fill_value,n_neighbors):
        if imputer is None:
                Imupter = DefaultFormer()
                return Imupter
        else:
            try:
                if imputer is 'knn':
                    assert type(n_neighbors)==type(1) 
                    Imupter = KNNImputer(n_neighbors=n_neighbors)
                elif imputer is 'drop':
                    Imupter = DropImputer()
                else:
                    Imupter = SimpleImputer(strategy=imputer,fill_value=fill_value)
                return Imupter
            except Exception as e:
                print('imputer设置错误')
              
    def fit(self,X,y=None):
        return self.full_pipeline.fit(X)
    
    def transform(self,X):
        return self.full_pipeline.transform(X)