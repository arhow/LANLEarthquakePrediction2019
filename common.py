import types
import pandas as pd
import numpy as np
import os
import datetime
from IPython.lib import kernel

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from fastFM import als, mcmc, sgd
from rgf.sklearn import RGFRegressor
#from pyfm import pylibfm

from scipy import sparse

import eli5
from eli5.sklearn import PermutationImportance

from models import *

class EP:

    def str2class(s):
        if s in globals() and isinstance(globals()[s], type):
                return globals()[s]
        if isinstance(eval(s), type):
            return eval(s)
        return None

    def check_param(param):
        def check_param_lvl_i(target_dict, base_dict, prefix):
            for k, v in base_dict.items():
                if k not in target_dict:
                    raise Exception('{} {} is not existed in param'.format(prefix, k))
                if type(v) is dict:
                    check_param_lvl_i(target_dict[k], v, prefix + k if prefix == '' else '-{}'.format(k))

        base_param = {
            'columns': [],
            'kfold': {
                'type': False,
                'n_splits': 5,
                'shuffle': True,
                'random_state': 1985,
            },
            'scaler': {
                'cls': 'StandardScaler',
            },
            'algorithm': {
                'cls': 'RandomForestRegressor',
                'init': {
                },
                'fit': {
                },
            },

        }
        check_param_lvl_i(param, base_param, '')
        return True

    #version2=>version3 set is_output_feature_importance from args not from param 
    #version1=>version2 use all data train(train and valid) and test to fit a scaler
    #version1
    def process(df_train, param, df_test=None, trial=None, remark=None, is_output_feature_importance=False):

        columns = param['columns']
        
        assert 'y' in df_train.columns.tolist(), 'y is not in df_train'
        assert 'index' in df_train.columns.tolist(), 'index is not in df_train'
        assert 'index' not in param['columns'], 'index is in features'
        assert 'y' not in param['columns'], 'y is in features'
        assert 'label' not in param['columns'], 'label is in features'
        assert 'group' not in param['columns'], 'group is in features'
        assert EP.check_param(param), 'param format is not right '
        assert (type(trial) == list) | (trial == None), 'trial is neither list nor none'
        assert len(columns) != 0, 'columns size is 0'

        df_test_pred = None
        if type(df_test) == pd.DataFrame:
            assert 'index' in df_test.columns.tolist(), 'index is not in df_test'
            df_test_pred = pd.concat([df_test_pred, df_test[['index']]], axis=1)

        history = []
        df_valid_pred = pd.DataFrame()
        df_feature_importances_i_list = []

        # stratified,group,timeseries
        if 'splits' in param['kfold']:
            splits = param['kfold']['splits']
        else:
            if param['kfold']['type'] == 'stratified':
                assert 'label' in df_train.columns.tolist(), 'label is not in df_train'
                folds = StratifiedKFold(n_splits=param['kfold']['n_splits'], shuffle=param['kfold']['shuffle'],
                                        random_state=param['kfold']['random_state'])
                splits = list(folds.split(df_train, df_train['label']))
            elif param['kfold']['type'] == 'group':
                assert 'group' in df_train.columns.tolist(), 'group is not in df_train'
                folds = GroupKFold(n_splits=param['kfold']['n_splits'])
                splits = list(folds.split(df_train, groups=df_train['group']))
            elif param['kfold']['type'] == 'timeseries':
                folds = TimeSeriesSplit(n_splits=param['kfold']['n_splits'])
                splits = list(folds.split(df_train))
            else:
                folds = KFold(n_splits=param['kfold']['n_splits'], shuffle=param['kfold']['shuffle'],
                              random_state=param['kfold']['random_state'])
                splits = list(folds.split(df_train))

        scaler_cls = EP.str2class(param['scaler']['cls'])
        regressor_cls = EP.str2class(param['algorithm']['cls'])
        permutation_random_state = 42
        
        if scaler_cls != None:
            scaler = scaler_cls(**param['scaler']['init'])
            if type(df_test) == pd.DataFrame:
                scaler.fit(np.concatenate([df_train[columns].values, df_test[columns].values], axis=0))
            else:
                scaler.fit(df_train[columns].values)

        for fold_n, (train_index, valid_index) in enumerate(splits):
            
            if (len(columns)==1)&(columns[0]=='X'):
                X = np.array(df_train['X'].values.tolist())
                X_train, X_valid = X[train_index, :], X[valid_index, :]
                y_train, y_valid = df_train['y'].values[train_index], df_train['y'].values[valid_index]
            else:
                X_train, X_valid = df_train[columns].values[train_index, :], df_train[columns].values[valid_index, :]
                y_train, y_valid = df_train['y'].values[train_index], df_train['y'].values[valid_index]

            if scaler_cls != None:
                X_train = scaler.transform(X_train)
                X_valid = scaler.transform(X_valid)

            algorithm_init_param = param['algorithm']['init'].copy()
            if 'alias' in list(algorithm_init_param.keys()):
                algorithm_init_param['alias'] = algorithm_init_param['alias'] + '_{}'.format(fold_n)
            model = regressor_cls(**algorithm_init_param)
            
            fit_param = param['algorithm']['fit'].copy()
            if 'eval_set' in fit_param:
                fit_param['eval_set'] = [(X_valid, y_valid)]
                
            if 'FMRegression' in param['algorithm']['cls']:
                X_train = sparse.csc_matrix(X_train)
                X_valid = sparse.csc_matrix(X_valid)
                
            model.fit(X_train, y_train, **fit_param)

            y_valid_pred = model.predict(X_valid)
            y_train_pred = model.predict(X_train)

            original_index = df_train['index'].values[valid_index]
            df_valid_pred_i = pd.DataFrame(
                {'index': original_index, 'predict': y_valid_pred, 'fold_n': np.zeros(y_valid_pred.shape[0]) + fold_n})
            df_valid_pred = pd.concat([df_valid_pred, df_valid_pred_i], axis=0)

            if is_output_feature_importance:
                df_feature_importances_i = pd.DataFrame({'feature': columns, 'model_weight': model.feature_importances_})
                df_feature_importances_i = df_feature_importances_i.sort_values(by=['feature'])
                df_feature_importances_i = df_feature_importances_i.reset_index(drop=True)

                perm = PermutationImportance(model, random_state=permutation_random_state).fit(X_valid, y_valid)
                df_feature_importances_i2 = eli5.explain_weights_dfs(perm, feature_names=columns, top=len(columns))[
                    'feature_importances']
                df_feature_importances_i2 = df_feature_importances_i2.sort_values(by=['feature'])
                df_feature_importances_i2 = df_feature_importances_i2.reset_index(drop=True)
                df_feature_importances_i = pd.merge(df_feature_importances_i, df_feature_importances_i2, on='feature')
                df_feature_importances_i_list.append(df_feature_importances_i)

            if type(df_test) == pd.DataFrame:
                
                if (len(columns)==1)&(columns[0]=='X'):
                    X_test = np.array(df_test['X'].values.tolist())
                else:
                    X_test = df_test[columns].values
                
                if scaler_cls != None:
                    X_test = scaler.transform(X_test)
                    
                if 'FMRegression' in param['algorithm']['cls']:
                    X_test = sparse.csc_matrix(X_test)
                    
                y_test_pred = model.predict(X_test)
                df_test_pred_i = pd.DataFrame({fold_n: y_test_pred})
                df_test_pred = pd.concat([df_test_pred, df_test_pred_i], axis=1)

            history.append({'fold_n': fold_n, 'train': mean_absolute_error(y_train, y_train_pred),
                            'valid': mean_absolute_error(y_valid, y_valid_pred)})

        df_his = pd.DataFrame(history)

        df_feature_importances = None
        if is_output_feature_importance:
            df_feature_importances = df_feature_importances_i_list[0]
            for idx, df_feature_importances_i in enumerate(df_feature_importances_i_list[1:]):
                df_feature_importances = pd.merge(df_feature_importances, df_feature_importances_i, on='feature',
                                                  suffixes=('', idx + 1))

        df_valid_pred = df_valid_pred.sort_values(by=['index'])
        df_valid_pred = df_valid_pred.reset_index(drop=True)

        if type(df_test) == pd.DataFrame:
            df_test_pred = df_test_pred.sort_values(by=['index'])
            df_test_pred = df_test_pred.reset_index(drop=True)

        if type(trial) == list:
            pid_ = os.getpid()
            datetime_ = datetime.datetime.now()
            connection_file = os.path.basename(kernel.get_connection_file())
            val_mae_mean = np.mean(df_his.valid)
            val_mae_var = np.var(df_his.valid)
            train_mae_mean = np.mean(df_his.train)
            train_mae_var = np.var(df_his.train)

            trial.append({'datetime': datetime_, 'kernel': connection_file, 'remark': remark, 'val_mae': val_mae_mean,
                          'train_mae': train_mae_mean, 'val_mae_var': val_mae_var, 'train_mae_var': train_mae_var,
                          'mae_diff': val_mae_mean - train_mae_mean,
                          'df_his': df_his, 'df_feature_importances': df_feature_importances,
                          'df_valid_pred': df_valid_pred, 'df_test_pred': df_test_pred, 'param': param.copy(),
                          'nfeatures': len(columns)})

        return df_his, df_feature_importances, df_valid_pred, df_test_pred
    
    def evaluate(df_feature_importances, key='average_model_weight'):
        df_feature_importances['average_permutation_weight'] = df_feature_importances[
            [col for col in df_feature_importances.columns.tolist() if ('weight' in col) & ('model' not in col)]].mean(
            axis=1)
        df_feature_importances['average_model_weight'] = df_feature_importances[
            [col for col in df_feature_importances.columns.tolist() if ('model_weight' in col)]].mean(axis=1)
        df_feature_importances = df_feature_importances.sort_values(by=[key], ascending=False)
        sorted_columns = df_feature_importances.feature.tolist()
        return sorted_columns

    def select_features_(df_train, param, trial, df_test=None, nfeats_best=10, nfeats_removed_per_try=10, key='average_model_weight', remark=None):
        param_i = param.copy()
        while True:
            df_his, df_feature_importances, df_valid_pred, df_test_pred = EP.process(df_train, param_i, df_test=df_test, trial=trial, is_output_feature_importance=True, remark=remark)
            sorted_columns = EP.evaluate(df_feature_importances, key)
            if (len(sorted_columns) <= nfeats_best)|(len(sorted_columns)-nfeats_removed_per_try<1):
                break
            else:
                param_i['columns'] = sorted_columns[:-nfeats_removed_per_try]
        return