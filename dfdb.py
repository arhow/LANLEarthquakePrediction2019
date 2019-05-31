import pandas as pd
import numpy as np
import copy
import os

class DFDB(object):
    
    '''
    data structure:
    table(dataframe) pd.DataFrame
    record(tuple) {'datetime':datetime.datetime.now(), 'kernel':'1827638172456.json', 'score':1.2, 'param':{'alpha':.1, }, 'df_his':pd.DataFrame(), 'y_pred':np.array(), 'parent':2, 'description':'no detail'}
    
    operation:
    load db from a file 
    insert a record
    update a record
    delete a record
    select copy a sub dataframe and  return a copy
    commit save
    rollback clear change in memory and reload file
    
    '''
    

#     key_string_error = TypeError('Key/name must be a string!')

    def __init__(self, location, auto_commit=False):
        '''Creates a database object and loads the data from the location path.
        If the file does not exist it will be created on the first update.
        '''
        self.location = location
        self.auto_commit = auto_commit
        self.load(self.location)
        return
    
    def set_auto_commit(auto_commit):
        '''reset auto_commit'''
        self.auto_commit = auto_commit
        return

    def load(self, location):
        '''Loads, reloads or changes the path to the db file'''
        location = os.path.expanduser(location)
        if os.path.exists(location):
            self.db = pd.read_pickle(location)
        else:
            self.db = pd.DataFrame()
        return
    
    def save(self):
        '''save df to the db file'''
        self.db.to_pickle(self.location)
        return
    
    def insert(self, record):
        '''insert a new record'''
        cache = list(self.db.T.to_dict().values())
        cache.append(record)
        self.db = pd.DataFrame(cache)
        del cache
        if self.auto_commit:
            self.save()
        return
    
    def select(self, idx_list=[], key_list=[]):
        
#          """
#          insert a new record 
#          idx_list is a index list
#          key_list is column name list
#          if some column's value is dict 
#          key cound use like key-key
#          for example param column's value is {'a':{'b':123}}
#          you can use ['a-b']
#          """
        
        assert type(idx_list) == list, 'idx_list should be a list'
        assert type(key_list) == list, 'key_list should be a list'
        columns_ = self.db.columns.tolist()
        original_columns = [col for col in key_list if col in columns_]
        decomposed_columns = [col for col in key_list if col not in columns_]
        
        if (len(idx_list) == 0) & (len(key_list) == 0):
            df_sub =  copy.deepcopy(self.db.loc[:,:])
        elif (len(idx_list) != 0) & (len(key_list) == 0):
            df_sub =  copy.deepcopy(self.db.loc[idx_list,:])
        elif (len(idx_list) == 0) & (len(key_list) != 0):
            df_sub =  copy.deepcopy(self.db.loc[:,key_list])
        else:
            df_sub =  copy.deepcopy(self.db.loc[idx_list,key_list])
            
        for col in decomposed_columns:
            keys = col.split('-')
            series = df_sub[keys[0]]
            for k in keys[1:]:
                series = series.apply(lambda x : x[k])
            df_sub[col] = series
            
        return df_sub
    
    def update(self, idx, key, value):
        '''update a record key's value'''
        self.db.loc[idx,key] = copy.deepcopy(value)
        if self.auto_commit:
            self.save()
        return
    
    def delete(self, idx):
        '''del a record '''
        self.db = self.db.drop([idx])
        if self.auto_commit:
            self.save()
        return
    
    def commit(self):
        '''save cache to file '''
        self.save()
        return 
    
    def rollback(self):
        '''reload file'''
        self.load(self.location)
        return
    
    