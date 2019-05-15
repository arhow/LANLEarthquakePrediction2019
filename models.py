from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, CuDNNGRU, CuDNNLSTM, RepeatVector, RepeatVector, concatenate,ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop
from keras import optimizers
from keras import backend as K
from keras import Sequential,Input, Model
from keras.models import load_model
from keras.regularizers import L1L2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import pandas as pd
import os

def create_path(base_dir, param):
    if base_dir == None:
        return None
    fold_path = base_dir + '/' + ','.join("{!s}={!r}".format(key,val) for (key,val) in param.items())
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    return fold_path

class KerasMLPRegressor(object):
    
    def __init__(self, batch, input_dim, hidden_layer_sizes, activation, dropout, solver, metric, lr, sgd_momentum, sgd_decay, base_save_dir, alias):
        
        self.batch = batch
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.metric = metric
        self.dropout = dropout
        self.lr = lr
        self.sgd_momentum = sgd_momentum
        self.sgd_decay = sgd_decay
        
        self.regressor = self.build_graph(input_dim, hidden_layer_sizes, activation, dropout)
        self.compile_graph(self.regressor, solver, metric, lr, sgd_momentum, sgd_decay)
        
        self.alias = alias
        self.base_save_dir = base_save_dir
        if (self.alias==None) & (self.base_save_dir==None):
            self.chkpt = None
        else:
            self.chkpt = os.path.join(base_save_dir,'{}.hdf5'.format(alias))

        return
    
    def build_graph(self, input_dim, hidden_layer_sizes, activation, dropout):
    
        i = Input(shape = (input_dim,))
        x = Dense(hidden_layer_sizes[0], activation=activation)(i)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        for units in hidden_layer_sizes[1:-1]:
            x = Dense(units, activation=activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        x = Dense(units, activation=activation)(x)
        x = BatchNormalization()(x)
        y = Dense(1)(x)
        regressor = Model(inputs = [i], outputs = [y])
        return regressor
    
    def compile_graph(self, model, solver, metric, lr, momentum, decay):
        if solver=='adam':
            optimizer = optimizers.adam(lr=lr)
        elif solver=='sgd':
            optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(optimizer=optimizer, loss=metric)
        return
    
    def fit(self, X_train, y_train, eval_set, versbose=11, epochs=200, early_stopping_rounds=20):
        
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=early_stopping_rounds//4, min_lr=self.lr*1e-2)
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=1, mode='auto')
        cp_cb = ModelCheckpoint(filepath = self.chkpt, monitor='val_loss', verbose=versbose/10, save_best_only=True, mode='auto')

#         his_train = self.regressor.fit_generator( generator =  train_gen, epochs = epochs,  verbose = 1,  validation_data = validation, callbacks = [cp_cb])
        his_train = self.regressor.fit( X_train, y_train, epochs = epochs,  verbose = versbose%10,  validation_data = eval_set[0], callbacks = [cp_cb,es_cb])
        df_train_his = pd.DataFrame(his_train.history)
        
#         df_train_his = pd.DataFrame()
#         prev_val_loss = 999999
#         for i in np.arange(epochs):
#             his_train = self.regressor.fit( X_train, y_train, epochs = 1,  verbose = versbose,  batch_size = self.batch,  validation_data = validation,  callbacks = [])
#             df_train_his_i = pd.DataFrame(his_train.history)
#             df_train_his_i['epochs'] = i+1
#             df_train_his = pd.concat([df_train_his, df_train_his_i], axis=0)
#             if (df_train_his_i.val_loss.values[0] < prev_val_loss) & (self.chkpt!=None):
#                 prev_val_loss = df_train_his_i.val_loss.values[0]
#                 self.regressor.save_weights(self.chkpt)
                
        df_train_his.to_csv(self.base_save_dir + '/{}_train_his.csv'.format(self.alias), index=True)
            
        return df_train_his
    
    def predict(self, X, use_best_epoch=True):
        self.regressor.load_weights(self.chkpt)
        return self.regressor.predict(X)[:,0]
    
    