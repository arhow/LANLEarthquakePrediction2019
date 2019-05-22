from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, CuDNNGRU, CuDNNLSTM, RepeatVector, RepeatVector, concatenate,ConvLSTM2D
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Convolution1D,TimeDistributed,Lambda, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros
from keras.optimizers import SGD, RMSprop
from keras import optimizers
from keras import backend as K
from keras import Sequential,Input, Model
from keras.models import load_model
from keras.regularizers import L1L2
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import numpy as np

import pandas as pd
import os

# https://www.kaggle.com/shujian/transformer-with-lstm

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn
    
class ReshapeStandardScaler(object):
    
    def  __init__(self, shape, mean, std):
        
        assert shape[-1] == len(std.shape), 'the shape is not matched'
        assert shape[-1] == len(mean.shape), 'the shape is not matched'
        self.shape = shape
        self.std = std
        self.mean = mean
        return
    
    def fit(self, *args, **kwargs):
        return
    
    def transform(self, X):
        original_shape = X.shape
        X = (X.reshape(self.shape) - self.mean)/self.std
        return X.reshape(original_shape)


def create_path(base_dir, param):
    if base_dir == None:
        return None
    fold_path = base_dir + '/' + ','.join("{!s}={!r}".format(key,val) for (key,val) in param.items())
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    return fold_path

class KerasMLPRegressor(object):
    
    def __init__(self, batch, input_dim, hidden_layer_sizes, activation, dropout, l1l2regularizer, solver, metric, lr, sgd_momentum, sgd_decay, base_save_dir, alias):
        
        self.batch = batch
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.metric = metric
        self.dropout = dropout
        self.l1l2regularizer = l1l2regularizer
        self.lr = lr
        self.sgd_momentum = sgd_momentum
        self.sgd_decay = sgd_decay
        
        self.regressor = self.build_graph(input_dim, hidden_layer_sizes, activation, dropout, l1l2regularizer)
        self.compile_graph(self.regressor, solver, metric, lr, sgd_momentum, sgd_decay)
        
        self.alias = alias
        self.base_save_dir = base_save_dir
        if (self.alias==None) & (self.base_save_dir==None):
            self.chkpt = None
        else:
            self.chkpt = os.path.join(base_save_dir,'{}.hdf5'.format(alias))

        return
    
    def build_graph(self, input_dim, hidden_layer_sizes, activation, dropout, l1l2regularizer):
        
        if type(l1l2regularizer) == type(None):
            regularizer=None
        else:
            regularizer = regularizers.l1_l2(l1l2regularizer)
    
        i = Input(shape = (input_dim,))
        x = Dense(hidden_layer_sizes[0], activation=activation, kernel_regularizer=regularizer, activity_regularizer=regularizer)(i)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        for units in hidden_layer_sizes[1:-1]:
            x = Dense(units, activation=activation, kernel_regularizer=regularizer, activity_regularizer=regularizer)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        x = Dense(hidden_layer_sizes[-1], activation=activation, kernel_regularizer=regularizer, activity_regularizer=regularizer)(x)
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
    
    def fit(self, X_train, y_train, eval_set, versbose=1, epochs=200, early_stopping_rounds=20):
        
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=early_stopping_rounds//4, min_lr=self.lr*1e-2)
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=1, mode='auto')
        cp_cb = ModelCheckpoint(filepath = self.chkpt, monitor='val_loss', verbose=versbose, save_best_only=True, mode='auto')

#         his_train = self.regressor.fit_generator( generator =  train_gen, epochs = epochs,  verbose = 1,  validation_data = validation, callbacks = [cp_cb])
        his_train = self.regressor.fit( X_train, y_train, epochs = epochs,  verbose = versbose,  validation_data = eval_set[0], callbacks = [])
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
    
    def predict(self, X, use_best_epoch=False):
        if use_best_epoch:
            self.regressor.load_weights(self.chkpt)
        return self.regressor.predict(X)[:,0]
    
class Generator(keras.utils.Sequence):

    def __init__(self, x, y, x_mean, x_std, start_indexes, ts_length, batch_size, steps_per_epoch, shaking=True):
        self.x = x
        self.y = y
        self.start_indexes = start_indexes
        self.ts_length = ts_length
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.x_mean = x_mean
        self.x_std = x_std
        self.shaking = shaking
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        
        start_indexes_epoch = np.random.choice(self.start_indexes, size=self.batch_size)
        if self.shaking:
            shifts = np.random.randint(0, int(self.ts_length*.2), size=self.batch_size) - int(self.ts_length*.1)
        else:
            shifts = np.zeros(self.batch_size)
            
        x_batch = np.empty((self.batch_size, self.ts_length))
        y_batch = np.empty(self.batch_size, )

        for i, start_idx in enumerate(start_indexes_epoch):
            end = start_idx + shifts[i] + self.ts_length
            if end < self.ts_length:
                end = self.ts_length
            if end >= self.x.shape[0]:
                end = self.x.shape[0]
            x_i = self.x[end-self.ts_length:end]
            x_batch[i, :] = x_i
            y_batch[i] = self.y[end - 1]
            
        x_batch = (x_batch - self.x_mean)/self.x_std

        return np.expand_dims(x_batch, axis=2), y_batch
    

class Keras1DCnnRegressor(object):
    
    def __init__(self, batch, timesteps, input_dim, cnn_layer_sizes, cnn_kernel_size, cnn_strides, cnn_activation, 
                    fc_layer_sizes, fc_activation, dropout, solver, metric, lr, sgd_momentum, sgd_decay, base_save_dir, alias, 
                 attention_n_head=5, attention_d_model=256, attention_d_k=64, attention_d_v=64, bilstm_layer_sizes=[]):
        
        self.batch = batch
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.cnn_layer_sizes = cnn_layer_sizes
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_strides = cnn_strides
        self.cnn_activation = cnn_activation
        self.fc_layer_sizes = fc_layer_sizes
        self.fc_activation = fc_activation
        self.dropout = dropout
        self.solver = solver
        self.metric = metric
        self.lr = lr
        self.sgd_momentum = sgd_momentum
        self.sgd_decay = sgd_decay
        
        self.regressor = self.build_graph(timesteps, input_dim, cnn_layer_sizes, cnn_kernel_size, cnn_strides, cnn_activation, 
                    fc_layer_sizes, fc_activation, attention_n_head, attention_d_model, attention_d_k, attention_d_v, bilstm_layer_sizes, dropout)
        self.compile_graph(self.regressor, solver, metric, lr, sgd_momentum, sgd_decay)
        
        self.alias = alias
        self.base_save_dir = base_save_dir
        if (self.alias==None) & (self.base_save_dir==None):
            self.chkpt = None
        else:
            self.chkpt = os.path.join(base_save_dir,'{}.hdf5'.format(alias))

        return
    
    def build_graph(self, timesteps, input_dim, cnn_layer_sizes, cnn_kernel_size, cnn_strides, cnn_activation, 
                    fc_layer_sizes, fc_activation, attention_n_head, attention_d_model, attention_d_k, attention_d_v, bilstm_layer_sizes, dropout):
        
        i = Input(shape = (timesteps, input_dim))
        x = Convolution1D( cnn_layer_sizes[0], kernel_size = cnn_kernel_size, strides = cnn_strides, activation=cnn_activation)(i)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        for units in cnn_layer_sizes[1:]:
            x = Convolution1D(units, kernel_size = cnn_kernel_size, strides = cnn_strides, activation=cnn_activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        for units in bilstm_layer_sizes:
            x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)
        x, slf_attn = MultiHeadAttention(n_head=attention_n_head, d_model=attention_d_model, d_k=attention_d_k, d_v=attention_d_v, dropout=dropout)(x, x, x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        for units in fc_layer_sizes[:-1]:
            x = Dense(units, activation=fc_activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        x = Dense(fc_layer_sizes[-1], activation=fc_activation)(x)
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
    
    def fit_generator(self, train_gen, eval_set, verbose=1, epochs=200):
        
        
        df_train_his = pd.DataFrame()
#         prev_val_loss = 999999
        for i in np.arange(epochs):
            if type(eval_set)==type(None):
                validation_data = None
            else:
                validation_data = eval_set[0]
            his_train = self.regressor.fit_generator( generator =  train_gen,  epochs = 1,  verbose = 0,  validation_data = validation_data, callbacks = [])
            df_train_his_i = pd.DataFrame(his_train.history)
            df_train_his_i['epochs'] = i
            df_train_his = pd.concat([df_train_his, df_train_his_i], axis=0)
            
            if verbose > 0:
                if validation_data == None:
                    print(df_train_his_i.epochs.values, df_train_his_i.loss.values)
                else:
                    print(df_train_his_i.epochs.values, df_train_his_i.loss.values, df_train_his_i.val_loss.values)
                
#             if (df_train_his_i.val_loss.values[0] < prev_val_loss) & (self.chkpt!=None) :
#                 prev_val_loss = df_train_his_i.val_loss.values[0]
#                 self.regressor.save_weights(self.chkpt)
        
        df_train_his.to_csv(self.base_save_dir + '/train_his.csv', index=True)
        return
    
    def fit(self, X_train, y_train, eval_set, verbose=1, epochs=200):
              
        df_train_his = pd.DataFrame()
#         prev_val_loss = 999999
        for i in np.arange(epochs):
            if type(eval_set)==type(None):
                validation_data = None
            else:
                validation_data = eval_set[0]
                assert type(eval_set[0])==tuple, 'validation_data[0] is not a tuple'
            his_train = self.regressor.fit( X_train, y_train, epochs = 1,  verbose = 0,  batch_size = self.batch,  validation_data = validation_data,  callbacks = [])
            df_train_his_i = pd.DataFrame(his_train.history)
            df_train_his_i['epochs'] = i
            df_train_his = pd.concat([df_train_his, df_train_his_i], axis=0)
            
            if verbose > 0:
                if validation_data == None:
                    print(df_train_his_i.epochs.values, df_train_his_i.loss.values)
                else:
                    print(df_train_his_i.epochs.values, df_train_his_i.loss.values, df_train_his_i.val_loss.values)
                
#             if (df_train_his_i.val_loss.values[0] < prev_val_loss) & (self.chkpt!=None) :
#                 prev_val_loss = df_train_his_i.val_loss.values[0]
#                 self.regressor.save_weights(self.chkpt)
                
        df_train_his.to_csv(self.base_save_dir + '/train_his.csv', index=True)
            
        return df_train_his
    
    def predict(self, X):
        return self.regressor.predict(X)[:,0]