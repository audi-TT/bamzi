#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
import numpy as np

def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionLayer(Layer):

    
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,attention_type='self', **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.attention_type=attention_type
        
        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        
        super(AttentionLayer, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
#~~~~~~~~~~~Global Attention 
        if self.attention_type=='global':
            
            #uit = dot_product(x, self.W)
            #print ('x hsape:' , np.shape(x))
            #print ('W hsape:' , np.shape(self.W))            
            #print ('uit hsape:' , np.shape(uit))            

            #option 1:
            uit= K.reshape(x,(-1,100,100))

            print ('uit shape:' , np.shape(uit))
            #option 2:


            
            if self.bias:
                uit += self.b

            
            #uit = K.tanh(uit)
            ait = dot_product(uit, self.u) #score(ut,u)

            a = K.exp(ait)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), a]
            else:
                return K.sum(weighted_input, axis=1)
    
#~~~~~~~~~~~Local Attention       
        elif self.attention_type=='local':
            ht = K.reshape(x,(-1,np.shape(x)[-1],np.shape(x)[-1])) 

            if self.bias:
                ht += self.b

            D=3  # local atention window size user defined
            pt=int(np.shape(x)[-1])*np.identity(np.shape(x)[-1], dtype='float32') #Tx
            sig=K.sigmoid(K.tanh(ht)) #sigmoid part
            pt=pt*sig #Tx*sigmoid

#play goround~~~~~~~~~~~~~~~

            print ('pt shape:' , np.shape(pt))
            ptt=K.squeeze(pt,axis=0)
            print ('ptt shape:' , np.shape(ptt))
            pttt=np.array(ptt)

   
            #except: 
            #    print('print row 2 array nashod')
#            pptt=5
#            a_local=a[:,pptt-D:pptt+D]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
            
            ait = dot_product(ht, self.u)

            ii=np.array(range(np.shape(x)[-1]),dtype='float32')
            a = K.exp(ait+(-1.0*(ii-pt)**2/(D*D/2.0)))

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())

##            pptt=5
##            a_local= K.exp(ait)[:,pptt-D:pptt+D]
            a_local=K.exp(ait)*ptt
###            try: a_local=K.exp(ait)*ptt
###            except: print('ptt nashod')

####            try: a_local=K.exp(ait)*pt
####            except: print('pt nashod')
            
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a_local, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), a]
            else:
                return K.sum(weighted_input, axis=1)    

        
#~~~~~~~~~~~Self Attention  
        elif self.attention_type=='self':
            uit = dot_product(x, self.W)

            if self.bias:
                uit += self.b

            uit = K.tanh(uit)
            ait = dot_product(uit, self.u)

            a = K.exp(ait)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), a]
            else:
                return K.sum(weighted_input, axis=1)
        
        else: pass
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

