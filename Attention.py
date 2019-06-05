#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
import numpy as np
import tensorflow as tf

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
              
            #eij = K.tanh(K.dot(x, self.W)) #worked but this is self
            eij=K.dot(x, self.W) 

            if self.bias:
                eij += self.b
            eij = dot_product(eij, self.u)                
            ai = K.exp(eij)

            weights = ai/K.cast(K.sum(ai, axis=1, keepdims=True) , K.floatx()) #+ K.epsilon()
    
            weights = K.expand_dims(weights)
            weighted_input = x * weights

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), weights]
            else:
                return K.sum(weighted_input, axis=1)  


#~~~~~~~~~~~Local Attention
                
        elif self.attention_type=='local':
              

            eij=K.dot(x, self.W) 

            if self.bias:
                eij += self.b
            eij = dot_product(eij, self.u) 


            D=4
            pt=5            
            ii=np.array(range(np.shape(x)[-1]),dtype='float32')               
            ai = K.exp(eij+(-1.0*(ii-pt)**2/(D*D/2.0)))
            
            ai_local = K.exp(eij)
            ai_local=ai_local[:,pt-D:pt+D]
            weights = ai/K.cast(K.sum(ai_local, axis=1, keepdims=True) , K.floatx()) 
    
            weights = K.expand_dims(weights)
            weighted_input = x * weights

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), weights]
            else:
                return K.sum(weighted_input, axis=1)              
            
########################################            
                
        elif self.attention_type=='localFULL':
              
            eij=K.dot(x, self.W) 

            if self.bias:
                eij += self.b
            eij = dot_product(eij, self.u) 

            D=3  # local atention window size user defined

            pt=int(np.shape(x)[-1])*K.eye(int(np.shape(x)[-1])) #Tx
            sig=K.sigmoid(K.tanh(eij)) #sigmoid part 
            pt=pt*sig #Tx*sigmoid

            print(K.eval(pt))
            
            ii=np.array(range(np.shape(x)[-1]),dtype='float32')               
            ai = K.exp(eij+(-1.0*(ii-pt)**2/(D*D/2.0)))
            
            ai_local = K.exp(eij)
            ai_local = K.dot(ai_local,K.reshape(pt,(-1,-1)))

            weights = ai/K.cast(K.sum(ai_local, axis=1, keepdims=True) , K.floatx()) 
    
            weights = K.expand_dims(weights)
            weighted_input = x * weights

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), weights]
            else:
                return K.sum(weighted_input, axis=1)              
 
        
#~~~~~~~~~~~Self Attention  
        elif self.attention_type=='self':
              
            eij = K.tanh(K.dot(x, self.W)) 


            if self.bias:
                eij += self.b
            eij = dot_product(eij, self.u)                
            ai = K.exp(eij)

            weights = ai/K.cast(K.sum(ai, axis=1, keepdims=True) , K.floatx()) 
    
            weights = K.expand_dims(weights)
            weighted_input = x * weights

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), weights]
            else:
                return K.sum(weighted_input, axis=1)  


        
        else: pass
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

