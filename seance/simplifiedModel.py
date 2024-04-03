import pandas as pd
import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegressor :
    def __init__(self,learning_rate=0.1,max_iter=100) :
        self.weights=[]
        self.max_iter=max_iter
        self.learning_rate=learning_rate
    def fit (self,x,y)  :
        x=np.array(x)
        x_with_bias=np.column_stack((np.ones(x.shape[0]),x))
        y=np.array(y)
        self.weights=np.random.rand(x_with_bias.shape[1])
        # learning process
        for _ in range(self.max_iter):
            # calculate y_predict
            y_predict=x_with_bias @self.weights
            # proba of this predicted y:
            y_predict_proba=sigmoid(y_predict)
            # calculate likelihood :
            gradient=x_with_bias.T @ (y-y_predict_proba)
            
            # update weights :
            self.weights +=self.learning_rate*gradient
    def predict(self,x):
        x=np.column_stack((np.ones(x.shape[0]),np.array(x)))
        y=sigmoid(x @self.weights)
        y= [ 1 if proba>0.5 else 0 for proba in y  ]
        return y
            
            
            
        
         