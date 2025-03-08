import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import fashion_mnist,mnist
import math
import Gradient_Descent as gd





'''
    class Differential :  contains all the necessary functions required for the artificial neurons to find the derivative of a function
    sig_dif -> differential function for sigmoid neuron
    tan_dif -> differential function for tanh neuron
    Rel_dif -> differential function for relu neuron
    Iden_dif ->
'''

class Differential :
    def sig_dif(self,a):
        # if f(x) = 1/1+e^(-x)
        # the f_dash(x) = f(x)*(1-f(x)), this is what is implemented here
        activ = Activations()
        g_x = activ.sigmoid(a)
        return g_x*(1-g_x)

    def tan_dif(self,a):
        # if f(x) = e^x - e^-x/e^x + e^-x
        # the f_dash(x) = (1 - f(x)^2), this is what is implemented here
        g_dash = np.tanh(a)
        return 1 - (g_dash**2)  

    def Rel_dif(self,a):
        #       if value inside entries of a>0 then set to true else set to false, 
        #       .astype('float64') converts true/false into 1/0
        return (a > 0).astype('float64')        
    
    def Iden_dif(self,a):
        # if f(x) = x
        # the f_dash(x) = 1, this is what is implemented here
        g_dash = a
        g_dash[:] = 1
        return g_dash