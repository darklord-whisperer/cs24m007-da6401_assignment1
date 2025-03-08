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
    class Activation :  contains all the necessary functions required for the artificial neurons
    sigmoid -> sigmoid activation function
    g3 -> relu activation function value_returned = max(a,0)
    softmax -> function for the final y_hat, this returns a probability distribution over all classes
'''

class Activations :
    def sigmoid(self,x) :
        x = np.clip(x,-200,200)
        return 1/(1 + np.exp(-x))
    
    def g3(self,a):
        return np.maximum(a,0)

    def SoftMax(self,a):
        max_a = np.max(a)
        exp_a = np.exp(a - max_a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
