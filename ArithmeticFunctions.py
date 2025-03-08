import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import fashion_mnist,mnist
import math
import Gradient_Descent as gd




class Arithmetic :
    def Add(self,u,v):          #Adds two matrices u,v of the same order
        for i in range(1,len(u)):
            u[i] = u[i] + v[i]
        return u

    def Subtract(self,v,dv,eta):    #Update rule : W(t+1) = W(t) + eta*delW
        for i in range(1,len(v)):
            v[i] = v[i] - (eta * dv[i])
        return v

    def RMSpropSubtract(self,v,dv,lv,eps,eta):     #Update rule : W(t+1) = W(t) + (eta/(sqrt(v) + Epsilon)*delW
        for i in range(1,len(v)):
            ueta = eta/(np.sqrt(np.sum(lv[i])) + eps)
            v[i] = v[i] - (ueta * dv[i])
        return v

    def AdamSubtract(self,V,mV_hat,vV_hat,eps,eta):     #Update rule : W(t+1) = W(t) + (eta/(sqrt(v) + Epsilon)*delW
        for i in range(1,len(V)):
            norm = np.linalg.norm(vV_hat[i])
            ueta = eta/(np.sqrt(norm) + eps)
            V[i] = V[i] - (ueta * mV_hat[i])
        return V

