import wandb
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

class FashionMNISTVisualizer:
    def __init__(self):
        self.class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.train_images, self.train_labels = None, None

    def load_data(self):
        (self.train_images, self.train_labels), (_, _) = fashion_mnist.load_data()
