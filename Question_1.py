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
        
    def plot_sample_images(self):
        plt.figure(figsize=(12, 8))

        # Iterate through each class to plot a sample image for each
        for class_idx in range(len(self.class_labels)):
            # Find the first occurrence of an image with the current label
            first_index = np.where(self.train_labels == class_idx)[0][0]

            sample_image = self.train_images[first_index]
            label = self.class_labels[self.train_labels[first_index]]

            plt.subplot(2, 5, class_idx + 1)
            plt.imshow(sample_image, cmap=plt.cm.binary)
            plt.title(label)
            plt.axis('off')

        plt.show()

    def log_images_to_wandb(self):
        wandb.init(project="Alik_Final_DA6401_1_DeepLearing_Assignment1", name="Quuestion_1_Fashion_MNIST_Grid_Visualization")
        wandb.log({"Fashion_MNIST_Sample_Images": wandb.Image(plt)})
        
        wandb.finish()


def main():
    # Initializing visualizer class
    visualizer = FashionMNISTVisualizer()
    
    # Loading dataset
    visualizer.load_data()

    visualizer.plot_sample_images()

    visualizer.log_images_to_wandb()

if __name__ == "__main__":
    main()
