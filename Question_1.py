import wandb
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

class FashionMNISTVisualizer:
    def __init__(self):
        self.class_labels = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self.train_images, self.train_labels = None, None
        self.fig = None  # We'll store the figure here

    def load_data(self):
        (self.train_images, self.train_labels), (_, _) = fashion_mnist.load_data()
        
    def plot_sample_images(self):
        """
        Creates a single figure with 10 subplots (one for each class).
        """
        self.fig = plt.figure(figsize=(12, 8))

        for class_idx in range(len(self.class_labels)):
            # Find the first occurrence of an image with the current label
            first_index = np.where(self.train_labels == class_idx)[0][0]
            sample_image = self.train_images[first_index]
            label = self.class_labels[self.train_labels[first_index]]

            plt.subplot(2, 5, class_idx + 1)
            plt.imshow(sample_image, cmap=plt.cm.binary)
            plt.title(label)
            plt.axis('off')

        plt.tight_layout()
        # Optionally, you can show the figure locally (comment out if running in a headless environment):
        # plt.show()

    def log_images_to_wandb(self):
        """
        Logs the figure (which contains all 10 images) to Weights & Biases.
        """
        wandb.init(project="Alik_Final_DA6401_DeepLearning_Assignment1",
                   name="Question_1_Fashion_MNIST_Grid_Visualization")

        # Log the stored figure as a WandB image
        # Using wandb.Image(self.fig) ensures the entire figure is logged.
        wandb.log({"Fashion_MNIST_Sample_Images": wandb.Image(self.fig)})
        
        wandb.finish()


def main():
    # Initialize the visualizer
    visualizer = FashionMNISTVisualizer()
    
    # Load the Fashion-MNIST dataset
    visualizer.load_data()

    # Create the figure with sample images
    visualizer.plot_sample_images()

    # Log the figure to Weights & Biases
    visualizer.log_images_to_wandb()

if __name__ == "__main__":
    main()
