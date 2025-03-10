<div style="text-align: justify;">

# DA6410 : Fundamentals of Deep Learning - Assignment 1

This repository contains all files for the first assignment of the DA6410 - Fundamentals of Deep Learning course at IIT Madras. The assignment involves building a FeedForward Neural Network (FFNN) from scratch with backpropagation, implementing multiple optimizers, and performing hyperparameter sweeps and visualizations using Weights & Biases (WandB).

## Contents

- [Task](#task)
- [Submission](#submission)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
  - [Structure](#structure)
  - [Optimizers](#optimizers)
  - [Criterion](#criterion)
  - [Backpropagation](#backpropagation)
  - [Flexibility](#flexibility)
- [Tools and Libraries Used](#tools-and-libraries-used)
  - [Packages Used](#packages-used)
  - [Requirements and Installation](#requirements-and-installation)
- [Usage](#usage)
  - [Running Manually](#running-manually)
  - [Running a Sweep using WandB](#running-a-sweep-using-wandb)
  - [Customization](#customization)

## Task

The primary task is to implement a FeedForward Neural Network (FFNN) from scratch with the following capabilities:
- **Forward and Backward Propagation:** Custom implementation with gradient clipping and learning rate decay.
- **Activation Functions:** Multiple activations (Sigmoid, ReLU, Softmax) and their derivatives.
- **Optimizers:** Support for SGD, Momentum, NAG, RMSProp, Adam, and Nadam.
- **Loss Functions:** Both Cross Entropy and Mean Squared Error (MSE) losses.
- **Hyperparameter Tuning:** Ability to run experiments and sweeps using WandB.
- **Visualization:** Tools to visualize sample images from Fashion MNIST and interactive confusion matrices.

## Submission

- **WandB Project:** [Your WandB Project Link]
- **WandB Report:** [Your WandB Report Link]

## Dataset

The assignment uses two popular datasets from the `keras.datasets` module:
- **Fashion MNIST:** Loaded using `from keras.datasets import fashion_mnist`
- **MNIST:** Loaded using `from keras.datasets import mnist`

The datasets are normalized (pixel values scaled to [0, 1]) and the training data is split into training, validation, and test sets (commonly an 80/10/10 split).

## Implementation Details

### Structure

The repository is organized into several modules, each addressing a specific functionality:

- **ActivationFunctions.py:**  
  Defines the `Activations` class which provides:
  - `sigmoid`: Implements the sigmoid activation.
  - `g3`: Implements the ReLU activation (returns `max(a, 0)`).
  - `SoftMax`: Implements the softmax function to produce a probability distribution.

- **ArithmeticFunctions.py:**  
  Contains the `Arithmetic` class with methods to perform matrix arithmetic required for parameter updates:
  - `Add`: Matrix addition.
  - `Subtract`: Standard parameter update.
  - `RMSpropSubtract` & `AdamSubtract`: Optimizer-specific update rules.

- **DifferentialFunctions.py:**  
  Implements the `Differential` class for computing derivatives:
  - `sig_dif`: Derivative of the sigmoid function.
  - `tan_dif`: Derivative of the tanh function.
  - `Rel_dif`: Derivative of the ReLU function.
  - `Iden_dif`: Derivative of the identity function.

- **Question_1.py:**  
  Contains the `FashionMNISTVisualizer` class which:
  - Loads the Fashion MNIST dataset.
  - Plots a grid of one sample image per class.
  - Logs the generated figure to WandB.

- **train_better_accuracy_test_cross_entropy.py:**  
  Implements the training pipeline for the FFNN using the cross-entropy loss function. It includes:
  - Command-line argument parsing for hyperparameter configuration.
  - Data loading with an 80/10/10 train/validation/test split.
  - Initialization of network weights (with Xavier or random initialization).
  - Forward and backward passes with gradient clipping.
  - Dynamic optimizer updates supporting SGD, Momentum, NAG, RMSProp, Adam, and Nadam.
  - Logging of performance metrics and both standard and interactive confusion matrices to WandB.
  - Support for hyperparameter sweeps.

- **train_better_accuracy_test_mse.py:**  
  Similar to the cross-entropy version, this script trains the FFNN using Mean Squared Error (MSE) as the loss function.

### Optimizers

The following optimizers are implemented:
- **SGD (Stochastic Gradient Descent)**
- **Momentum**
- **NAG (Nesterov Accelerated Gradient)**
- **RMSProp (Root Mean Square Propagation)**
- **Adam (Adaptive Moment Estimation)**
- **Nadam (Nesterov Adaptive Moment Estimation)**

### Criterion

Two loss functions are provided:
- **Cross Entropy:** Ideal for classification tasks.
- **Mean Squared Error (MSE):** An alternative for regression-style outputs.

### Backpropagation

Key features include:
- **Forward Pass:** Computes activations for each layer using the selected activation function.
- **Backward Pass:** Uses the chain rule to compute gradients and updates weights accordingly.
- **Gradient Clipping:** Applied to mitigate exploding gradients.
- **Learning Rate Decay:** Dynamically adjusts the learning rate after each epoch.

### Flexibility

The training scripts are highly configurable via command-line arguments. Options include:
- Choosing the dataset (`fashion_mnist` or `mnist`).
- Selecting the loss function (`cross_entropy` or `mean_squared_error`).
- Configuring the optimizer and its associated hyperparameters.
- Setting the number of epochs, batch size, number of layers, hidden layer size, and activation function.
- Running a single training run or a hyperparameter sweep via WandB.

## Tools and Libraries Used

- **Python 3.10.1** (or higher)
- **Numpy:** For numerical computations.
- **Matplotlib:** For plotting and visualization.
- **Keras:** For loading the MNIST and Fashion MNIST datasets.
- **Scikit-learn:** For data preprocessing and confusion matrix computations.
- **WandB (Weights & Biases):** For experiment tracking, hyperparameter tuning, and logging.
- **Plotly:** For interactive visualization of confusion matrices.

### Packages Used

1. **Numpy**
2. **Matplotlib**
3. **Keras**
4. **WandB**
5. **Scikit-learn**
6. **Plotly**

### Requirements and Installation

The project uses a `requirements.txt` file to manage dependencies. The following packages (with specified versions) are required:

wandb==0.16.4

scikit-learn==0.24.2

numpy==1.21.0

matplotlib==3.4.2

keras==2.7.0

## Installation:

### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### Create a Virtual Environment (Optional but Recommended)
```bash 
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

#### On Windows use: 
``` bash 
venv\Scripts\activate
```

### Install the Dependencies
``` bash 
pip install -r requirements.txt
```

## Usage
### Running Manually
#### Cross-Entropy Loss Version:

```bash
python train_better_accuracy_test_cross_entropy.py \
    --wandb_project Your_Project_Name \
    --wandb_entity Your_Entity_Name \
    --dataset fashion_mnist \
    --epochs 10 \
    --batch_size 64 \
    --loss cross_entropy \
    --optimizer adam \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --beta 0.9 \
    --beta1 0.9 \
    --beta2 0.999 \
    --epsilon 1e-8 \
    --weight_decay 0.0005 \
    --weight_init xavier \
    --num_layers 2 \
    --hidden_size 128 \
    --activation ReLU
```

### MSE Loss Version :

  ```bash
  python train_better_accuracy_test_mse.py \
    --wandb_project Your_Project_Name \
    --wandb_entity Your_Entity_Name \
    --dataset fashion_mnist \
    --epochs 5 \
    --batch_size 32 \
    --loss mean_squared_error \
    --optimizer sgd \
    --learning_rate 0.1 \
    --momentum 0.5 \
    --beta 0.5 \
    --beta1 0.5 \
    --beta2 0.5 \
    --epsilon 1e-6 \
    --weight_decay 0 \
    --weight_init random \
    --num_layers 1 \
    --hidden_size 4 \
    --activation sigmoid
```

## Running a Sweep using WandB

  ```bash
  python train_better_accuracy_test_cross_entropy.py --sweep \
    --wandb_project Your_Project_Name \
    --wandb_entity Your_Entity_Name
```

Or for the MSE version:

```bash
python train_better_accuracy_test_mse.py --sweep \
    --wandb_project Your_Project_Name \
    --wandb_entity Your_Entity_Name
```

## Customization
The following command-line options allow you to customize the training process:

| Option            | Description |
|-------------------|-------------|
| `-h`, `--help`     | Display help message and exit |
| `-wp`, `--wandb_project` | WandB project name |
| `-we`, `--wandb_entity`  | WandB entity name |
| `-d`, `--dataset`        | Dataset to use (`fashion_mnist` or `mnist`) |
| `-e`, `--epochs`         | Number of epochs |
| `-b`, `--batch_size`     | Batch size |
| `-l`, `--loss`           | Loss function (`cross_entropy` or `mean_squared_error`) |
| `-o`, `--optimizer`      | Optimizer to use (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `-lr`, `--learning_rate` | Learning rate |
| `-m`, `--momentum`       | Momentum value for Momentum and NAG optimizers |
| `--beta`                 | Beta value for RMSProp |
| `--beta1`                | Beta1 for Adam/Nadam |
| `--beta2`                | Beta2 for Adam/Nadam |
| `-eps`, `--epsilon`      | Epsilon value for Adam/Nadam |
| `-w_d`, `--weight_decay` | Weight decay |
| `-w_i`, `--weight_init`  | Weight initialization method (`random` or `xavier`) |
| `-nhl`, `--num_layers`   | Number of hidden layers |
| `-sz`, `--hidden_size`   | Size (number of neurons) for each hidden layer |
| `-a`, `--activation`     | Activation function (`sigmoid`, `tanh`, `relu`, `identity`) |
| `--sweep`                | Run a hyperparameter sweep instead of a single training run |

  














