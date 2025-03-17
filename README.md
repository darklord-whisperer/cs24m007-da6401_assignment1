<div style="text-align: justify;">

# DA6401 : Fundamentals of Deep Learning - Assignment 1

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

- **Github Repolink:** https://github.com/darklord-whisperer/cs24m007-da6401_assignment1/
- **WandB Project:** https://wandb.ai/cs24m007-iit-madras/Alik_Final_DA6401_DeepLearning_Assignment1/
- **WandB Report:** https://wandb.ai/cs24m007-iit-madras/Alik_Final_DA6401_DeepLearning_Assignment1/reports/DA-6401-Assignment-1-by-CS24M007-Alik-Sarkar--VmlldzoxMTcxOTUzMA?accessToken=ghybipxdmdstcq2fi5wfi6tmmqubmweohe19zyqej9k6mkx3wu9a4slrraoc9twr

## Dataset

The assignment uses two popular datasets from the `keras.datasets` module:
- **Fashion MNIST:** Loaded using `from keras.datasets import fashion_mnist`
- **MNIST:** Loaded using `from keras.datasets import mnist`

The datasets are normalized (pixel values scaled to [0, 1]) and the training data is split into training, validation, and test sets (commonly an 80/10/10 split).

## Implementation Details

### Structure

The repository is organized into two main modules, each addressing a specific functionality:

**train.py**:
The central training script that manages:

  - Command-line argument parsing for hyperparameter configuration.
  - Data loading with an 80/10/10 train/validation/test split.
  - Initialization of network weights (using Xavier or random initialization).
  - Forward and backward passes with gradient clipping.
  - Dynamic optimizer updates (all optimizers are imported from optimizer_functions.py).
  - Logging of performance metrics and both standard and interactive confusion matrices to WandB.
  - Hyperparameter sweeps support via WandB

**optimizer_functions.py**:
  - Contains the implementation of various optimizer wrappers. All functions adhere to a unified interface, making it easy to add new optimization algorithms. The file also maintains a g lobal OPTIMIZERS dictionary for easy mapping between optimizer names and their implementations.

**Question_1.py:**  
  Contains the `FashionMNISTVisualizer` class which:
  - Loads the Fashion MNIST dataset.
  - Plots a grid of one sample image per class.
  - Logs the generated figure to WandB.

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
- **Cross Entropy(CE):** Ideal for classification tasks.
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

plotly

## Installation:

### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```


### Install the Dependencies
``` python 
pip install -r requirements.txt
```

## Usage
### Running Manually (Will run default hyperparameters which stores the best hyperparameters)

```python
python train.py --wandb_project Your_Project_Name --wandb_entity Your_Entity_Name
```
### Running Question 1:

```python
python Question_1.py
```

#### Cross-Entropy Loss Version:


## Running a Sweep using WandB

  ```python
  python train.py --sweep --wandb_project Your_Project_Name --wandb_entity Your_Entity_Name
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

  
```python
python train.py \
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
    --weight_init Xavier \
    --num_layers 2 \
    --hidden_size 128 \
    --activation ReLU
```

OR for example like this (P.S.: The parameters below are just random hyperparameters, best parameters are present in default itself):

```python
python train.py -e 10 -o nadam -a tanh -b 64 -nhl 3 -lr 0.001 -nhl 3 --w_d 0.0005 --w_i Xavier -sz 128 -beta 0.5 -beta1 0.5 -beta2 0.5 -eps 0.000001 -m 0.5 -w_i Xavier
```

## Customization for Optimization function

If you wish to add a new optimization algorithm, you can do so easily in optimizer_functions.py. A new function is already created for the user to add their own optimizer function. This is extremely flexible and easy to use – you only need to add your logic in Python and place it in the function body shown below:

```python
def customized_user_optimizer(name, func):
    """
    Registers a new optimizer wrapper function.
    
    Parameters:
      - name: (str) the name of the optimizer
      - func: a function with the unified interface: 
              (w, b, state) = func(w, b, grads_w, grads_b, state, args)
    """
    OPTIMIZERS[name.lower()] = func
```

All optimizer functions adhere to a unified interface, meaning they all take the same arguments. This approach avoids having different parameter sets or function signatures for each optimizer and minimizes changes needed in the training loop. The mapping complexities are inherently handled by the train.py script, which maintains a global OPTIMIZERS dictionary to keep track of all implemented optimizer functions.












