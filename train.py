#!/usr/bin/env python3
"""
Vectorized FFNN training script.

Maintains:
 • Same dataset splits (80/10/10)
 • Same epochs and sweep configuration
 • Batch-wise forward/backward propagation for efficiency
 • Row-wise percentage and absolute counts in the creative confusion matrix

Hyperparameters remain unchanged.
"""

import argparse
import sys
import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from optimizer_functions import update_weights  # Importing the optimizer functions

# =========================================================
# 1. Argument Parsing
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FeedForward Neural Network with backpropagation from scratch")
    parser.add_argument("-wp", "--wandb_project", type=str, default="Alik_Final_DA6401_DeepLearning_Assignment1",
                        help="WandB project name.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m007-iit-madras",
                        help="WandB entity (username/team).")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        choices=["fashion_mnist", "mnist"],
                        help="Dataset to train on.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function.")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer type.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Initial learning rate.")
    parser.add_argument("-m", "--momentum", type=float, default=0.9,
                        help="Momentum factor.")
    parser.add_argument("-beta", "--beta", type=float, default=0.9,
                        help="Beta for RMSProp.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                        help="Beta1 for Adam/Nadam.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                        help="Beta2 for Adam/Nadam.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8,
                        help="Epsilon for numerical stability.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0,
                        help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default="Xavier",
                        choices=["random", "Xavier"],
                        help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5,
                        help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, default="ReLU",
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help="Activation function.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run hyperparameter sweep.")    # Sweep argument for manual sweeps
    return parser.parse_args()


# =========================================================
# 2. Data Loading and Splitting (80/10/10)
# =========================================================
def load_data(dataset):
    if dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    X_all = X_all / 255.0

    total = len(X_all)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)
    return (X_all[:train_end], y_all[:train_end]), (X_all[train_end:val_end], y_all[train_end:val_end]), (X_all[val_end:], y_all[val_end:])

# =========================================================
# 3. Weight Initialization
# =========================================================
def xavier_init(fan_in, fan_out):
    return np.random.normal(0.0, np.sqrt(2/(fan_in+fan_out)), (fan_out, fan_in))

def init_weights(n_inputs, n_hidden_layers, hidden_size, n_outputs, init_mode="random"):
    weights, biases = [], []
    dims = [n_inputs] + [hidden_size]*n_hidden_layers + [n_outputs]
    for i in range(len(dims) - 1):
        if init_mode.lower() == "xavier":
            w = xavier_init(dims[i], dims[i+1])
        else:
            w = np.random.randn(dims[i+1], dims[i])
        b = np.zeros(dims[i+1])
        weights.append(w)
        biases.append(b)
    return weights, biases

# =========================================================
# 4. Activation Functions and Softmax
# =========================================================
def activation_forward(z, activation):
    act = activation.lower()
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -709.78, 709.78)))
    elif act == "tanh":
        return np.tanh(z)
    elif act == "relu":
        return np.maximum(0, z)
    elif act == "identity":
        return z
    return z

def activation_backward(z, activation):
    act = activation.lower()
    if act == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -709.78, 709.78)))
        return s * (1 - s)
    elif act == "tanh":
        return 1.0 - np.tanh(z)**2
    elif act == "relu":
        return (z > 0).astype(float)
    elif act == "identity":
        return np.ones_like(z)
    return np.ones_like(z)

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# =========================================================
# 5. Forward Pass (Vectorized over Batch)
# =========================================================
def forward_pass(X_batch, weights, biases, activation):
    h_vals = [X_batch]
    z_vals = []
    current = X_batch
    for i in range(len(weights) - 1):
        z = current @ weights[i].T + biases[i]
        z_vals.append(z)
        current = activation_forward(z, activation)
        h_vals.append(current)
    z_out = current @ weights[-1].T + biases[-1]
    z_vals.append(z_out)
    output = softmax(z_out)
    return z_vals, h_vals, output

# =========================================================
# 6. Backward Pass (Vectorized)
# =========================================================
def backward_pass(X_batch, y_batch, z_vals, h_vals, weights, biases, activation, loss="cross_entropy"):
    batch_size = X_batch.shape[0]
    num_classes = weights[-1].shape[0]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), y_batch] = 1.0

    grads_w = [None] * len(weights)
    grads_b = [None] * len(biases)

    a_out = softmax(z_vals[-1])
    if loss == "cross_entropy":
        delta = a_out - y_onehot
    else:
        delta = 2 * (a_out - y_onehot) * a_out * (1 - a_out)

    grads_w[-1] = (delta.T @ h_vals[-1]) / batch_size
    grads_b[-1] = np.mean(delta, axis=0)

    for i in reversed(range(len(weights) - 1)):
        d_act = activation_backward(z_vals[i], activation)
        delta = (delta @ weights[i+1]) * d_act
        grads_w[i] = (delta.T @ h_vals[i]) / batch_size
        grads_b[i] = np.mean(delta, axis=0)
    return grads_w, grads_b

# =========================================================
# 7. Training and Evaluation (Vectorized)
# =========================================================
def train_one_epoch(X_train, y_train, w, b, args, state):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    total_loss = 0.0
    batch_size = args.batch_size

    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        batch_idx = indices[start:end]
        X_batch = X_train[batch_idx].reshape(-1, w[0].shape[1])
        y_batch = y_train[batch_idx]
        
        z_vals, h_vals, out = forward_pass(X_batch, w, b, args.activation)
        
        if args.loss == "cross_entropy":
            loss_batch = -np.sum(np.log(out[np.arange(len(y_batch)), y_batch] + 1e-10))
        else:
            num_classes = w[-1].shape[0]
            y_oh = np.zeros((len(y_batch), num_classes))
            y_oh[np.arange(len(y_batch)), y_batch] = 1.0
            loss_batch = 0.5 * np.sum((out - y_oh)**2)
        total_loss += loss_batch

        grads_w, grads_b = backward_pass(X_batch, y_batch, z_vals, h_vals, w, b, args.activation, args.loss)
        
        clip_thresh = 50.0
        total_norm = sum(np.sum(g**2) for g in grads_w) + sum(np.sum(g**2) for g in grads_b)
        total_norm = np.sqrt(total_norm)
        if total_norm > clip_thresh:
            scale = clip_thresh / total_norm
            grads_w = [g * scale for g in grads_w]
            grads_b = [g * scale for g in grads_b]
        
        w, b, state = update_weights(args.optimizer, w, b, grads_w, grads_b, state, args)

    avg_loss = total_loss / len(X_train)
    return w, b, avg_loss

def evaluate_accuracy(X, y, w, b, activation):
    batch = 256
    correct = 0
    for start in range(0, len(X), batch):
        end = min(start + batch, len(X))
        X_batch = X[start:end].reshape(-1, w[0].shape[1])
        _, _, out = forward_pass(X_batch, w, b, activation)
        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == y[start:end])
    return correct / len(X)

# =========================================================
# 8. Confusion Matrix Functions
# =========================================================
def build_confusion_matrix(X, y, w, b, activation, class_names=None):
    """
    An interactive Plotly confusion matrix with a light color scale 
    for improved text visibility.
    """
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    preds = []
    batch_size = 256
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = X[start:end].reshape(-1, w[0].shape[1])
        _, _, out = forward_pass(X_batch, w, b, activation)
        preds.extend(np.argmax(out, axis=1))
    
    cm = confusion_matrix(y, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_percent = (cm / row_sums) * 100.0
    text_labels = [[f"{cm_percent[i, j]:.1f}%" for j in range(len(class_names))] for i in range(len(class_names))]

    teal_colorscale = [
        [0.0, "rgb(229, 244, 245)"],
        [0.2, "rgb(180, 226, 228)"],
        [0.4, "rgb(135, 206, 206)"],
        [0.6, "rgb(90, 180, 180)"],
        [0.8, "rgb(40, 150, 150)"],
        [1.0, "rgb(0, 120, 120)"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_percent,
            x=class_names,
            y=class_names,
            customdata=cm,
            text=text_labels,
            texttemplate="%{text}",
            textfont={"color": "black"},
            colorscale=teal_colorscale,
            zmin=0,
            zmax=100,
            hovertemplate=("True: %{y}<br>Predicted: %{x}<br>Count: %{customdata}<br>Percentage: %{z:.2f}%%<extra></extra>"),
            colorbar=dict(title="Percentage"),
        )
    )
    fig.update_layout(
        title="Interactive Confusion Matrix (Fashion MNIST)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(tickmode="array", tickvals=list(range(len(class_names))), ticktext=class_names, autorange="reversed"),
        xaxis=dict(tickmode="array", tickvals=list(range(len(class_names))), ticktext=class_names),
        width=200,
        height=200,
    )
    return fig

def build_normal_confusion_matrix(X, y, w, b, activation, class_names=None):
    """
    For a static (normal) confusion matrix using Matplotlib and returns a confusion matrix figure.
    """
    if class_names is None:
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    preds = []
    batch_size = 256
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = X[start:end].reshape(-1, w[0].shape[1])
        _, _, out = forward_pass(X_batch, w, b, activation)
        preds.extend(np.argmax(out, axis=1))
    cm = confusion_matrix(y, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha="center", va="center", color="red")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normal Confusion Matrix")
    fig.tight_layout()
    return fig

# =========================================================
# 9. Main Training Function
# =========================================================
def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    run_tag = f"hl_{config.num_layers}_opt_{config.optimizer}_bs_{config.batch_size}_ac_{config.activation}"
    wandb.run.name = run_tag

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(config.dataset)
    n_inputs = X_train.shape[1] * X_train.shape[2]
    n_outputs = 10

    w, b = init_weights(n_inputs, config.num_layers, config.hidden_size, n_outputs, init_mode=config.weight_init)
    state = {}

    for epoch in range(config.epochs):
        w, b, train_loss = train_one_epoch(X_train, y_train, w, b, config, state)
        train_acc = evaluate_accuracy(X_train, y_train, w, b, config.activation)
        val_acc = evaluate_accuracy(X_val, y_val, w, b, config.activation)
        test_acc = evaluate_accuracy(X_test, y_test, w, b, config.activation)

        batch = 256
        total_val_loss = 0.0
        for start in range(0, len(X_val), batch):
            end = min(start + batch, len(X_val))
            X_batch = X_val[start:end].reshape(-1, n_inputs)
            y_batch = y_val[start:end]
            z_vals, h_vals, out = forward_pass(X_batch, w, b, config.activation)
            if config.loss == "cross_entropy":
                total_val_loss += -np.sum(np.log(out[np.arange(len(y_batch)), y_batch] + 1e-10))
            else:
                oh = np.zeros_like(out)
                oh[np.arange(len(y_batch)), y_batch] = 1.0
                total_val_loss += 0.5 * np.sum((out - oh)**2)
        val_loss = total_val_loss / len(X_val)

        print(f"Epoch {epoch+1}/{config.epochs}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}", flush=True)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc
        })

        new_lr = max(1e-5, config.learning_rate * 0.995)
        config.update({"learning_rate": new_lr}, allow_val_change=True)

    final_test_acc = evaluate_accuracy(X_test, y_test, w, b, config.activation)
    class_names = [str(i) for i in range(n_outputs)]
    creative_cm = build_confusion_matrix(X_test, y_test, w, b, config.activation, class_names)
    normal_cm_fig = build_normal_confusion_matrix(X_test, y_test, w, b, config.activation, class_names)

    wandb.log({
        "creative_confusion_matrix": creative_cm,
        "normal_confusion_matrix": wandb.Image(normal_cm_fig),
        "final_test_accuracy": final_test_acc
    })

    print(f"Final Test Accuracy = {final_test_acc*100:.2f}%", flush=True)
    wandb.finish()

# =========================================================
# Sweep Configuration
# =========================================================
sweep_config = {
    "name": "CS24M007_cross_entropy_fashion_mnist",
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
        "loss": {"values": ["cross_entropy"]},
        "weight_decay": {"values": [0.0, 0.5, 0.0005]}
    }
}

# =========================================================
# Main Entry Point
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        print("Sweep ID:", sweep_id, flush=True)
        wandb.agent(sweep_id, function=main)
    else:
        main()
