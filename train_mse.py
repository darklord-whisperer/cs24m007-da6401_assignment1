#!/usr/bin/env python3
"""
Usage Examples:
--------------
# To run a single training run:
python train.py --wandb_project Alik_Final_CS6190_DeepLearing_Assignment1 \
                --wandb_entity cs24m007-iit-madras \
                --dataset fashion_mnist \
                --epochs 5 \
                --batch_size 32 \
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
                --num_layers 3 \
                --hidden_size 64 \
                --activation relu

# To run a sweep (multiple hyperparameter configurations):
python train.py --sweep --wandb_project Alik_Final_CS6190_DeepLearing_Assignment1 \
                --wandb_entity cs24m007-iit-madras
"""

import argparse
import sys
import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist

# -------------------------------
# 1. Argument Parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a FFNN with various optimizers & W&B logging.")
    
    # WandB project/entity
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname",
                        help="Project name for Weights & Biases.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname",
                        help="Entity (username/team) for Weights & Biases.")
    
    # Dataset choice
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        choices=["fashion_mnist", "mnist"],
                        help="Dataset to train on.")
    
    # Training hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=4,
                        help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function.")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer name.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,
                        help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.5,
                        help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5,
                        help="Beta used by RMSProp optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                        help="Beta1 used by Adam/Nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5,
                        help="Beta2 used by Adam/Nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001,
                        help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0,
                        help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default="random",
                        choices=["random", "xavier"],
                        help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1,
                        help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, default="sigmoid",
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help="Activation function for hidden layers.")
    
    # New flag to run sweep or a single run.
    parser.add_argument("--sweep", action="store_true",
                        help="If provided, run a hyperparameter sweep; otherwise run a single training run.")
    
    args = parser.parse_args()
    return args

# -------------------------------
# 2. Data Loading and Splitting
# -------------------------------
def load_data(dataset):
    if dataset == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    n = len(X_train_full)
    split_idx = int(0.9 * n)
    X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
    y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# -------------------------------
# 3. Initialization of Weights
# -------------------------------
def init_weights(n_inputs, n_hidden_layers, hidden_size, n_outputs, init_mode="random"):
    w = []
    b = []
    
    def xavier_init(fan_in, fan_out):
        return np.random.normal(0.0, np.sqrt(2/(fan_in+fan_out)), (fan_out, fan_in))
    
    layer_sizes = [n_inputs] + [hidden_size]*n_hidden_layers + [n_outputs]
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
        if init_mode.lower() == "xavier":
            w_i = xavier_init(fan_in, fan_out)
        else:
            w_i = np.random.randn(fan_out, fan_in)
        b_i = np.zeros(fan_out)
        w.append(w_i)
        b.append(b_i)
    return w, b

# -------------------------------
# 4. Activation Functions
# -------------------------------
def activation_forward(z, activation):
    if activation.lower() == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -709.78, 709.78)))
    elif activation.lower() == "tanh":
        return np.tanh(z)
    elif activation.lower() in ["relu", "reLU"]:
        return np.maximum(0, z)
    elif activation.lower() == "identity":
        return z
    else:
        return z

def activation_backward(z, activation):
    if activation.lower() == "sigmoid":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -709.78, 709.78)))
        return sig * (1 - sig)
    elif activation.lower() == "tanh":
        return 1.0 - np.tanh(z)**2
    elif activation.lower() in ["relu", "reLU"]:
        return (z > 0).astype(float)
    elif activation.lower() == "identity":
        return np.ones_like(z)
    else:
        return np.ones_like(z)

def softmax(z):
    z_shift = z - np.max(z)
    exps = np.exp(z_shift)
    return exps / np.sum(exps)

# -------------------------------
# 5. Forward and Backward Passes
# -------------------------------
def forward_pass(x, w, b, activation):
    hs = [x]  # input layer activation
    zs = []
    for i in range(len(w) - 1):
        z = np.dot(w[i], hs[-1]) + b[i]
        zs.append(z)
        hs.append(activation_forward(z, activation))
    z_out = np.dot(w[-1], hs[-1]) + b[-1]
    zs.append(z_out)
    out = softmax(z_out)
    return zs, hs, out

def backward_pass(x, y, zs, hs, w, b, activation, loss="cross_entropy"):
    num_classes = w[-1].shape[0]
    y_onehot = np.zeros(num_classes)
    y_onehot[y] = 1.0
    grads_w = [None] * len(w)
    grads_b = [None] * len(b)
    # Output layer
    a_out = softmax(zs[-1])
    if loss == "cross_entropy":
        delta = a_out - y_onehot
    else:  # MSE
        delta = 2 * (a_out - y_onehot) * a_out * (1 - a_out)
    grads_w[-1] = np.outer(delta, hs[-1])
    grads_b[-1] = delta
    # Backpropagate through hidden layers
    for i in reversed(range(len(w) - 1)):
        z = zs[i]
        dact = activation_backward(z, activation)
        delta = np.dot(w[i+1].T, delta) * dact
        grads_w[i] = np.outer(delta, hs[i])
        grads_b[i] = delta
    return grads_w, grads_b

# -------------------------------
# 6. Optimizer Update Functions with Stability Fixes
# -------------------------------
def sgd_update(w, b, dw, db, lr, weight_decay):
    for i in range(len(w)):
        w[i] -= lr * (dw[i] + weight_decay * w[i])
        b[i] -= lr * db[i]
    return w, b

def momentum_update(w, b, dw, db, vw, vb, lr, momentum, weight_decay):
    for i in range(len(w)):
        vw[i] = momentum * vw[i] + lr * (dw[i] + weight_decay * w[i])
        vb[i] = momentum * vb[i] + lr * db[i]
        w[i] -= vw[i]
        b[i] -= vb[i]
    return w, b, vw, vb

def nag_update(w, b, dw, db, vw, vb, lr, momentum, weight_decay):
    for i in range(len(w)):
        prev_vw = vw[i].copy()
        prev_vb = vb[i].copy()
        vw[i] = momentum * vw[i] + lr * (dw[i] + weight_decay * w[i])
        vb[i] = momentum * vb[i] + lr * db[i]
        w[i] -= (momentum * prev_vw + vw[i])
        b[i] -= (momentum * prev_vb + vb[i])
    return w, b, vw, vb

def rmsprop_update(w, b, dw, db, vw, vb, lr, beta, epsilon, weight_decay):
    for i in range(len(w)):
        vw[i] = beta * vw[i] + (1 - beta) * (dw[i]**2)
        vb[i] = beta * vb[i] + (1 - beta) * (db[i]**2)
        denom_w = np.maximum(np.sqrt(vw[i]), epsilon)
        denom_b = np.maximum(np.sqrt(vb[i]), epsilon)
        w[i] -= lr * (dw[i] + weight_decay * w[i]) / denom_w
        b[i] -= lr * db[i] / denom_b
    return w, b, vw, vb

def adam_update(w, b, dw, db, mw, mb, vw, vb, lr, beta1, beta2, epsilon, t, weight_decay):
    for i in range(len(w)):
        mw[i] = beta1 * mw[i] + (1 - beta1) * (dw[i] + weight_decay * w[i])
        mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
        vw[i] = beta2 * vw[i] + (1 - beta2) * (dw[i] + weight_decay * w[i])**2
        vb[i] = beta2 * vb[i] + (1 - beta2) * (db[i]**2)
        mw_hat = mw[i] / (1 - beta1**t)
        mb_hat = mb[i] / (1 - beta1**t)
        vw_hat = vw[i] / (1 - beta2**t)
        vb_hat = vb[i] / (1 - beta2**t)
        denom_w = np.maximum(np.sqrt(vw_hat), epsilon)
        denom_b = np.maximum(np.sqrt(vb_hat), epsilon)
        w[i] -= lr * mw_hat / denom_w
        b[i] -= lr * mb_hat / denom_b
    return w, b, mw, mb, vw, vb

def nadam_update(w, b, dw, db, mw, mb, vw, vb, lr, beta1, beta2, epsilon, t, weight_decay):
    for i in range(len(w)):
        mw[i] = beta1 * mw[i] + (1 - beta1) * (dw[i] + weight_decay * w[i])
        mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
        vw[i] = beta2 * vw[i] + (1 - beta2) * (dw[i] + weight_decay * w[i])**2
        vb[i] = beta2 * vb[i] + (1 - beta2) * (db[i]**2)
        mw_hat = mw[i] / (1 - beta1**t)
        mb_hat = mb[i] / (1 - beta1**t)
        vw_hat = vw[i] / (1 - beta2**t)
        vb_hat = vb[i] / (1 - beta2**t)
        denom_w = np.maximum(np.sqrt(vw_hat), epsilon)
        denom_b = np.maximum(np.sqrt(vb_hat), epsilon)
        w[i] -= lr * (beta1 * mw_hat + (1 - beta1)/(1 - beta1**t)*(dw[i] + weight_decay * w[i])) / denom_w
        b[i] -= lr * (beta1 * mb_hat + (1 - beta1)/(1 - beta1**t)*db[i]) / denom_b
    return w, b, mw, mb, vw, vb

# -------------------------------
# 7. Training and Evaluation Functions
# -------------------------------
def train_one_epoch(X_train, y_train, w, b, args, state):
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    batch_size = args.batch_size
    if "t" not in state:
        state["t"] = 0
    total_loss = 0.0
    for start_idx in range(0, len(X_train), batch_size):
        end_idx = min(start_idx + batch_size, len(X_train))
        batch_idx = idx[start_idx:end_idx]
        grads_w = [np.zeros_like(wi) for wi in w]
        grads_b = [np.zeros_like(bi) for bi in b]
        for i in batch_idx:
            x = X_train[i].flatten()
            y = y_train[i]
            zs, hs, out = forward_pass(x, w, b, args.activation)
            if args.loss == "cross_entropy":
                total_loss += -np.log(out[y] + 1e-10)
            else:
                onehot = np.zeros(len(out))
                onehot[y] = 1.0
                total_loss += 0.5 * np.sum((out - onehot)**2)
            dW, dB = backward_pass(x, y, zs, hs, w, b, args.activation, args.loss)
            for j in range(len(w)):
                grads_w[j] += dW[j]
                grads_b[j] += dB[j]
        for j in range(len(w)):
            grads_w[j] /= len(batch_idx)
            grads_b[j] /= len(batch_idx)
        opt = args.optimizer.lower()
        lr = args.learning_rate
        wd = args.weight_decay
        if opt == "sgd":
            w, b = sgd_update(w, b, grads_w, grads_b, lr, wd)
        elif opt == "momentum":
            if "vw" not in state:
                state["vw"] = [np.zeros_like(wi) for wi in w]
                state["vb"] = [np.zeros_like(bi) for bi in b]
            w, b, state["vw"], state["vb"] = momentum_update(w, b, grads_w, grads_b,
                                                             state["vw"], state["vb"], lr, args.momentum, wd)
        elif opt == "nag":
            if "vw" not in state:
                state["vw"] = [np.zeros_like(wi) for wi in w]
                state["vb"] = [np.zeros_like(bi) for bi in b]
            w, b, state["vw"], state["vb"] = nag_update(w, b, grads_w, grads_b,
                                                        state["vw"], state["vb"], lr, args.momentum, wd)
        elif opt == "rmsprop":
            if "vw" not in state:
                state["vw"] = [np.zeros_like(wi) for wi in w]
                state["vb"] = [np.zeros_like(bi) for bi in b]
            w, b, state["vw"], state["vb"] = rmsprop_update(w, b, grads_w, grads_b,
                                                           state["vw"], state["vb"], lr, args.beta, args.epsilon, wd)
        elif opt == "adam":
            if "mw" not in state:
                state["mw"] = [np.zeros_like(wi) for wi in w]
                state["mb"] = [np.zeros_like(bi) for bi in b]
                state["vw"] = [np.zeros_like(wi) for wi in w]
                state["vb"] = [np.zeros_like(bi) for bi in b]
            state["t"] += 1
            w, b, state["mw"], state["mb"], state["vw"], state["vb"] = adam_update(
                w, b, grads_w, grads_b, state["mw"], state["mb"], state["vw"], state["vb"],
                lr, args.beta1, args.beta2, args.epsilon, state["t"], wd)
        elif opt == "nadam":
            if "mw" not in state:
                state["mw"] = [np.zeros_like(wi) for wi in w]
                state["mb"] = [np.zeros_like(bi) for bi in b]
                state["vw"] = [np.zeros_like(wi) for wi in w]
                state["vb"] = [np.zeros_like(bi) for bi in b]
            state["t"] += 1
            w, b, state["mw"], state["mb"], state["vw"], state["vb"] = nadam_update(
                w, b, grads_w, grads_b, state["mw"], state["mb"], state["vw"], state["vb"],
                lr, args.beta1, args.beta2, args.epsilon, state["t"], wd)
        else:
            w, b = sgd_update(w, b, grads_w, grads_b, lr, wd)
    avg_loss = total_loss / len(X_train)
    return w, b, avg_loss

def evaluate_accuracy(X, y, w, b, activation):
    correct = 0
    for i in range(len(X)):
        x = X[i].flatten()
        _, _, out = forward_pass(x, w, b, activation)
        if np.argmax(out) == y[i]:
            correct += 1
    return correct / len(X)

def confusion_matrix_plot(X, y, w, b, activation, class_names):
    preds = []
    for i in range(len(X)):
        x = X[i].flatten()
        _, _, out = forward_pass(x, w, b, activation)
        preds.append(np.argmax(out))
    cm = wandb.plot.confusion_matrix(probs=None, y_true=y, preds=preds, class_names=class_names)
    return cm

# -------------------------------
# 8. Main Training Function (Single Run)
# -------------------------------
def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    # Set a meaningful run name as specified
    run_name = f"e_{config.epochs}_hl_{config.num_layers}_hs_{config.hidden_size}_lr_{config.learning_rate}_opt_{config.optimizer}_bs_{config.batch_size}_init_{config.weight_init}_ac_{config.activation}_wd_{config.weight_decay}"
    wandb.run.name = run_name

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(config.dataset)
    n_inputs = X_train.shape[1] * X_train.shape[2]
    n_outputs = 10  # 10 classes for MNIST/Fashion-MNIST
    w, b = init_weights(n_inputs, config.num_layers, config.hidden_size, n_outputs, init_mode=config.weight_init)
    state = {}
    for epoch in range(config.epochs):
        w, b, train_loss = train_one_epoch(X_train, y_train, w, b, config, state)
        train_acc = evaluate_accuracy(X_train, y_train, w, b, config.activation)
        val_acc = evaluate_accuracy(X_val, y_val, w, b, config.activation)
        test_acc = evaluate_accuracy(X_test, y_test, w, b, config.activation)
        # Calculate validation loss
        val_loss_sum = 0.0
        for i in range(len(X_val)):
            x = X_val[i].flatten()
            y = y_val[i]
            zs, hs, out = forward_pass(x, w, b, config.activation)
            if config.loss == "cross_entropy":
                val_loss_sum += -np.log(out[y] + 1e-10)
            else:
                onehot = np.zeros(n_outputs)
                onehot[y] = 1.0
                val_loss_sum += 0.5 * np.sum((out - onehot)**2)
        val_loss = val_loss_sum / len(X_val)
        # Print epoch details and flush output
        print(f"Epoch {epoch+1}/{config.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}", flush=True)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc
        })
    final_test_acc = evaluate_accuracy(X_test, y_test, w, b, config.activation)
    class_names = [str(i) for i in range(n_outputs)]
    cm_plot = confusion_matrix_plot(X_test, y_test, w, b, config.activation, class_names)
    wandb.log({"confusion_matrix": cm_plot, "final_test_accuracy": final_test_acc})
    print(f"Final Test Accuracy = {final_test_acc*100:.2f}%", flush=True)
    wandb.finish()

# -------------------------------
# 9. Sweep Configuration
# -------------------------------
sweep_config = {
    "name": "my_sweep",
    "method": "random",
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
        "weight_decay": {"values": [0.0, 0.0005, 0.5]}
    }
}

# -------------------------------
# 10. Main Entry Point: Single Run vs. Sweep
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        print("Sweep ID:", sweep_id, flush=True)
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()
