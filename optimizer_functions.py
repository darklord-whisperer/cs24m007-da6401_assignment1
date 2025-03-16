import numpy as np

def sgd_update(w, b, dw, db, lr, wd):
    for i in range(len(w)):
        w[i] -= lr * (dw[i] + wd * w[i])
        b[i] -= lr * db[i]
    return w, b

def momentum_update(w, b, dw, db, vw, vb, lr, mom, wd):
    for i in range(len(w)):
        vw[i] = mom * vw[i] + lr * (dw[i] + wd * w[i])
        vb[i] = mom * vb[i] + lr * db[i]
        w[i] -= vw[i]
        b[i] -= vb[i]
    return w, b, vw, vb

def nag_update(w, b, dw, db, vw, vb, lr, mom, wd):
    for i in range(len(w)):
        prev_vw = vw[i].copy()
        prev_vb = vb[i].copy()
        vw[i] = mom * vw[i] + lr * (dw[i] + wd * w[i])
        vb[i] = mom * vb[i] + lr * db[i]
        w[i] -= (mom * prev_vw + vw[i])
        b[i] -= (mom * prev_vb + vb[i])
    return w, b, vw, vb

def rmsprop_update(w, b, dw, db, vw, vb, lr, beta, eps, wd):
    for i in range(len(w)):
        vw[i] = beta * vw[i] + (1-beta) * (dw[i]**2)
        vb[i] = beta * vb[i] + (1-beta) * (db[i]**2)
        w[i] -= lr * (dw[i] + wd * w[i]) / (np.sqrt(vw[i]) + eps)
        b[i] -= lr * db[i] / (np.sqrt(vb[i]) + eps)
    return w, b, vw, vb

def adam_update(w, b, dw, db, mw, mb, vw, vb, lr, b1, b2, eps, t, wd):
    for i in range(len(w)):
        mw[i] = b1 * mw[i] + (1 - b1) * (dw[i] + wd * w[i])
        mb[i] = b1 * mb[i] + (1 - b1) * db[i]
        vw[i] = b2 * vw[i] + (1 - b2) * (dw[i] + wd * w[i])**2
        vb[i] = b2 * vb[i] + (1 - b2) * (db[i]**2)
        mw_hat = mw[i] / (1 - b1**t)
        mb_hat = mb[i] / (1 - b1**t)
        vw_hat = vw[i] / (1 - b2**t)
        vb_hat = vb[i] / (1 - b2**t)
        w[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
        b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    return w, b, mw, mb, vw, vb

def nadam_update(w, b, dw, db, mw, mb, vw, vb, lr, b1, b2, eps, t, wd):
    for i in range(len(w)):
        mw[i] = b1 * mw[i] + (1 - b1) * (dw[i] + wd * w[i])
        mb[i] = b1 * mb[i] + (1 - b1) * db[i]
        vw[i] = b2 * vw[i] + (1 - b2) * (dw[i] + wd * w[i])**2
        vb[i] = b2 * vb[i] + (1 - b2) * (db[i]**2)
        mw_hat = mw[i] / (1 - b1**t)
        mb_hat = mb[i] / (1 - b1**t)
        vw_hat = vw[i] / (1 - b2**t)
        vb_hat = vb[i] / (1 - b2**t)
        w[i] -= lr * (b1 * mw_hat + (1 - b1)/(1 - b1**t)*(dw[i] + wd * w[i])) / (np.sqrt(vw_hat) + eps)
        b[i] -= lr * (b1 * mb_hat + (1 - b1)/(1 - b1**t)*db[i]) / (np.sqrt(vb_hat) + eps)
    return w, b, mw, mb, vw, vb

# --- Optimizer Wrappers ---
# Each wrapper must adhere to the interface:
#    (w, b, state) = wrapper(w, b, grads_w, grads_b, state, args)

def sgd_wrapper(w, b, grads_w, grads_b, state, args):
    w, b = sgd_update(w, b, grads_w, grads_b, args.learning_rate, args.weight_decay)
    return w, b, state

def momentum_wrapper(w, b, grads_w, grads_b, state, args):
    if "vw" not in state:
        state["vw"] = [np.zeros_like(wi) for wi in w]
        state["vb"] = [np.zeros_like(bi) for bi in b]
    w, b, state["vw"], state["vb"] = momentum_update(w, b, grads_w, grads_b, state["vw"], state["vb"], args.learning_rate, args.momentum, args.weight_decay)
    return w, b, state

def nag_wrapper(w, b, grads_w, grads_b, state, args):
    if "vw" not in state:
        state["vw"] = [np.zeros_like(wi) for wi in w]
        state["vb"] = [np.zeros_like(bi) for bi in b]
    w, b, state["vw"], state["vb"] = nag_update(w, b, grads_w, grads_b, state["vw"], state["vb"], args.learning_rate, args.momentum, args.weight_decay)
    return w, b, state

def rmsprop_wrapper(w, b, grads_w, grads_b, state, args):
    if "vw" not in state:
        state["vw"] = [np.zeros_like(wi) for wi in w]
        state["vb"] = [np.zeros_like(bi) for bi in b]
    w, b, state["vw"], state["vb"] = rmsprop_update(w, b, grads_w, grads_b, state["vw"], state["vb"], args.learning_rate, args.beta, args.epsilon, args.weight_decay)
    return w, b, state

def adam_wrapper(w, b, grads_w, grads_b, state, args):
    if "mw" not in state:
        state["mw"] = [np.zeros_like(wi) for wi in w]
        state["mb"] = [np.zeros_like(bi) for bi in b]
        state["vw"] = [np.zeros_like(wi) for wi in w]
        state["vb"] = [np.zeros_like(bi) for bi in b]
    state["t"] = state.get("t", 0) + 1
    w, b, state["mw"], state["mb"], state["vw"], state["vb"] = adam_update(
        w, b, grads_w, grads_b, state["mw"], state["mb"], state["vw"], state["vb"],
        args.learning_rate, args.beta1, args.beta2, args.epsilon, state["t"], args.weight_decay)
    return w, b, state

def nadam_wrapper(w, b, grads_w, grads_b, state, args):
    if "mw" not in state:
        state["mw"] = [np.zeros_like(wi) for wi in w]
        state["mb"] = [np.zeros_like(bi) for bi in b]
        state["vw"] = [np.zeros_like(wi) for wi in w]
        state["vb"] = [np.zeros_like(bi) for bi in b]
    state["t"] = state.get("t", 0) + 1
    w, b, state["mw"], state["mb"], state["vw"], state["vb"] = nadam_update(
        w, b, grads_w, grads_b, state["mw"], state["mb"], state["vw"], state["vb"],
        args.learning_rate, args.beta1, args.beta2, args.epsilon, state["t"], args.weight_decay)
    return w, b, state

# Global dictionary mapping optimizer names to their wrappers.
OPTIMIZERS = {
    "sgd": sgd_wrapper,
    "momentum": momentum_wrapper,
    "nag": nag_wrapper,
    "rmsprop": rmsprop_wrapper,
    "adam": adam_wrapper,
    "nadam": nadam_wrapper
}

def customized_user_optimizer(name, func):
    """
    Registers a new optimizer wrapper function.
    
    Parameters:
      - name: (str) the name of the optimizer
      - func: a function with the unified interface: 
              (w, b, state) = func(w, b, grads_w, grads_b, state, args)
    """
    OPTIMIZERS[name.lower()] = func

def update_weights(optimizer, w, b, grads_w, grads_b, state, args):
    opt_name = optimizer.lower()
    opt_func = OPTIMIZERS.get(opt_name, sgd_wrapper)
    w, b, state = opt_func(w, b, grads_w, grads_b, state, args)
    return w, b, state
