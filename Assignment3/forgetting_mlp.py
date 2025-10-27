#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS599 / IST597 – Catastrophic Forgetting on Permuted-MNIST
----------------------------------------------------------
TensorFlow 2 implementation with multi-layer perceptrons (MLPs)
trained sequentially across 10 permuted-MNIST tasks.
Measures performance via task matrix R, ACC, and BWT.

Author: Sheena Shaha
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def set_random_seed(seed_value: int = 42):
    """Ensures reproducible runs."""
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def create_dir(folder="results"):
    """Create results directory if not already present."""
    os.makedirs(folder, exist_ok=True)
    return folder

def to_one_hot(y, n_classes=10):
    """Convert integer labels to one-hot encoding."""
    return tf.one_hot(tf.cast(y, tf.int32), depth=n_classes)

# ---------------------------------------------------------------------
# Build permuted MNIST tasks
# ---------------------------------------------------------------------
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 784)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 784)).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def generate_permutations(num_tasks=10, seed=0):
    rng = np.random.default_rng(seed)
    base = np.arange(784)
    return [rng.permutation(base) for _ in range(num_tasks)]

def prepare_tasks(num_tasks=10, seed=0):
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    permutations = generate_permutations(num_tasks, seed)
    tasks = []
    for perm in permutations:
        tasks.append({
            "x_train": x_train[:, perm],
            "y_train": y_train,
            "x_test":  x_test[:, perm],
            "y_test":  y_test
        })
    return tasks

# ---------------------------------------------------------------------
# MLP model factory
# ---------------------------------------------------------------------
def make_mlp(depth=2, width=256, dropout=0.2, l2_val=0.0):
    """Construct MLP with given depth, width, and dropout."""
    reg = keras.regularizers.l2(l2_val) if l2_val > 0 else None
    inputs = keras.Input(shape=(784,))
    x = inputs
    for _ in range(depth):
        x = layers.Dense(width, activation="relu", kernel_regularizer=reg)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# ---------------------------------------------------------------------
# Loss & Optimizer
# ---------------------------------------------------------------------
def choose_loss(name="nll"):
    name = name.lower()
    if name == "nll":
        return keras.losses.SparseCategoricalCrossentropy()
    elif name == "l1":
        return lambda y_true, y_pred: tf.reduce_mean(tf.abs(to_one_hot(y_true) - y_pred))
    elif name == "l2":
        return lambda y_true, y_pred: tf.reduce_mean(tf.square(to_one_hot(y_true) - y_pred))
    elif name in ["l1+l2", "l1_l2"]:
        return lambda y_true, y_pred: tf.reduce_mean(tf.abs(to_one_hot(y_true) - y_pred)) + \
                                      tf.reduce_mean(tf.square(to_one_hot(y_true) - y_pred))
    else:
        raise ValueError("Unsupported loss type.")

def choose_optimizer(opt_name="adam", lr=None):
    opt_name = opt_name.lower()
    lr = lr or (1e-3 if opt_name != "sgd" else 0.01)
    if opt_name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    elif opt_name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError("Unknown optimizer.")

# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def test_accuracy(model, x, y):
    """Evaluate model accuracy on dataset."""
    result = model.evaluate(x, y, batch_size=256, verbose=0, return_dict=True)
    return float(result["accuracy"])

def test_all_tasks(model, tasks):
    """Evaluate accuracy on all T tasks."""
    return [test_accuracy(model, t["x_test"], t["y_test"]) for t in tasks]

# ---------------------------------------------------------------------
# Main continual training
# ---------------------------------------------------------------------
def continual_training(depth=2, width=256, dropout=0.2,
                       optimizer="adam", loss="nll", seed=0,
                       output_dir="results", l2_reg=0.0):
    set_random_seed(seed)
    create_dir(output_dir)
    num_tasks = 10
    tasks = prepare_tasks(num_tasks, seed)

    model = make_mlp(depth, width, dropout, l2_reg)
    model.compile(optimizer=choose_optimizer(optimizer),
                  loss=choose_loss(loss),
                  metrics=["accuracy"])

    R = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    val_history = {}

    for t in range(num_tasks):
        epochs = 50 if t == 0 else 20
        print(f"\n--- Training Task {t+1}/{num_tasks} for {epochs} epochs ---")
        history = model.fit(tasks[t]["x_train"], tasks[t]["y_train"],
                            validation_data=(tasks[t]["x_test"], tasks[t]["y_test"]),
                            epochs=epochs, batch_size=128, verbose=2)
        val_history[f"task_{t+1}"] = history.history.get("val_accuracy", [])

        accs = test_all_tasks(model, tasks)
        R[t, :] = accs

    # Metrics
    ACC = float(np.mean(R[-1, :]))
    BWT = float(np.mean(R[-1, :-1] - np.diag(R)[:-1]))

    # Save metrics
    tag = f"d{depth}_do{dropout}_opt{optimizer}_loss{loss}_seed{seed}"
    np.savetxt(os.path.join(output_dir, f"R_{tag}.csv"), R, delimiter=",", fmt="%.6f")
    with open(os.path.join(output_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump({"ACC": ACC, "BWT": BWT}, f, indent=2)

    # Save validation plots
    try:
        import matplotlib.pyplot as plt
        for task, curve in val_history.items():
            plt.figure()
            plt.plot(range(1, len(curve)+1), curve)
            plt.title(f"{task} Validation Accuracy ({tag})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{task}_{tag}.png"), dpi=150)
            plt.close()
    except Exception as e:
        print("Plotting skipped:", e)

    print("\n=== FINAL RESULTS ===")
    print(f"ACC: {ACC:.4f} | BWT: {BWT:.4f}")
    return R, ACC, BWT

# ---------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--loss", type=str, default="nll",
                        choices=["nll", "l1", "l2", "l1+l2", "l1_l2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    continual_training(depth=args.depth,
                       dropout=args.dropout,
                       optimizer=args.optimizer,
                       loss=args.loss,
                       seed=args.seed,
                       output_dir=args.outdir)

if __name__ == "__main__":
    main()
