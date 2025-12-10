# rnn_assignment.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import os

# --------------------------
# Load Dataset
# --------------------------
def load_notmnist_like():
    """
    Use MNIST as surrogate if notMNIST is not available.
    Convert each image into a sequence of 28 time steps with 28 features.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_notmnist_like()
print("Dataset Loaded:", x_train.shape, y_train.shape)

# --------------------------
# GRU Cell
# --------------------------
class GRUCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Update gate
        self.Wz = self.add_weight(name='Wz', shape=(input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bz = self.add_weight(name='bz', shape=(self.units,), initializer='zeros')

        # Reset gate
        self.Wr = self.add_weight(name='Wr', shape=(input_dim + self.units, self.units), initializer='glorot_uniform')
        self.br = self.add_weight(name='br', shape=(self.units,), initializer='zeros')

        # Candidate
        self.Ws = self.add_weight(name='Ws', shape=(input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bs = self.add_weight(name='bs', shape=(self.units,), initializer='zeros')

    def call(self, x, prev_h):
        concat = tf.concat([prev_h, x], axis=1)

        z = tf.sigmoid(tf.matmul(concat, self.Wz) + self.bz)
        r = tf.sigmoid(tf.matmul(concat, self.Wr) + self.br)

        r_h = r * prev_h
        concat_candidate = tf.concat([r_h, x], axis=1)
        h_tilda = tf.tanh(tf.matmul(concat_candidate, self.Ws) + self.bs)

        h_new = (1 - z) * prev_h + z * h_tilda
        return h_new

# --------------------------
# MGU Cell
# --------------------------
class MGUCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Forget gate
        self.Wf = self.add_weight(name='Wf', shape=(input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bf = self.add_weight(name='bf', shape=(self.units,), initializer='zeros')

        # Candidate
        self.Ws = self.add_weight(name='Ws', shape=(input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bs = self.add_weight(name='bs', shape=(self.units,), initializer='zeros')

    def call(self, x, prev_h):
        concat = tf.concat([prev_h, x], axis=1)

        f = tf.sigmoid(tf.matmul(concat, self.Wf) + self.bf)
        f_h = f * prev_h

        concat_candidate = tf.concat([f_h, x], axis=1)
        h_tilda = tf.tanh(tf.matmul(concat_candidate, self.Ws) + self.bs)

        h_new = (1 - f) * prev_h + f * h_tilda
        return h_new

# --------------------------
# Custom RNN Wrapper
# --------------------------
class CustomRNN(tf.keras.Model):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.zeros((batch_size, self.cell.units))

        for t in range(x.shape[1]):
            h = self.cell(x[:, t, :], h)
        return h

# --------------------------
# Full Model
# --------------------------
class RNNClassifier(tf.keras.Model):
    def __init__(self, cell_type="gru", hidden_units=128):
        super().__init__()
        if cell_type.lower() == "gru":
            self.rnn = CustomRNN(GRUCell(hidden_units))
        else:
            self.rnn = CustomRNN(MGUCell(hidden_units))

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        h = self.rnn(x)
        return self.out(h)

# --------------------------
# Training Function
# --------------------------
def train_model(cell_type="gru", hidden_units=128, epochs=5):
    model = RNNClassifier(cell_type=cell_type, hidden_units=hidden_units)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    return history

# --------------------------
# Run 3 Trials
# --------------------------
def run_trials(cell_type):
    histories = []
    for i in range(3):
        print(f"\n=== Trial {i+1} {cell_type.upper()} ===")
        histories.append(train_model(cell_type=cell_type))
    return histories

# --------------------------
# Plotting Function
# --------------------------
def plot_history(histories, title):
    plt.figure(figsize=(8,5))
    for h in histories:
        plt.plot(h.history["val_accuracy"])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.show()

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    gru_hist = run_trials("gru")
    mgu_hist = run_trials("mgu")

    plot_history(gru_hist, "GRU Validation Accuracy")
    plot_history(mgu_hist, "MGU Validation Accuracy")

