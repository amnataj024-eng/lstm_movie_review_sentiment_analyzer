"""Small helpers: random seeds and plots."""
import random

import matplotlib
matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)


def plot_history(history: dict, path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["accuracy"], marker="o", label="Training")
    ax[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, history["loss"], marker="o", label="Training")
    ax[1].plot(epochs, history["val_loss"], marker="o", label="Validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def plot_confusion_matrix(cm, path) -> None:
    names = ["Negative", "Positive"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(names); ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
