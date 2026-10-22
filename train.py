"""Train the LSTM and save the model + training history."""
import json

from tensorflow import keras

from . import config
from .data_loader import get_word_index, load_imdb
from .model import build_model
from .utils import plot_history, set_seed


def train() -> None:
    set_seed(config.SEED)
    for folder in (config.MODELS_DIR, config.RESULTS_DIR, config.DATA_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print("Loading IMDB data (the first run downloads it, about 20 MB)...")
    (x_train, y_train), _ = load_imdb()
    get_word_index()  # cache the dictionary for later predictions
    print(f"Training reviews: {x_train.shape[0]}, each {x_train.shape[1]} words long")

    model = build_model()
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving and keep the best version.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train, y_train,
        validation_split=config.VALIDATION_SPLIT,
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )
    model.save(config.MODEL_PATH)
    config.HISTORY_PATH.write_text(json.dumps(history.history), encoding="utf-8")
    plot_history(history.history, config.RESULTS_DIR / "training_curves.png")
    print(f"\nModel saved to {config.MODEL_PATH}")
    print(f"Curves saved to {config.RESULTS_DIR / 'training_curves.png'}")
