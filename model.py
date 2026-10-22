"""The LSTM network."""
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_model() -> keras.Model:
    """Embedding -> LSTM -> Dense -> Sigmoid output (a probability that the review is positive)."""
    model = keras.Sequential(
        [
            keras.Input(shape=(config.MAX_LEN,)),
            # Each word number becomes a learned vector of EMBEDDING_DIM numbers.
            # mask_zero=True tells the LSTM to ignore the padding (zeros).
            layers.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, mask_zero=True),
            layers.LSTM(config.LSTM_UNITS, dropout=config.DROPOUT),   # reads the words in order
            layers.Dropout(config.DROPOUT),                           # reduces overfitting
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),                    # 0 = negative ... 1 = positive
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model
