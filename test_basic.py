"""Tests that need no internet and no trained model."""
import numpy as np

from src import config
from src.data_loader import START, UNK, encode_text, tokenize
from src.model import build_model


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Great movie!!! It's GOOD.") == ["great", "movie", "it's", "good"]


def test_encode_text_shape_and_special_tokens():
    word_index = {"great": 1, "movie": 2, "rare": config.VOCAB_SIZE + 50}
    x = encode_text("Great movie, unknownword rare", word_index)
    assert x.shape == (1, config.MAX_LEN)
    assert x[0, 0] == START
    assert list(x[0, 1:3]) == [4, 5]          # word number + 3
    assert x[0, 3] == UNK                      # not in dictionary
    assert x[0, 4] == UNK                      # outside the top VOCAB_SIZE words
    assert x[0, 5] == 0                        # padding


def test_long_review_is_truncated():
    x = encode_text("good " * 1000, {"good": 5})
    assert x.shape == (1, config.MAX_LEN)


def test_model_output_is_a_probability():
    model = build_model()
    out = model.predict(np.random.randint(0, config.VOCAB_SIZE, (4, config.MAX_LEN)), verbose=0)
    assert out.shape == (4, 1) and ((out >= 0) & (out <= 1)).all()
