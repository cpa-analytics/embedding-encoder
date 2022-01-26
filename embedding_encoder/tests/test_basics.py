import random

import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder

from embedding_encoder import EmbeddingEncoder


def set_all_seeds(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_all_seeds(0)


@pytest.mark.parametrize(
    "task,numeric_vars,categorical_vars,target",
    [
        ("classification", None, ["A"], [4, 5, 6]),
        ("classification", ["B"], ["A", "C"], [0, 1, 0]),
        ("regression", ["B", "D"], ["A", "C", "E"], [27, 5.5, -1.2]),
    ],
)
def test_inputs(task, numeric_vars, categorical_vars, target):
    X = pd.DataFrame(
        {
            "A": ["x", "y", "z"],
            "B": [1, 2, 3],
            "C": ["a", "b", "c"],
            "D": [6, 7, 8],
            "E": [9, 10, 11],
        }
    )
    y = np.array(target)
    numeric_vars = numeric_vars or []
    ee = EmbeddingEncoder(task=task, numeric_vars=numeric_vars, epochs=1)
    ee.fit(X[categorical_vars + numeric_vars], y)
    X_transformed = ee.transform(X[categorical_vars + numeric_vars])
    n_cols = 2 * len(categorical_vars)
    assert X_transformed.shape == (X.shape[0], n_cols)


def test_array_X():
    X = np.array([["x", 1], ["y", 2], ["z", 3]])
    y = np.array([1, 0, 0])
    ee = EmbeddingEncoder(task="classification", epochs=1)
    ee.fit(X, y)
    X_transformed = ee.transform(X)
    assert X_transformed.shape == (X.shape[0], 4)


@pytest.mark.parametrize("encode,dimensions", [(True, None), (False, [3, 2])])
def test_basic_parameters(encode, dimensions):
    X = pd.DataFrame(
        {"A": ["a", "b", "c", "d", "e", "f"], "B": ["z", "y", "x", "w", "v", "u"]}
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    ee = EmbeddingEncoder(
        task="classification", encode=encode, epochs=1, dimensions=dimensions
    )
    if encode is False:
        encoder = OrdinalEncoder()
        X[["A", "B"]] = encoder.fit_transform(X[["A", "B"]])
    ee.fit(X, y)
    X_transformed = ee.transform(X)
    # 7 unique values, + 1 for oov, divided by 2 and rounded up = 4 * 2 variables = 8
    n_cols = sum(dimensions) if dimensions else 8
    assert X_transformed.shape == (X.shape[0], n_cols)


@pytest.mark.parametrize(
    "layers_units,dropout,validation_split",
    [([24], 0.0, 0.0), ([64, 32, 16], 0.5, 0.1)],
)
def test_nn_parameters(layers_units, dropout, validation_split):
    X = pd.DataFrame(
        {"A": ["a", "b", "c", "d", "e", "f"], "B": ["z", "y", "x", "w", "v", "u"]}
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    ee = EmbeddingEncoder(
        task="classification",
        epochs=1,
        layers_units=layers_units,
        dropout=dropout,
        validation_split=validation_split,
    )
    ee.fit(X, y)
    X_transformed = ee.transform(X)
    # 7 unique values, + 1 for oov, divided by 2 and rounded up = 4 * 2 variables = 8
    assert X_transformed.shape == (X.shape[0], 8)
