import pytest
import pandas as pd
import numpy as np

from dense_feature_mixer import DenseFeatureMixer
from dense_feature_mixer.tests.test_basics import set_all_seeds


set_all_seeds(0)


@pytest.mark.parametrize(
    "task,numeric_vars,categorical_vars,target,dimensions",
    [
        ("classification", ["B"], ["A", "C"], [0, 1, 0], [2, 3]),
        ("regression", ["B", "D"], ["A", "C", "E"], [27, 5.5, -1.2], None),
    ],
)
def test_inverse_transform(task, numeric_vars, categorical_vars, target, dimensions):
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
    dfm = DenseFeatureMixer(
        task=task, numeric_vars=numeric_vars, dimensions=dimensions, epochs=1
    )
    dfm.fit(X[categorical_vars + numeric_vars], y)
    X_transformed = dfm.transform(X[categorical_vars + numeric_vars])
    inverted = dfm.inverse_transform(X_transformed)
    assert np.array_equal(inverted, X[categorical_vars])
