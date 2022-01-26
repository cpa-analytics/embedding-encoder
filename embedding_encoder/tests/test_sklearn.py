import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from embedding_encoder import EmbeddingEncoder
from embedding_encoder.tests.test_basics import set_all_seeds


set_all_seeds(0)


def test_pipeline_cv():
    X = pd.DataFrame(
        {
            "A": ["a", "b", "c", np.nan, "e", "f"],
            "B": ["z", "y", "x", "w", "v", "u"],
            "C": np.random.normal(100, 20, 6),
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    numeric_vars = ["C"]
    categorical_vars = ["A", "B"]
    ee = EmbeddingEncoder(task="classification", epochs=1)
    cat_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"), ee)
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_vars),
            ("cat", cat_transformer, categorical_vars),
        ]
    )
    pipeline = make_pipeline(preprocessor, LogisticRegression())
    param_grid = {
        "columntransformer__cat__embeddingencoder__layers_units": [
            [64, 32, 16],
            [16, 8],
        ]
    }
    cv = GridSearchCV(pipeline, param_grid, cv=2)
    cv.fit(X, y)
    assert hasattr(cv, "best_estimator_")
