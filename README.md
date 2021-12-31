# Dense Feature Mixer

## Introduction

Dense Feature Mixer (DFM) is a Scikit-Learn-compliant transformer that converts categorical variables to numerical vector representations. This is achieved by creating a small neural network architecture in which each categorical variable is passed through an embedding layer, for which weights are extracted and turned into DataFrame columns.

## Usage

DFM works like any Scikit-Learn transformer, the only difference being that it requires `y` to be passed as it is the neural network's target.

The `numeric_vars` argument is optional. If available, these variables will be included as an additional input to the neural network. This may reduce the interpretability of the final model as the complete effect of numeric variables on the target variable can become unclear.

```python
from dense_feature_mixer import DenseFeatureMixer

dfm = DenseFeatureMixer(task="regression", categorical_vars=["..."], numeric_vars=["..."])
dfm.fit(X=X_train, y=y_train)
output = dfm.transform(X=X_train)
```

DFM expects categorical variables to be integer arrays, which can be done easily with scikit's `OrdinalEncoder`. This can be included in a sklearn `Pipeline` provided transformations preceding DFM are held inside the provided `ColumnTransformerWithNames`. Without it, DFM can be included only as the first step in a `Pipeline` because it requires its `X` input to be a `DataFrame` with proper column names, which cannot be guaranteed if previous transformations are applied.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from dense_feature_mixer import DenseFeatureMixer
from dense_feature_mixer.compose import ColumnTransformerWithNames

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
col_transformer = ColumnTransformerWithNames([("num_transformer", StandardScaler(), numeric_vars),
                                              ("cat_transformer", ordinal_encoder, categorical_vars)],
                                             remainder="drop")

pipe = make_pipeline(col_transformer,
                     DenseFeatureMixer(task="classification",
                                       categorical_vars=categorical_vars,
                                       unknown_category=999),
                     LogisticRegression())
pipe.fit(X_train, y_train)
```