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

DFM expects categorical variables to be integer arrays, which can be done easily with scikit's `OrdinalEncoder`. However, at the moment `DenseFeatureMixer` can only be the first step in a `Pipeline` because it requires its `X` input to be a `DataFrame` with proper column names, which cannot be guaranteed if previous transformations are applied. This means that ordinal encoding has to be performed outside of the pipeline.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
X_train[categorical_vars] = encoder.fit_transform(X_train[categorical_vars])
X_test[categorical_vars] = encoder.transform(X_test[categorical_vars])

num_transformer = make_column_transformer((StandardScaler(), numeric_vars), remainder="passthrough")

pipe = make_pipeline(DenseFeatureMixer(task="classification",
                                       categorical_vars=categorical_vars),
                     num_transformer,
                     LogisticRegression())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```