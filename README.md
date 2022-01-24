# Dense Feature Mixer

## Overview

Dense Feature Mixer (DFM) is a scikit-learn-compliant transformer that converts categorical variables to numeric vector representations. This is achieved by creating a small multilayer perceptron architecture in which each categorical variable is passed through an embedding layer, for which weights are extracted and turned into DataFrame columns.

## Installation and dependencies

DFM can be installed (soon) with

```bash
pip install dense-feature-mixer
```

DFM has the following dependencies
* scikit-learn
* Tensorflow
* numpy
* pandas

## Usage

DFM works like any scikit-learn transformer, the only difference being that it requires `y` to be passed as it is the neural network's target. By default DFM will convert categorical variables into integer arrays by applying scikit-learn's `OrdinalEncoder`.

It will assume that all input columns are categorical and will calculate embeddings for each, unless the `numeric_vars` argument is passed. In that case, numeric variables will be included as an additional input to the neural network but no embeddings will be calculated for them, and they will not be included in the output transformation.

Please note that including numeric variables may reduce the interpretability of the final model as their total influence on the target variable can become difficult to disentangle.

The simplest usage example is

```python
from dense_feature_mixer import DenseFeatureMixer

dfm = DenseFeatureMixer(task="regression")
dfm.fit(X=X, y=y)
output = dfm.transform(X=X)
```

## Compatibility with scikit-learn

DFM can be included in pipelines as a regular transformer, and is compatible with cross-validation and hyperparameter optimization.

In the case of pipelines, if `numeric_vars` is specificed DFM has to be the first step in the pipeline. This is because a DFM with `numeric_vars` requires that its `X` input be a `DataFrame` with proper column names, which cannot be guaranteed if previous transformations are applied as is.

Alternatively, previous transformations can be included provided they are held inside the `ColumnTransformerWithNames` class in this library.


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from dense_feature_mixer import DenseFeatureMixer
from dense_feature_mixer.compose import ColumnTransformerWithNames

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dfm = DenseFeatureMixer(task="classification", numeric_vars=numeric_vars)
num_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
cat_transformer = SimpleImputer(strategy="most_frequent")
col_transformer = ColumnTransformerWithNames([("num_transformer", num_pipe, numeric_vars),
                                              ("cat_transformer", cat_transformer, categorical_vars)])

pipe = make_pipeline(col_transformer,
                     dfm,
                     LogisticRegression())
pipe.fit(X_train, y_train)
```

Like scikit transformers, DFM also has a `inverse_transform` method that recomposes the original input.