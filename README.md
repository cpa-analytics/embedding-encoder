# Embedding Encoder

## Overview

Embedding Encoder is a scikit-learn-compliant transformer that converts categorical variables to numeric vector representations. This is achieved by creating a small multilayer perceptron architecture in which each categorical variable is passed through an embedding layer, for which weights are extracted and turned into DataFrame columns.

## Installation and dependencies

Embedding Encoder can be installed (soon) with

```bash
pip install embedding-encoder
```

Embedding Encoder has the following dependencies
* scikit-learn
* Tensorflow
* numpy
* pandas

## Usage

Embedding Encoder works like any scikit-learn transformer, the only difference being that it requires `y` to be passed as it is the neural network's target. By default it will convert categorical variables into integer arrays by applying scikit-learn's `OrdinalEncoder`.

Embedding Encoder will assume that all input columns are categorical and will calculate embeddings for each, unless the `numeric_vars` argument is passed. In that case, numeric variables will be included as an additional input to the neural network but no embeddings will be calculated for them, and they will not be included in the output transformation.

Please note that including numeric variables may reduce the interpretability of the final model as their total influence on the target variable can become difficult to disentangle.

The simplest usage example is

```python
from embedding_encoder import EmbeddingEncoder

ee = EmbeddingEncoder(task="regression")
ee.fit(X=X, y=y)
output = ee.transform(X=X)
```

## Compatibility with scikit-learn

Embedding Encoder can be included in pipelines as a regular transformer, and is compatible with cross-validation and hyperparameter optimization.

In the case of pipelines, if `numeric_vars` is specificed Embedding Encoder has to be the first step in the pipeline. This is because a Embedding Encoder with `numeric_vars` requires that its `X` input be a `DataFrame` with proper column names, which cannot be guaranteed if previous transformations are applied as is.

Alternatively, previous transformations can be included provided they are held inside the `ColumnTransformerWithNames` class in this library, which retains feature names.


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from embedding_encoder import EmbeddingEncoder
from embedding_encoder.compose import ColumnTransformerWithNames

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ee = EmbeddingEncoder(task="classification", numeric_vars=numeric_vars)
num_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
cat_transformer = SimpleImputer(strategy="most_frequent")
col_transformer = ColumnTransformerWithNames([("num_transformer", num_pipe, numeric_vars),
                                              ("cat_transformer", cat_transformer, categorical_vars)])

pipe = make_pipeline(col_transformer,
                     ee,
                     LogisticRegression())
pipe.fit(X_train, y_train)
```

Like scikit transformers, Embedding Encoder also has a `inverse_transform` method that recomposes the original input.

## Advanced usage

Embedding Encoder gives some control over the neural network. In particular, its constructor allows setting how deep and large the network should be (by modifying `layers_units`), as well as the dropout rate between dense layers. Epochs and batch size can also be modified.

These can be optimized with regular scikit-learn hyperparameter optimization techiniques.

The training loop includes an early stopping callback that restores the best weights (by default, the ones that minimize the validation loss).
