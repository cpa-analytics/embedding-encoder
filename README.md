![](https://raw.githubusercontent.com/cpa-analytics/embedding-encoder/main/logo.png)

## Overview

Embedding Encoder is a scikit-learn-compliant transformer that converts categorical variables into numeric vector representations. This is achieved by creating a small multilayer perceptron architecture in which each categorical variable is passed through an embedding layer, for which weights are extracted and turned into DataFrame columns.

While the idea is not new (it was popularized after [the team that landed in the 3rd place of the Rossmann Kaggle competition used it](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974)), and although Python implementations have surfaced over the years, we are not aware of any library that integrates this functionality into scikit-learn.

## Installation and dependencies

Embedding Encoder can be installed with

```bash
pip install embedding-encoder[tf]
```

Embedding Encoder has the following dependencies
* scikit-learn
* Tensorflow
* numpy
* pandas

Please see notes on non-Tensorflow usage at the end of this readme.

## Documentation

Full documentation including this readme and API reference can be found at [RTD](https://embedding-encoder.readthedocs.io/en/latest).

## Usage

Embedding Encoder works like any scikit-learn transformer, the only difference being that it requires `y` to be passed as it is the neural network's target.

Embedding Encoder will assume that all input columns are categorical and will calculate embeddings for each, unless the `numeric_vars` argument is passed. In that case, numeric variables will be included as an additional input to the neural network but no embeddings will be calculated for them, and they will not be included in the output transformation.

Please note that including numeric variables may reduce the interpretability of the final model as their total influence on the target variable can become difficult to disentangle.

The simplest usage example is

```python
from embedding_encoder import EmbeddingEncoder

ee = EmbeddingEncoder(task="regression") # or "classification"
ee.fit(X=X, y=y)
output = ee.transform(X=X)
```

## Compatibility with scikit-learn

Embedding Encoder can be included in pipelines as a regular transformer, and is compatible with cross-validation and hyperparameter optimization.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from embedding_encoder import EmbeddingEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ee = EmbeddingEncoder(task="classification")
num_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), ee)
col_transformer = ColumnTransformer([("num_transformer", num_pipe, numeric_vars),
                                     ("cat_transformer", cat_pipe, categorical_vars)])

pipe = make_pipeline(col_transformer,
                     LogisticRegression())
param_grid = {
    "columntransformer__cat__embeddingencoder__layers_units": [
        [64, 32, 16],
        [16, 8],
    ]
}
cv = GridSearchCV(pipeline, param_grid)
```

In the case of pipelines, if `numeric_vars` is specificed Embedding Encoder has to be the first step in the pipeline. This is because a Embedding Encoder with `numeric_vars` requires that its `X` input be a `DataFrame` with proper column names, which cannot be guaranteed if previous transformations are applied as is.

Alternatively, previous transformations can be included provided they are held inside the `ColumnTransformerWithNames` class in this library, which retains feature names.


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from embedding_encoder import EmbeddingEncoder
from embedding_encoder.utils import ColumnTransformerWithNames

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

## Plotting embeddings

The idea behind embeddings is that categories that are conceptually similar should have similar vector representations. For example, "December" and "January" should be close to each other when the target variable is ice cream sales (here in the Southern Hemisphere at least!).

This can be analyzed with the `plot_embeddings` function, which depends on Seaborn (`pip install embedding-encoder[sns]` or `pip install embedding-encoder[full]` which includes Tensorflow).

```python
from embedding_encoder import EmbeddingEncoder

ee = EmbeddingEncoder(task="classification")
ee.fit(X=X, y=y)
ee.plot_embeddings(variable="...", model="pca")
```

## Advanced usage

Embedding Encoder gives some control over the neural network. In particular, its constructor allows setting how deep and large the network should be (by modifying `layers_units`), as well as the dropout rate between dense layers. Epochs and batch size can also be modified.

These can be optimized with regular scikit-learn hyperparameter optimization techiniques.

The training loop includes an early stopping callback that restores the best weights (by default, the ones that minimize the validation loss).

## Non-Tensorflow usage

Tensorflow can be tricky to install on some systems, which could make Embedding Encoder less appealing if the user has no intention of using TF for modeling.

There are actually two partial ways of using Embedding Encoder without a TF installation.

1. Because TF is only used and imported in the `EmbeddingEncoder.fit()` method, once EE or the pipeline that contains EE has been fit, TF can be safely uninstalled; calls to methods like `EmbeddingEncoder.transform()` or `Pipeline.predict()` should raise no errors.
2. Embedding Encoder can save the mapping from categorical variables to embeddings to a JSON file which can be later imported by setting `pretrained=True`, requiring no TF whatsoever. This also opens up the opportunity to train embeddings for common categorical variables on common tasks and saving them for use in downstream tasks.

Installing EE without Tensorflow is as easy as removing "[tf]" from the install command.

```bash
pip install embedding-encoder
```
