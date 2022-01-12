from __future__ import annotations
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from tensorflow.keras import layers, Model
from sklearn.base import BaseEstimator, TransformerMixin


class DenseFeatureMixer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        task: str,
        categorical_vars: Optional[List[str]] = None,
        unknown_category: int = 999,
        numeric_vars: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        classif_classes: Optional[int] = None,
        classif_loss: Optional[str] = None,
        optimizer: str = "adam",
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 0,
    ):
        """Obtain numeric embeddings from categorical variables previously encoded as integers.

        Dense Feature Mixer trains a small neural network with categorical inputs passed through
        embedding layers. Numeric variables can be included as additional inputs.

        DFM returns (unique_values + 1) / 2 vectors per categorical variable, with a minimum of 2
        and a maximum of 50. However, this can be changed by passing a list of integers to `dimensions`.

        Parameters
        ----------
        task :
            "regression" or "classification". This determines the units in the head layer, loss and
            metrics used.
        categorical_vars :
            Array-like of strings containing the names of the categorical variables which will be
            processed.
        unknown_category :
            Out of vocabulary values will be mapped to this category. This should match the unknown
            value used in OrdinalEncoder.
        numeric_vars :
            Array-like of strings containing the names of the numeric variables that will be included
            as inputs to the network.
        dimensions :
            Array-like of integers containing the number of embedding dimensions for each categorical
            feature. If none, the dimension will be `min(50, int(np.ceil((unique + 1) / 2)))`
        classif_classes :
            Number of classes in `y` for classification tasks.
        classif_loss : Optional[str], optional
            Loss function for classification tasks.
        optimizer :
            Optimizer, default "adam".
        epochs :
            Number of epochs, default 10.
        batch_size : int, optional
            Batches size, default 32.
        verbose : int, optional
            Verbosity of the Keras `fit` method, default 0.

        Raises
        ------
        ValueError
            If `task` is not "regression" or "classification".
        ValueError
            If `classif_classes` or `classif_loss` are specified for regression tasks.
        ValueError
            If `numeric_vars` is specified and `categorical_vars` is not.
        ValueError
            If `dimensions` is specified and is not of the same length as `categorical_vars`.
        ValueError
            If `classif_classes` is specified but `classif_loss` is not.
        """
        if not task in ["regression", "classification"]:
            raise ValueError("task must be either regression or classification")
        self.task = task

        if task == "regression" and (classif_classes or classif_loss):
            raise ValueError(
                "classif_classes and classif_loss must be None for regression"
            )
        if not categorical_vars and numeric_vars:
            raise ValueError("categorical_vars must be specified if numeric_vars is specified")
        self.categorical_vars = categorical_vars
        self.unknown_category = unknown_category
        self.numeric_vars = numeric_vars

        if dimensions:
            if len(dimensions) != len(categorical_vars):
                raise ValueError(
                    "Dimensions must be of same length as categorical variables"
                )
        self.dimensions = dimensions

        if (classif_classes and not classif_loss) or (
            classif_loss and not classif_classes
        ):
            raise ValueError(
                "If any of classif_classes or classif_loss is specified, both must be specified"
            )
        self.classif_classes = classif_classes
        self.classif_loss = classif_loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ) -> DenseFeatureMixer:
        self._validate_data(X=X, y=y)
        if self.categorical_vars:
            self._categorical_vars = self.categorical_vars
        elif isinstance(X, pd.DataFrame):
            self._categorical_vars = X.columns
        else:
            # Assume it's a numpy array
            X = pd.DataFrame(X, columns=[f"cat{i}" for i in range(X.shape[1])])
            self._categorical_vars = X.columns
        self._numeric_vars = self.numeric_vars if self.numeric_vars else []

        categorical_inputs = []
        categorical_embedded = []
        for i, catvar in enumerate(self._categorical_vars):
            unique = X[catvar].nunique() + 1 # add one more for unknown category
            if self.dimensions:
                dimension = self.dimensions[i]
            else:
                dimension = min(50, int(np.ceil(unique / 2)))
            categorical_input = layers.Input(
                shape=(), name=f"categorical_input_{catvar}"
            )
            categorical_inputs.append(categorical_input)
            embedding = layers.Embedding(unique, dimension, name=f"embedding_{catvar}")(
                categorical_input
            )
            categorical_embedded.append(embedding)

        if len(self._categorical_vars) > 1:
            all_categorical = layers.Concatenate()(categorical_embedded)
        else:
            all_categorical = categorical_embedded[0]
        if self._numeric_vars:
            numeric_input = layers.Input(
                shape=(
                    len(
                        self._numeric_vars,
                    )
                ),
                name="numeric_input",
            )
            x = layers.Concatenate()([numeric_input, all_categorical])
        else:
            x = all_categorical
            numeric_input = []
        x = layers.Dense(32, activation="relu")(x) # we could allow the user to provide their own nn body architecture
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation="relu")(x)
        if self.task == "regression":
            output = layers.Dense(1, activation="relu")(x)
            loss = "mse"
            metrics = [loss]
        else:
            metrics = ["accuracy"]
            if self.classif_classes:
                output_units = self.classif_classes
                loss = self.classif_loss
            else:
                nunique_y = len(np.unique(y))
                if y.ndim == 1 and nunique_y == 2:
                    output_units = 1
                elif y.dim == 1 and nunique_y > 2:
                    output_units = nunique_y
                else:
                    output_units = y.shape[1]
            if output_units == 1:
                output_activation = "sigmoid"
                loss = "binary_crossentropy"
            else:
                output_activation = "softmax"
                if y.dim == 1:
                    loss = "sparse_categorical_crossentropy"
                else:
                    loss = "categorical_crossentropy"
            output = layers.Dense(output_units, activation=output_activation)(x)
        if len(self._categorical_vars) > 1:
            self._model = Model(inputs=[numeric_input] + categorical_inputs, outputs=output)
        elif self._numeric_vars:
            self._model = Model(inputs=[numeric_input] + categorical_inputs, outputs=output)
        else:
            self._model = Model(inputs=categorical_inputs[0], outputs=output)

        self._model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
        numeric_x = [X[self._numeric_vars]] if self._numeric_vars else []
        merged_x = numeric_x + [X[i] for i in self._categorical_vars]
        self._model.fit(
            x=merged_x,
            y=y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        self._weights = {
            k: self._model.get_layer(f"embedding_{k}").weights
            for k in self._categorical_vars
        }
        self._embeddings_mapping = {
            k: pd.DataFrame(
                self._weights[k][0].numpy(),
                index=np.sort(np.append(X[k].unique(), self.unknown_category)),
                columns=[
                    f"embedding_{k}_{i}" for i in range(self._weights[k][0].shape[1])
                ],
            )
            for k in self._categorical_vars
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"cat{i}" for i in range(X.shape[1])])
        if not all(i in X.columns for i in self._categorical_vars):
            raise ValueError("X must contain all categorical variables specified in the constructor. If none were specified, X must have the same number of columns as X used in fit.")
        final_embeddings = []
        for k in self._categorical_vars:
            final_embedding = X.join(self._embeddings_mapping[k], on=k, how="left").drop(
                self._categorical_vars, axis=1
            )
            final_embeddings.append(final_embedding)
        final_embeddings = pd.concat(final_embeddings, axis=1)

        final_x = pd.concat(
            [X.drop(self._categorical_vars, axis=1), final_embeddings], axis=1
        )
        final_x = final_x.loc[:, ~final_x.columns.duplicated()]

        return final_x
