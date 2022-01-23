from __future__ import annotations
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class DenseFeatureMixer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        task: str,
        encode: bool = True,
        unknown_category: int = 999,
        numeric_vars: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        layers_units: Optional[List[int]] = None,
        dropout: float = 0.2,
        classif_classes: Optional[int] = None,
        classif_loss: Optional[str] = None,
        optimizer: str = "adam",
        epochs: int = 5,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0,
    ):
        """Obtain numeric embeddings from categorical variables.

        Dense Feature Mixer trains a small neural network with categorical inputs passed through
        embedding layers. Numeric variables can be included as additional inputs by setting
        `numeric_vars`.

        By default, non numeric variables are encoded with Scikit-Learn's `OrdinalEncoder`. This
        can be changed by setting `encode=False` if no encoding is necessary.

        DFM returns (unique_values + 1) / 2 vectors per categorical variable, with a minimum of 2
        and a maximum of 50. However, this can be changed by passing a list of integers to `dimensions`.

        The neural network architecture and training loop can be partially modified. `layers_units`
        takes an array of integers, each representing an additional dense layer, i.e, `[32, 24, 16]`
        will create 3 hidden layers with the corresponding units, with dropout layers interleaved,
        while `dropout` controls the dropout rate.

        While DFM will try to infer the appropiate number of units for the output layer and the
        model's loss for classification tasks, these can be set with `classif_classes` and
        `classif_loss`. Regression tasks will always have 1 unit in the output layer and mean
        squared error loss.

        `optimizer` and `batch_size` are passed directly to Keras.

        `validation_split` is also passed to Keras. Setting it to something higher than 0 will use
        validation loss in order to decide whether to stop training early. Otherwise train loss
        will be used.

        Parameters
        ----------
        task :
            "regression" or "classification". This determines the units in the head layer, loss and
            metrics used.
        encode :
            Whether to apply `OrdinalEncoder` to categorical variables, by default True.
        unknown_category :
            Out of vocabulary values will be mapped to this category. This should match the unknown
            value used in OrdinalEncoder.
        numeric_vars :
            Array-like of strings containing the names of the numeric variables that will be included
            as inputs to the network.
        dimensions :
            Array-like of integers containing the number of embedding dimensions for each categorical
            feature. If none, the dimension will be `min(50, int(np.ceil((unique + 1) / 2)))`
        layers_units :
            Array-like of integers which define how many dense layers to include and how many units
            they should have. By default None, which creates two hidden layers with 24 and 12 units.
        dropout :
            Dropout rate used between dense layers.
        classif_classes :
            Number of classes in `y` for classification tasks.
        classif_loss : Optional[str], optional
            Loss function for classification tasks.
        optimizer :
            Optimizer, default "adam".
        epochs :
            Number of epochs, default 3.
        batch_size :
            Batches size, default 32.
        validation_split :
            Passed to Keras `Model.fit`.
        verbose :
            Verbosity of the Keras `Model.fit`, default 0.

        Raises
        ------
        ValueError
            If `task` is not "regression" or "classification".
        ValueError
            If `classif_classes` or `classif_loss` are specified for regression tasks.
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
        self.encode = encode
        self.unknown_category = unknown_category
        self.numeric_vars = numeric_vars
        self.dimensions = dimensions
        self.layers_units = layers_units
        self.dropout = dropout

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
        self.validation_split = validation_split
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ) -> DenseFeatureMixer:
        """
        Fit the DenseFeatureMixer to X.

        Parameters
        ----------
        X :
            The data to process. It can include numeric variables that will not be encoded but will
            be used in the neural network as additional inputs.

        y :
            Target data. Used as target in the neural network.

        Returns
        -------
        self : object
            Fitted DFM.
        """
        self._numeric_vars = self.numeric_vars or []
        self._layers_units = self.layers_units or [24, 12]

        if self._numeric_vars and not isinstance(X, pd.DataFrame):
            raise ValueError("Cannot specify numeric_vars if X is not a DataFrame.")

        if self.dimensions:
            if len(self.dimensions) != (X.shape[1] - len(self._numeric_vars)):
                raise ValueError(
                    "Dimensions must be of same length as non-numeric variables"
                )

        X_copy = X.copy()
        if self.encode:
            self._validate_data(y=y)
        else:
            self._validate_data(X=X_copy, y=y)

        if isinstance(X_copy, pd.DataFrame):
            self._categorical_vars = [
                x for x in X_copy.columns if x not in self._numeric_vars
            ]
        else:
            # Assume it's a numpy array and that all columns are categorical
            X_copy = pd.DataFrame(
                X_copy, columns=[f"cat{i}" for i in range(X_copy.shape[1])]
            )
            self._categorical_vars = list(X_copy.columns)
        self._fit_dtypes = X_copy.dtypes

        if self.encode:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=self.unknown_category
            )
            X_copy[self._categorical_vars] = self._ordinal_encoder.fit_transform(
                X_copy[self._categorical_vars]
            )

        categorical_inputs = []
        categorical_embedded = []
        for i, catvar in enumerate(self._categorical_vars):
            # Add one more dimension for unseen values (oov)
            unique = X_copy[catvar].nunique() + 1
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
                shape=(len(self._numeric_vars)), name="numeric_input"
            )
            x = layers.Concatenate()([numeric_input, all_categorical])
        else:
            x = all_categorical
            numeric_input = []

        for units in self._layers_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.dropout)(x)

        if self.task == "regression":
            output = layers.Dense(1)(x)
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
                elif y.ndim == 1 and nunique_y > 2:
                    output_units = nunique_y
                else:
                    output_units = y.shape[1]
            if output_units == 1:
                output_activation = "sigmoid"
                loss = "binary_crossentropy"
            else:
                output_activation = "softmax"
                if y.ndim == 1:
                    y = tf.one_hot(y, output_units)
                loss = "categorical_crossentropy"
            output = layers.Dense(output_units, activation=output_activation)(x)

        if len(self._categorical_vars) > 1 or self._numeric_vars:
            self._model = Model(
                inputs=[numeric_input] + categorical_inputs, outputs=output
            )
        else:
            self._model = Model(inputs=categorical_inputs[0], outputs=output)
        self._model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

        numeric_x = (
            [np.array(X_copy[self._numeric_vars]).astype(np.float32)]
            if self._numeric_vars
            else []
        )
        merged_x = numeric_x + [
            X_copy[i].astype(np.float32) for i in self._categorical_vars
        ]
        if self.validation_split > 0.0:
            monitor = "val_loss"
        else:
            monitor = "loss"
        self._history = self._model.fit(
            x=merged_x,
            y=y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=[callbacks.EarlyStopping(monitor=monitor, patience=2,
                                               restore_best_weights=True)]
        )

        self._weights = {
            k: self._model.get_layer(f"embedding_{k}").weights
            for k in self._categorical_vars
        }
        self._embeddings_mapping = {
            k: pd.DataFrame(
                self._weights[k][0].numpy(),
                index=np.sort(np.append(X_copy[k].unique(), self.unknown_category)),
                columns=[
                    f"embedding_{k}_{i}" for i in range(self._weights[k][0].shape[1])
                ],
            )
            for k in self._categorical_vars
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X using computed variable embeddings.

        Parameters
        ----------
        X :
            The data to process.

        Returns
        -------
        embeddings :
            Vector embeddings for each categorical variable.
        """
        if not X.shape[1] == len(self._categorical_vars) + len(self._numeric_vars):
            raise ValueError("X must have the same dimensions as used in training.")

        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(
                X_copy, columns=[f"cat{i}" for i in range(X_copy.shape[1])]
            )
        if not all(i in X_copy.columns for i in self._categorical_vars):
            raise ValueError("X must contain all categorical variables.")

        if self.encode:
            X_copy[self._categorical_vars] = self._ordinal_encoder.transform(
                X_copy[self._categorical_vars]
            )
        final_embeddings = []
        for k in self._categorical_vars:
            final_embedding = X_copy.join(self._embeddings_mapping[k], on=k, how="left")
            final_embeddings.append(final_embedding)
        final_embeddings = pd.concat(final_embeddings, axis=1).drop(
            self._categorical_vars + self._numeric_vars, axis=1
        )

        self.columns_out = final_embeddings.columns

        return final_embeddings

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform X using computed variable embeddings.

        Parameters
        ----------
        X :
            The data to process.

        Returns
        -------
        Original DataFrame.
        """
        X_copy = X.copy()

        inverted_dfs = []
        for k in self._categorical_vars:
            mapping = self._embeddings_mapping[k]
            embeddings_columns = list(mapping.columns)
            mapping = mapping.reset_index().set_index(embeddings_columns)
            inverted = X_copy[embeddings_columns].join(mapping, on=embeddings_columns)
            inverted = inverted.drop(embeddings_columns, axis=1)
            inverted = inverted.rename({"index": k}, axis=1)
            inverted_dfs.append(inverted)
        output = pd.concat(inverted_dfs, axis=1)

        if self.encode:
            original = self._ordinal_encoder.inverse_transform(output)
            original = pd.DataFrame(original, columns=output.columns, index=X_copy.index)
        else:
            original = output
        original = original.astype(dict(zip(original.columns, self._fit_dtypes)))
        return original

    def get_feature_names_out(self, input_features=None):
        return self.columns_out

    def get_feature_names(self, input_features=None):
        return self.columns_out
