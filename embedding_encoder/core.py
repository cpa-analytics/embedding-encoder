from __future__ import annotations
import json
from typing import List, Optional, Union, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from embedding_encoder.custom_ordinal import OrdinalEncoderStart1


class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Obtain numeric embeddings from categorical variables.

    Embedding Encoder trains a small neural network with categorical inputs passed through
    embedding layers. Numeric variables can be included as additional inputs by setting
    :attr:`numeric_vars`.

    Embedding Encoder returns (unique_values + 1) / 2 vectors per categorical variable, with a minimum of 2
    and a maximum of 50. However, this can be changed by passing a list of integers to :attr:`dimensions`.

    The neural network architecture and training loop can be partially modified. :attr:`layers_units`
    takes an array of integers, each representing an additional dense layer, i.e, `[32, 24, 16]`
    will create 3 hidden layers with the corresponding units, with dropout layers interleaved,
    while :attr:`dropout` controls the dropout rate.

    While Embedding Encoder will try to infer the appropiate number of units for the output layer and the
    model's loss for classification tasks, these can be set with :attr:`classif_classes` and
    :attr:`classif_loss`. Regression tasks will always have 1 unit in the output layer and mean
    squared error loss.

    :attr:`optimizer` and :attr:`batch_size` are passed directly to Keras.

    :attr:`validation_split` is also passed to Keras. Setting it to something higher than 0 will use
    validation loss in order to decide whether to stop training early. Otherwise train loss
    will be used.

    :attr:`mapping_path` is the path to a JSON file where the embedding mapping will be saved. If
    :attr:`pretrained` is set to True, the mapping will be loaded from this file and no model will
    be trained.

    Parameters
    ----------
    task :
        "regression" or "classification". This determines the units in the head layer, loss and
        metrics used.
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
    mapping_path :
        Path to a JSON file where the mapping from categorical variables to embeddings will be saved.
        If :attr:`pretrained` is True, the mapping will be loaded from this file and no model will
        be trained.
    pretrained :
        Whether to use pretrained embeddings found in the JSON at :attr:`mapping_path`.
    keep_model :
        Whether to assign the Tensorflow model to :attr:`_model`. Setting to True will prevent the
        EmbeddingEncoder from being pickled. Default False. Please note that the model's `history`
        dict is available at :attr:`_history`.

    Attributes
    ----------
    _history : `dict`
        Keras `model.history.history` containing training data.
    _model : `keras.Model`
        Keras model. Only available if :attr:`keep_model` is True.
    _embeddings_mapping : dict
        Dictionary mapping categorical variables to their embeddings.

    Raises
    ------
    ValueError
        If `task` is not "regression" or "classification".
    ValueError
        If `classif_classes` or `classif_loss` are specified for regression tasks.
    ValueError
        If `classif_classes` is specified but `classif_loss` is not.
    """

    _required_parameters = ["task"]

    def __init__(
        self,
        task: str,
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
        mapping_path: Optional[Union[str, Path]] = None,
        pretrained: bool = False,
        keep_model: bool = False,
    ):
        if not task in ["regression", "classification"]:
            raise ValueError("task must be either regression or classification")
        self.task = task

        if task == "regression" and (classif_classes or classif_loss):
            raise ValueError(
                "classif_classes and classif_loss must be None for regression"
            )
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
        if pretrained is True and not mapping_path:
            raise ValueError("mapping_path must be specified when pretrained is True")
        self.mapping_path = mapping_path
        self.pretrained = pretrained
        self.keep_model = keep_model

    def _more_tags(self):
        return {
            "requires_y": True,
            "non_deterministic": True,
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
            "_xfail_checks": {"check_fit_idempotent": "EE is non-deterministic"},
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ) -> EmbeddingEncoder:
        """
        Fit the EmbeddingEncoder to X.

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
            Fitted Embedding Encoder.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (pd.DataFrame, np.ndarray)):
            y = np.array(y)
        try:
            self._validate_data(X=X, y=y, dtype=None, ensure_min_samples=3)
        except ValueError as error:
            if "Expected 2D array" in str(error):
                raise ValueError("EmbeddingEncoder does not accept sparse data.")
            else:
                raise error
        self._numeric_vars = self.numeric_vars or []

        if self._numeric_vars and not isinstance(X, pd.DataFrame):
            raise ValueError("Cannot specify numeric_vars if X is not a DataFrame.")

        if self.dimensions:
            if len(self.dimensions) != (X.shape[1] - len(self._numeric_vars)):
                raise ValueError(
                    "Dimensions must be of same length as non-numeric variables"
                )

        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            self._categorical_vars = [
                col for col in X_copy.columns if col not in self._numeric_vars
            ]
        else:
            # Assume it's a numpy array and that all columns are categorical
            X_copy = pd.DataFrame(
                np.copy(X), columns=[f"cat{i}" for i in range(X.shape[1])]
            )
            self._categorical_vars = list(X_copy.columns)
        self._fit_dtypes = X_copy.dtypes

        self._ordinal_encoder = OrdinalEncoderStart1()
        X_copy[self._categorical_vars] = self._ordinal_encoder.fit_transform(
            X_copy[self._categorical_vars]
        )

        if self.pretrained:
            self._embeddings_mapping = self.mapping_from_json()

        else:
            try:
                import tensorflow as tf
                from tensorflow.keras import layers, Model, callbacks
            except ImportError as error:
                raise Exception(
                    "Tensorflow not installed, Use 'pip install embedding-encoder[tf]'"
                ) from error

            self._layers_units = self.layers_units or [24, 12]
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
                embedding = layers.Embedding(
                    unique, dimension, name=f"embedding_{catvar}"
                )(categorical_input)
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
                model = Model(
                    inputs=[numeric_input] + categorical_inputs, outputs=output
                )
            else:
                model = Model(inputs=categorical_inputs[0], outputs=output)
            model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

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
            history = model.fit(
                x=merged_x,
                y=y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=self.validation_split,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor=monitor, patience=2, restore_best_weights=True
                    )
                ],
            )
            self._history = history.history
            if self.keep_model:
                self._model = model

            self._weights = {
                k: model.get_layer(f"embedding_{k}").weights
                for k in self._categorical_vars
            }
            self._embeddings_mapping = {
                k: pd.DataFrame(
                    self._weights[k][0].numpy(),
                    index=np.sort(np.append(X_copy[k].unique(), 0)),
                    columns=[
                        f"embedding_{k}_{i}"
                        for i in range(self._weights[k][0].shape[1])
                    ],
                )
                for k in self._categorical_vars
            }
            if self.mapping_path:
                self.mapping_to_json()

        columns_out = []
        for k in self._embeddings_mapping.values():
            columns_out.extend(k.columns)
        self._columns_out = columns_out

        return self

    def mapping_to_json(self) -> None:
        path = Path(self.mapping_path)
        json_mapping = {k: v.to_dict() for k, v in self._embeddings_mapping.items()}
        with open(path, "w") as f:
            json.dump(json_mapping, f)
        return

    def mapping_from_json(self) -> Dict[str, pd.DataFrame]:
        path = Path(self.mapping_path)
        with open(path, "r") as f:
            json_mapping = json.load(f)
        json_mapping = {k: pd.DataFrame.from_dict(v) for k, v in json_mapping.items()}
        for df in json_mapping.values():
            df.index = pd.to_numeric(df.index)
        return json_mapping

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
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
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.array(X)
        self._validate_data(X=X, dtype=None)
        if not X.shape[1] == len(self._categorical_vars) + len(self._numeric_vars):
            raise ValueError("X must have the same dimensions as used in training.")

        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(
                X_copy, columns=[f"cat{i}" for i in range(X_copy.shape[1])]
            )
        if not all(i in X_copy.columns for i in self._categorical_vars):
            raise ValueError("X must contain all categorical variables.")

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

        if isinstance(X, np.ndarray):
            final_embeddings = np.array(final_embeddings)

        return final_embeddings

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
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
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.array(X)
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(
                X_copy, columns=[f"cat{i}" for i in range(X_copy.shape[1])]
            )

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

        original = self._ordinal_encoder.inverse_transform(output)
        original = pd.DataFrame(original, columns=output.columns, index=X_copy.index)

        original = original.astype(dict(zip(original.columns, self._fit_dtypes)))
        if isinstance(X, np.ndarray):
            original = original.values
        return original

    def get_feature_names_out(self, input_features=None):
        return self._columns_out

    def get_feature_names(self, input_features=None):
        return self._columns_out
