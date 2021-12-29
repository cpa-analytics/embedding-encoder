import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, TransformerMixin


class DenseFeatureMixer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x_dataset, y_dataset, num_columns, cat_columns, dim):
        cat_columns.sort()
        self.cat_columns = cat_columns
        feature_columns = []
        for i in num_columns:
            feature_columns.append(tf.feature_column.numeric_column(i))

        uniques = {col: x_dataset[col].unique() for col in cat_columns}

        uniques_idx = {
            col: x_dataset[col].drop_duplicates().index for col in cat_columns
        }

        cat_vocab_dict = {
            i: tf.feature_column.categorical_column_with_vocabulary_list(
                i, uniques.get(i)
            )
            for i in cat_columns
        }

        embeddings_dict = {
            col: tf.feature_column.embedding_column(
                cat_vocab_dict.get(col), dimension=dim
            )
            for col in cat_columns
        }

        for i in embeddings_dict:
            feature_columns.append(embeddings_dict.get(i))

        self.feature_layer = layers.DenseFeatures(feature_columns)

        y_dataset = pd.DataFrame(y_dataset)

        trainset = tf.data.Dataset.from_tensor_slices(
            (dict(x_dataset), dict(y_dataset))
        ).batch(32)

        self.model = tf.keras.models.Sequential()
        self.model.add(self.feature_layer)
        self.model.add(layers.Dense(units=512, activation="relu"))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Dense(units=1))
        self.model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.model.fit(trainset, epochs=20, verbose=2)

        weights = [self.model.get_weights()[i] for i in range(len(self.model.weights))]

        mapped_weights = dict(zip(cat_columns, weights))

        self.frames = {
            cat_columns[k]: [
                str(cat_columns[k]) + "_embedding_" + str(i) for i in range(2)
            ]
            for k in range(len(cat_columns))
        }

        frames_col_names = {
            cat_columns[k]: [
                str(cat_columns[k]) + "_embedding_" + str(i) for i in range(2)
            ]
            for k in range(len(cat_columns))
        }

        for i in weights:
            frames_content = [
                pd.DataFrame(
                    mapped_weights.get(i),
                    columns=frames_col_names.get(i),
                    index=uniques_idx.get(i),
                )
                for i in mapped_weights
            ]

        mapped_frames = dict(zip(cat_columns, frames_content))

        self.datasets = [
            mapped_frames.get(i).join(x_dataset[i], how="left") for i in mapped_frames
        ]

        return self

    def transform(self, x_dataset):

        final_datasets = [
            x_dataset.merge(
                self.datasets[i], how="inner", on=self.cat_columns[i], copy=False
            )
            for i in range(len(self.datasets))
        ]

        for i in range(len(final_datasets)):
            final_datasets[i].drop(x_dataset.columns, axis=1, inplace=True)

        for i in range(len(final_datasets)):
            x_dataset = x_dataset.merge(
                final_datasets[i], left_index=True, right_index=True, how="inner"
            )

        return x_dataset
