import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class ColumnTransformerWithNames(ColumnTransformer):
    """A ColumnTransformer that retains DataFrame column names. Obtained from
    https://stackoverflow.com/questions/61079602/how-do-i-get-feature-names-using-a-column-transformer/68671424#68671424
    """

    def get_feature_names(self):
        """Get feature names from all transformers.

        Returns
        -------
        feature_names : List[str]
            Names of the features produced by transform.
        """

        # Turn loopkup into function for better handling with pipeline later
        def get_names(trans):
            # >> Original get_feature_names() method
            if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
                return []
            if trans == "passthrough":
                if hasattr(self, "_df_columns"):
                    if (not isinstance(column, slice)) and all(
                        isinstance(col, str) for col in column
                    ):
                        return column
                    else:
                        return self._df_columns[column]
                else:
                    indices = np.arange(self._n_features)
                    return ["x%d" % i for i in indices[column]]
            if not hasattr(trans, "get_feature_names"):
                if column is None:
                    return []
                else:
                    return [f for f in column]

            return [f for f in trans.get_feature_names()]

        feature_names = []
        # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
        if type(self) == Pipeline:
            l_transformers = [
                (name, trans, None, None) for _, name, trans in self._iter()
            ]
        else:
            # For column transformers, follow the original method
            l_transformers = list(self._iter(fitted=True))

        for _, trans, column, _ in l_transformers:
            if type(trans) == Pipeline:
                # Recursive call on pipeline
                _names = self._get_feature_names_with_transformer(trans)
                # if pipeline has no transformer that returns names
                if len(_names) == 0:
                    _names = [f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans))

        return feature_names

    def transform(self, X):
        indices = X.index.values.tolist()
        X_mat = super().transform(X)
        new_cols = self.get_feature_names()
        new_X = pd.DataFrame(X_mat, index=indices, columns=new_cols)
        return new_X

    def fit_transform(self, X, y=None):
        super().fit_transform(X, y)
        return self.transform(X)

    def _get_feature_names_with_transformer(self, column_transformer):
        def get_names(trans):
            if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
                return []
            if trans == "passthrough":
                if hasattr(column_transformer, "_df_columns"):
                    if (not isinstance(column, slice)) and all(
                        isinstance(col, str) for col in column
                    ):
                        return column
                    else:
                        return column_transformer._df_columns[column]
                else:
                    indices = np.arange(column_transformer._n_features)
                    return ["x%d" % i for i in indices[column]]
            if not hasattr(trans, "get_feature_names"):
                if column is None:
                    return []
                else:
                    return [f for f in column]

            return [f for f in trans.get_feature_names()]

        feature_names = []
        if type(column_transformer) == Pipeline:
            l_transformers = [
                (name, trans, None, None)
                for _, name, trans in column_transformer._iter()
            ]
        else:
            l_transformers = list(column_transformer._iter(fitted=True))

        for _, trans, column, _ in l_transformers:
            if type(trans) == Pipeline:
                _names = self._get_feature_names_with_transformer(trans)
                if len(_names) == 0:
                    _names = [f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans))

        return feature_names
