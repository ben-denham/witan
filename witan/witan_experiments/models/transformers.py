from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import _VectorizerMixin
from typing import List, Dict, Optional, cast


class DataFrameTransformer(TransformerMixin, BaseEstimator, ABC):
    """Abstract base class for transformers that consume and return pandas
    DataFrames."""

    def fit(self,
            df_X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> 'DataFrameTransformer':
        return self

    @abstractmethod
    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        pass


class SelectFeatureSubsetDataFrameTransformer(DataFrameTransformer):
    """Transforms a DataFrame by selecting only the given feature_names."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        super().__init__()

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        return df_X[self.feature_names]


class TransformFeatureSubsetDataFrameTransformer(DataFrameTransformer):
    """Applies the given transformer to only the named features in the
    DataFrame (column order may change)."""

    def __init__(self,
                 feature_names: List[str],
                 transformer: DataFrameTransformer):
        self.feature_names = feature_names
        self.transformer = transformer
        super().__init__()

    def fit(self,
            df_X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> 'TransformFeatureSubsetDataFrameTransformer':
        super().fit(df_X, y)
        if len(self.feature_names) > 0:
            self.transformer.fit(df_X[self.feature_names], y)
        return self

    def reconcat(self,
                 df_A: pd.DataFrame,
                 df_B: pd.DataFrame,
                 index: pd.Index) -> pd.DataFrame:
        """As the transformation may have altered the index of one subset of
        the DataFrame, we reset the index on both, join them, and then
        set the index back."""
        dfs = [df.reset_index(drop=True) for df in [df_A, df_B]]
        df_concat = pd.concat(dfs, axis=1)
        df_concat.index = index
        return df_concat

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        # Short-circuit if the subset of features is empty.
        if len(self.feature_names) < 1:
            return df_X

        transformed_df = self.transformer.transform(df_X[self.feature_names])
        rest_df = df_X.drop(self.feature_names, axis=1)
        return self.reconcat(rest_df, transformed_df, index=df_X.index)

    def fit_transform(self,
                      df_X: pd.DataFrame,
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        # Short-circuit if the subset of features is empty.
        if len(self.feature_names) < 1:
            return df_X

        transformed_df = self.transformer.fit_transform(df_X[self.feature_names], y)
        rest_df = df_X.drop(self.feature_names, axis=1)
        return self.reconcat(rest_df, transformed_df, index=df_X.index)


class DataFrameVectorizer(DataFrameTransformer):
    """Apply a given vectorizer to each column in the provided DataFrame,
    but return a sparse DataFrame with meaningful column names."""

    def __init__(self,
                 vectorizer: _VectorizerMixin,
                 column_prefixes: Optional[Dict[str, str]] = None):
        self.vectorizer = vectorizer
        self.column_prefixes = {} if column_prefixes is None else column_prefixes

    def fit(self,
            df_X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> 'DataFrameVectorizer':
        super().fit(df_X, y=y)
        self.vectorizers = {}
        for column_name, column_series in df_X.iteritems():
            self.vectorizers[column_name] = clone(self.vectorizer)
            self.vectorizers[column_name].fit(column_series, y)
        return self

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        column_prefixes = {
            column_name: self.column_prefixes.get(column_name, column_name)
            for column_name in df_X.columns
        }
        # column_prefixes should not have duplicates
        assert len(set(column_prefixes.values())) == len(list(column_prefixes.values()))

        result_dfs = []
        for column_name, column_series in df_X.iteritems():
            result_sparse_array = self.vectorizers[column_name].transform(column_series)
            result_df = pd.DataFrame.sparse.from_spmatrix(result_sparse_array)
            col_prefix = ('' if column_prefixes[column_name] is None
                          else f'{column_prefixes[column_name]}__')
            result_df.columns = [
                f'{col_prefix}{feature}'
                for feature in self.vectorizers[column_name].get_feature_names()
            ]
            result_dfs.append(result_df)
        return pd.concat(result_dfs, axis=1)


class DataFrameDtypeTransformer(DataFrameTransformer):
    """Convert a DataFrame to have a single dtype (useful before explicit
    or implicit conversion to numpy). If exclude_columns is provided,
    then the dtypes of those columns will not be changed."""

    def __init__(self,
                 dtype: np.dtype,
                 exclude_columns: Optional[List[str]] = None):
        self.dtype = dtype
        self.exclude_columns = exclude_columns

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        columns = set(df_X.columns)
        if self.exclude_columns is not None:
            columns -= set(self.exclude_columns)
        column_dtypes = {col: self.dtype for col in columns}
        return cast(pd.DataFrame, df_X.astype(column_dtypes))


class SparseMatrixTransformer(DataFrameTransformer):
    """Convert a sparse DataFrame to a sparse numpy array."""

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        return df_X.sparse.to_coo().tocsr()


class DenseTransformer(TransformerMixin, BaseEstimator):
    """Convert a numpy array to a dense representation."""

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None) -> 'DenseTransformer':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.todense()
