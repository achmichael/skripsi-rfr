import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from core.feature_engineer import FeatureEngineer
from utils.config import config


class Preprocessor:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.target_column = config["target"][dataset_type]
        self.numeric_columns = set(config["numeric_features"].get(dataset_type, []))
        self.numeric_columns.add(self.target_column)
        self.categorical_columns = config["categorical_features"].get(dataset_type, [])
        self.unused_columns = config["unused_features"].get(dataset_type, [])
        self.numeric_fill_values = {}
        self.categorical_fill_values = {}
        self.outlier_bounds = {}
        self.columns_after_encoding = None

    def fit(self, df):
        prepared = self._prepare_base(df)
        self._fit_missing_values(prepared)
        prepared = self._apply_missing_values(prepared)
        self._fit_outlier_bounds(prepared)
        prepared = self._apply_outlier_bounds(prepared)
        encoded = self._one_hot_encode(prepared)
        self.columns_after_encoding = list(encoded.columns)
        return self

    def transform(self, df):
        if self.columns_after_encoding is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        prepared = self._prepare_base(df)
        prepared = self._apply_missing_values(prepared)
        prepared = self._apply_outlier_bounds(prepared)
        encoded = self._one_hot_encode(prepared)
        encoded = encoded.reindex(columns=self.columns_after_encoding, fill_value=0)
        self._validate_numeric(encoded)
        return encoded

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def _prepare_base(self, df):
        prepared = df.copy()
        existing_unused_cols = [col for col in self.unused_columns if col in prepared.columns]
        prepared = prepared.drop(columns=existing_unused_cols)
        prepared = self._convert_numeric_columns(prepared)
        prepared = FeatureEngineer(prepared, dataset_type=self.dataset_type).engineer_features()
        return prepared

    def _convert_numeric_columns(self, df):
        converted = df.copy()

        for col in converted.columns:
            if col not in self.numeric_columns:
                continue

            converted[col] = (
                converted[col]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            converted[col] = pd.to_numeric(converted[col], errors="coerce")

        return converted

    def _fit_missing_values(self, df):
        for col in df.columns:
            if col == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if config["data_preprocessing"]["numeric_missing_strategy"] == "mean":
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()

                self.numeric_fill_values[col] = 0 if pd.isna(fill_value) else fill_value
            elif is_object_dtype(df[col]) or is_string_dtype(df[col]):
                mode = df[col].mode(dropna=True)
                self.categorical_fill_values[col] = "Unknown" if mode.empty else mode.iloc[0]

        if self.target_column in df.columns and pd.api.types.is_numeric_dtype(df[self.target_column]):
            target_fill = df[self.target_column].median()
            self.numeric_fill_values[self.target_column] = 0 if pd.isna(target_fill) else target_fill

    def _apply_missing_values(self, df):
        filled = df.copy()

        for col, value in self.numeric_fill_values.items():
            if col in filled.columns:
                filled[col] = filled[col].fillna(value)

        for col, value in self.categorical_fill_values.items():
            if col in filled.columns:
                filled[col] = filled[col].fillna(value)

        return filled

    def _fit_outlier_bounds(self, df):
        if not config["data_preprocessing"]["handle_outliers"]:
            return

        if config["data_preprocessing"]["outlier_method"] != "iqr":
            return

        clip_target = config["data_preprocessing"].get("clip_target", False)

        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            if col == self.target_column and not clip_target:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            self.outlier_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    def _apply_outlier_bounds(self, df):
        clipped = df.copy()

        for col, bounds in self.outlier_bounds.items():
            if col in clipped.columns:
                lower_bound, upper_bound = bounds
                clipped[col] = clipped[col].clip(lower_bound, upper_bound)

        return clipped

    def _one_hot_encode(self, df):
        existing_cols = [
            col
            for col in self.categorical_columns
            if col in df.columns and (is_object_dtype(df[col]) or is_string_dtype(df[col]))
        ]

        return pd.get_dummies(df, columns=existing_cols, drop_first=False, dtype=int)

    def _validate_numeric(self, df):
        non_numeric_cols = df.select_dtypes(include=["object", "string"]).columns

        if len(non_numeric_cols) > 0:
            raise ValueError(f"Non-numeric columns remain after preprocessing: {list(non_numeric_cols)}")
