import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_string_dtype

from src.core.feature_engineer import FeatureEngineer
from src.utils.config import config


class Preprocessor:
    NO_USAGE_CATEGORY_BY_DEVICE = {
        "Kulkas": "Tidak ada",
        "TV": "Tidak ada / tidak digunakan",
        "AC": "Tidak ada / tidak digunakan",
        "Kipas": "Tidak ada / tidak digunakan",
        "RiceCooker": "Tidak ada / tidak digunakan",
        "MesinCuci": "Tidak ada / tidak digunakan",
    }

    DEVICE_DEPENDENT_SUFFIXES = [
        "_EstimasiWattPerUnit",
        "_EstimasiJamPerHari",
        "_EstimasiFrekuensiPerMinggu",
        "_EstimasiDurasiSekaliPakaiJam",
        "_Energi_WhPerHari",
        "_Energi_kWhPerHari",
    ]

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
        prepared = df.copy()
        
        # 1. Drop unused columns first
        existing_unused_cols = [col for col in self.unused_columns if col in prepared.columns]
        prepared = prepared.drop(columns=existing_unused_cols)
        
        # 2. Convert data types appropriately (replace text strings to NaN before imputation)
        prepared = self._convert_numeric_columns(prepared)
        
        # 3. Handle missing values (calculation/fitting first)
        self._fit_missing_values(prepared)
        prepared = self._apply_missing_values(prepared)
        prepared = self._normalize_device_consistency(prepared)
        
        # 4. Do Feature Engineering using the clean dataframe (now completely free of NaN)
        prepared = FeatureEngineer(prepared, dataset_type=self.dataset_type).engineer_features()
        
        # 5. Fit outlier bounds (if applicable)
        self._fit_outlier_bounds(prepared)
        prepared = self._apply_outlier_bounds(prepared)
        prepared = self._normalize_device_consistency(prepared)
        
        # 6. Fit One-hot Encoding
        encoded = self._one_hot_encode(prepared)
        self.columns_after_encoding = list(encoded.columns)
        return self

    def transform(self, df):
        if self.columns_after_encoding is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        prepared = df.copy()
        
        existing_unused_cols = [col for col in self.unused_columns if col in prepared.columns]
        prepared = prepared.drop(columns=existing_unused_cols)
        
        prepared = self._convert_numeric_columns(prepared)
        
        prepared = self._apply_missing_values(prepared)
        prepared = self._normalize_device_consistency(prepared)
        
        prepared = FeatureEngineer(prepared, dataset_type=self.dataset_type).engineer_features()
        
        prepared = self._apply_outlier_bounds(prepared)
        prepared = self._normalize_device_consistency(prepared)
        
        encoded = self._one_hot_encode(prepared)
        encoded = encoded.reindex(columns=self.columns_after_encoding, fill_value=0)
        self._validate_numeric(encoded)
        return encoded

    def clean_raw_dataset(self, df, dataset_type):
        cleaned = df.copy()
        target_column = config["target"][dataset_type]
        cleaning_config = config.get("target_cleaning", {}).get(dataset_type, {})

        if target_column in cleaned.columns:
            cleaned[target_column] = (
                cleaned[target_column]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
            cleaned = cleaned.dropna(subset=[target_column]).reset_index(drop=True)

        if dataset_type == "pascabayar" and cleaning_config.get("enabled", False):
            threshold = cleaning_config.get("small_bill_threshold", 1000)
            multiplier = cleaning_config.get("small_bill_multiplier", 1000)
            small_bill_mask = (
                cleaned[target_column].gt(0) &
                cleaned[target_column].lt(threshold)
            )
            corrected_count = int(small_bill_mask.sum())

            if corrected_count > 0:
                cleaned.loc[small_bill_mask, target_column] = (
                    cleaned.loc[small_bill_mask, target_column] * multiplier
                )
                print(f"Corrected {corrected_count} small pascabayar bill values.")

        return cleaned
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def _convert_numeric_columns(self, df):
        converted = df.copy()

        # Custom logic for "Tidak diisi" and "Tidak tahu" strings before coercion
        converted.replace(["Tidak diisi", "Tidak tahu"], pd.NA, inplace=True)

        for col in converted.columns:
            if col not in self.numeric_columns:
                continue

            # Strip non-numeric characters for valid conversion and force empty to NA
            converted[col] = (
                converted[col]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            # Convert to numeric, unparseable values become NaN
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

    def _normalize_device_consistency(self, df):
        normalized = df.copy()

        for device, no_usage_category in self.NO_USAGE_CATEGORY_BY_DEVICE.items():
            jumlah_col = f"{device}_Jumlah"
            kategori_col = f"{device}_Kategori"

            if jumlah_col not in normalized.columns:
                continue

            jumlah = pd.to_numeric(normalized[jumlah_col], errors="coerce")
            inactive_mask = jumlah.le(0).fillna(False)

            if not inactive_mask.any():
                continue

            normalized.loc[inactive_mask, jumlah_col] = 0

            if kategori_col in normalized.columns:
                normalized.loc[inactive_mask, kategori_col] = no_usage_category

            for suffix in self.DEVICE_DEPENDENT_SUFFIXES:
                col = f"{device}{suffix}"
                if col in normalized.columns:
                    normalized.loc[inactive_mask, col] = 0

        if "Alat_Lain_Ada" in normalized.columns:
            no_other_device_mask = normalized["Alat_Lain_Ada"].astype(str).str.lower().eq("tidak")

            if no_other_device_mask.any():
                for i in range(1, 4):
                    prefix = f"Alat_Lain_{i}"

                    for col in [f"{prefix}_Jenis", f"{prefix}_Kategori"]:
                        if col in normalized.columns:
                            normalized.loc[no_other_device_mask, col] = "Tidak diisi"

                    numeric_cols = [
                        col for col in normalized.columns
                        if col.startswith(prefix) and ("Estimasi" in col or "Energi" in col)
                    ]
                    for col in numeric_cols:
                        normalized.loc[no_other_device_mask, col] = 0

        return normalized

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

        missing_cols = df.columns[df.isna().any()]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing values remain after preprocessing: {list(missing_cols)}")

        numeric = df.select_dtypes(include=["number"])
        non_finite_mask = ~np.isfinite(numeric.to_numpy()).all(axis=0)
        non_finite_cols = numeric.columns[non_finite_mask]
        if len(non_finite_cols) > 0:
            raise ValueError(f"Non-finite numeric values remain after preprocessing: {list(non_finite_cols)}")
