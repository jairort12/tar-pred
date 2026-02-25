import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        use_log_ch4=True,
        use_tred_er_inter=True,
        use_er_sq=True,
        use_gas_ratios=False,
        use_complex_inter=False,
    ):
        self.use_log_ch4 = use_log_ch4
        self.use_tred_er_inter = use_tred_er_inter
        self.use_er_sq = use_er_sq
        self.use_gas_ratios = use_gas_ratios
        self.use_complex_inter = use_complex_inter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        epsilon = 1e-6  # prevent division by zero

        # Ensure 'BM_Cluster_Final' is categorical if present
        if "BM_Cluster_Final" in X_copy.columns and X_copy["BM_Cluster_Final"].dtype != "category":
            X_copy["BM_Cluster_Final"] = X_copy["BM_Cluster_Final"].astype("category")

        numerical_cols = X_copy.select_dtypes(include=np.number).columns
        categorical_cols = X_copy.select_dtypes(include="category").columns

        X_num = X_copy[numerical_cols].copy()
        X_cat = X_copy[categorical_cols].copy()

        if self.use_log_ch4 and "CH4" in X_num.columns:
            X_num["log_CH4"] = np.log1p(np.maximum(X_num["CH4"], 0))

        if self.use_tred_er_inter and "Tred" in X_num.columns and "ER" in X_num.columns:
            X_num["Tred_ER"] = X_num["Tred"] * X_num["ER"]

        if self.use_er_sq and "ER" in X_num.columns:
            X_num["ER_sq"] = X_num["ER"] ** 2

        if self.use_gas_ratios:
            if "CO2" in X_num.columns and "CO" in X_num.columns:
                X_num["CO2_CO_ratio"] = X_num["CO2"] / (X_num["CO"] + epsilon)
            if "H2" in X_num.columns and "CO" in X_num.columns:
                X_num["H2_CO_ratio"] = X_num["H2"] / (X_num["CO"] + epsilon)

        if self.use_complex_inter and "VM" in X_num.columns and "Tred" in X_num.columns:
            X_num["VM_Tred"] = X_num["VM"] * X_num["Tred"]

        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)
        return pd.concat([X_num, X_cat], axis=1)