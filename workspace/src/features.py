import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PRODUCTS = ["SA", "MF", "OVD", "CC", "CL"]
FLOW_COLS = [
    "VolumeCred",
    "VolumeCred_CA",
    "TransactionsCred",
    "TransactionsCred_CA",
    "VolumeDeb",
    "VolumeDeb_CA",
    "VolumeDebCash_Card",
    "VolumeDebCashless_Card",
    "VolumeDeb_PaymentOrder",
    "TransactionsDeb",
    "TransactionsDeb_CA",
    "TransactionsDebCash_Card",
    "TransactionsDebCashless_Card",
    "TransactionsDeb_PaymentOrder",
]
BAL_COLS = [f"ActBal_{p}" for p in ["CA", *PRODUCTS]]
NON_FEATURES = ["Client", "is_labeled"] + [
    f"{k}_{p}" for k in ("Sale", "Revenue") for p in ("MF", "CC", "CL")
]


def slog(x):
    """Signed log1p: safe for skewed values that may be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def sdiv(a, b):
    """Safe division: 0 where the denominator is 0."""
    return (a / b.where(b > 0)).fillna(0)


class FlowImputer(BaseEstimator, TransformerMixin):
    """mode='zero' (base case) or 'median' (sensitivity); medians fit on train folds only."""

    def __init__(self, mode="zero"):
        self.mode = mode

    def fit(self, X, y=None):
        self.fill_ = (
            X[FLOW_COLS].median()
            if self.mode == "median"
            else pd.Series(0.0, index=FLOW_COLS)
        )
        return self

    def transform(self, X):
        X = X.copy()
        X[FLOW_COLS] = X[FLOW_COLS].fillna(self.fill_)
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless feature engineering; expects flows already imputed."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        f = pd.DataFrame(index=X.index)
        f["age"] = X.Age
        f["tenure_years"] = X.Tenure / 12
        f["sex_M"] = (X.Sex == "M").astype(
            int
        )  # 3 UNK rows fall in the non-M group (immaterial)
        f["no_flow_data"] = X.no_flow_data.astype(int)

        has = [f"has_{p}" for p in PRODUCTS]
        f[["Count_CA"] + [f"Count_{p}" for p in PRODUCTS] + has] = X[
            ["Count_CA"] + [f"Count_{p}" for p in PRODUCTS] + has
        ]
        f["n_products"] = X[has].sum(axis=1)

        for c in BAL_COLS + FLOW_COLS:
            f[f"log_{c}"] = slog(X[c])
        f["log_assets"] = slog(X[["ActBal_CA", "ActBal_SA", "ActBal_MF"]].sum(axis=1))
        f["log_liabilities"] = slog(
            X[["ActBal_OVD", "ActBal_CC", "ActBal_CL"]].sum(axis=1)
        )

        f["net_flow"] = slog(X.VolumeCred - X.VolumeDeb)
        f["cashless_share"] = sdiv(X.VolumeDebCashless_Card, X.VolumeDeb)
        f["cash_share"] = sdiv(X.VolumeDebCash_Card, X.VolumeDeb)
        f["payorder_share"] = sdiv(X.VolumeDeb_PaymentOrder, X.VolumeDeb)
        f["ca_credit_share"] = sdiv(X.VolumeCred_CA, X.VolumeCred)
        f["avg_ticket_deb"] = slog(sdiv(X.VolumeDeb, X.TransactionsDeb))
        f["avg_ticket_cred"] = slog(sdiv(X.VolumeCred, X.TransactionsCred))
        return f
