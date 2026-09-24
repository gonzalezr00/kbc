import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FeatureBuilder, FlowImputer, NON_FEATURES

PRODUCTS = ["MF", "CC", "CL"]


def make_model(kind, flow_fill="zero", seed=12):
    if kind == "logit":
        est = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    else:
        est = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    return make_pipeline(FlowImputer(flow_fill), FeatureBuilder(), est)


def lift_at(y, p, frac=0.15):
    idx = np.argsort(-p)[: int(len(y) * frac)]
    return y[idx].mean() / y.mean()


def cv_eval(master, product, kind, flow_fill="zero", n_splits=5, seed=12):
    lab = master[master.is_labeled]
    X, y = lab.drop(columns=NON_FEATURES), lab[f"Sale_{product}"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(
        make_model(kind, flow_fill, seed), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    return oof, {
        "base_rate": y.mean(),
        "roc_auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "brier": brier_score_loss(y, oof),
        "lift@15%": lift_at(y, oof),
    }
