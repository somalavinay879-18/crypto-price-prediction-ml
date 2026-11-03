"""Models compared in the project, plus the metrics used to compare them."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> dict:
    """Name -> unfitted model. The majority-class model is the floor every other model must beat."""
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=50,
            random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1, eval_metric="logloss",
        ),
    }


def evaluate(model, x_test, y_test) -> dict:
    """Accuracy, balanced accuracy, F1 and ROC-AUC on the held-out test set."""
    preds = model.predict(x_test)
    auc = np.nan
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, proba)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": auc,
    }
