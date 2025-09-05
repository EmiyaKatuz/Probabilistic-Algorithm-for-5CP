import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

SUMMER_MONTHS = [6,7,8,9]

def build_models(random_state=42):
    RF = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('clf', RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1))])
    ET = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('clf', ExtraTreesClassifier(n_estimators=800, random_state=random_state, n_jobs=-1))])
    return {'Random Forest': RF, 'Extra Trees': ET}


def fit_and_predict(models: dict, X_train, y_train, X_test):
    fitted = {}
    prob_test = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        fitted[name] = m
        if hasattr(m, "predict_proba"):
            s = m.predict_proba(X_test)[:,1]
        else:
            s = m.decision_function(X_test)
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        # sanitize to [0,1] and NaN-safe
        s = np.clip(np.nan_to_num(s, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        prob_test[name] = s
    return fitted, prob_test


def confusion_static_0p5(test_df: pd.DataFrame, probs: np.ndarray, y_col='is_CP'):
    d = test_df.copy()
    d['p'] = probs
    d = d.dropna(subset=['p', y_col]).copy()
    d[y_col] = d[y_col].astype(int)
    d = d[d['month'].isin(SUMMER_MONTHS)].copy()
    y_true = d[y_col].values
    y_pred = (d['p'].values >= 0.5).astype(int)
    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    return TP,FP,TN,FN


def confusion_dynamic_top4(test_df: pd.DataFrame, probs: np.ndarray, y_col='is_CP', min_days_per_month=20):
    """Per-month take Top-4 by probability as positive (replicates dynamic threshold evaluation)."""
    d = test_df.copy()
    d['p'] = probs
    d = d.dropna(subset=['p', y_col]).copy()
    d[y_col] = d[y_col].astype(int)
    d = d[d['month'].isin(SUMMER_MONTHS)].copy()
    # only months with enough days
    months_ok = [(y,m) for (y,m),sub in d.groupby(['year','month']) if len(sub)>=min_days_per_month]
    if not months_ok:
        return 0,0,0,0
    mask_ok = pd.Series(False, index=d.index)
    for (y,m) in months_ok:
        sub_idx = d[(d['year']==y)&(d['month']==m)].index
        k = min(4, len(sub_idx))
        top_idx = d.loc[sub_idx].sort_values('p', ascending=False).head(k).index
        mask_ok.loc[top_idx] = True
    y_true = d[y_col].values
    y_pred = mask_ok.values.astype(int)
    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    return TP,FP,TN,FN
