import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .mc_engine import build_base_df, scenario_probabilities_summer_topk, evaluate_confusion_4cp

FORECAST_MAP = {'ECA': 'ECA: IESO-Ontario Demand Historic Forecast',
                'RTO': 'RTO: IESO-Ontario Demand Historic Forecast',
                'TESLA': 'TESLA: IESO-Ontario Demand Historic Forecast'}


def horizon_probs(paths_by_h: Dict[int, str], source: str, n_sims: int = 2000) -> Dict[int, pd.DataFrame]:
    """Compute per-horizon probability DataFrames for a given forecast source (ECA/RTO/TESLA)."""
    fcast_col = FORECAST_MAP[source]
    out: Dict[int, pd.DataFrame] = {}
    for h, path in sorted(paths_by_h.items()):
        df_base = build_base_df(path, fcast_col)
        prob = scenario_probabilities_summer_topk(df_base, n_sims=n_sims, k=4, return_peak_hour=False)
        out[h] = prob.sort_index()
    return out


def inverse_brier_weights(prob_by_h: Dict[int, pd.DataFrame], labels_df: pd.DataFrame, kind: str) -> Dict[int, float]:
    """Compute inverse-Brier weights over the intersection period with labels."""
    from sklearn.metrics import brier_score_loss
    scores: Dict[int, float] = {}
    for h, prob in prob_by_h.items():
        # --- FIX: turn index 'date' into a real column to avoid ambiguity ---
        merged = prob[kind].rename('p').rename_axis('date').reset_index()
        lab = labels_df[['timestamp', 'is_CP']].copy()
        lab['date'] = pd.to_datetime(lab['timestamp']).dt.floor('D')

        d = pd.merge(merged, lab[['date', 'is_CP']], on='date', how='left')
        d = d.dropna(subset=['p', 'is_CP']).copy()
        if d.empty:
            continue

        y = d['is_CP'].astype(int).values
        p = d['p'].astype(float).values
        try:
            b = brier_score_loss(y, p)
        except Exception:
            b = np.nan
        if np.isfinite(b) and b > 0:
            scores[h] = b
    if not scores:
        H = list(prob_by_h.keys())
        return {h: 1.0 / max(len(H), 1) for h in H}
    inv = {h: 1.0 / s for h, s in scores.items()}
    ssum = sum(inv.values())
    return {h: v / ssum for h, v in inv.items()}


def weighted_fusion(prob_by_h: Dict[int, pd.DataFrame], weights: Dict[int, float], kind: str) -> pd.DataFrame:
    """Fuse per-horizon probabilities using learned weights."""
    cols = []
    hs = sorted(prob_by_h.keys())
    for h in hs:
        s = prob_by_h[h][kind].rename(f"h{h}")
        cols.append(s)
    M = pd.concat(cols, axis=1).sort_index()  # 行：date
    w = np.array([weights.get(h, 0.0) for h in hs], dtype=float)
    if w.sum() <= 0:
        w[:] = 1.0 / len(hs)
    else:
        w /= w.sum()
    fused_vals = np.nansum(M.values * w, axis=1)
    fused = pd.Series(fused_vals, index=M.index, name=kind).to_frame()
    return fused
