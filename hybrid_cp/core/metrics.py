import numpy as np
import pandas as pd

def confusion_from_probs(prob: pd.Series, labels_df: pd.DataFrame) -> tuple:
    df = pd.DataFrame({'p': prob}).copy()
    df['date'] = df.index
    lab = labels_df[['timestamp','is_CP']].copy()
    lab['date'] = lab['timestamp'].dt.floor('D')
    d = pd.merge(df, lab[['date','is_CP']], on='date', how='left').dropna(subset=['p','is_CP'])
    d = d[d['date'].dt.month.isin([6,7,8,9])].copy()
    y_true = d['is_CP'].astype(int).values
    y_pred = (d['p'].values >= 0.5).astype(int)
    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    return TP,FP,TN,FN
