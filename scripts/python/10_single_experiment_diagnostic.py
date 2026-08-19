import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
import lightgbm as lgb

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

train = pd.read_csv(PROCESSED / "split_train.csv")
y_train = train["label"].values

results = []
for exp in ["alpha", "cdc15", "cdc28", "elu"]:
    cols = [c for c in train.columns if c.startswith(f"{exp}_")]
    X = train[cols]

    pos_ratio = y_train.mean()
    spw = (1 - pos_ratio) / pos_ratio
    model = lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=spw, random_state=42, verbose=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pred = (oof >= 0.5).astype(int)

    results.append({
        "Deney": exp,
        "n_ozellik": len(cols),
        "AUC": roc_auc_score(y_train, oof),
        "F1_M": f1_score(y_train, pred, average="macro"),
        "MCC": matthews_corrcoef(y_train, pred),
        "Etikette_kullanildi_mi": exp != "elu",
    })

df = pd.DataFrame(results).sort_values("AUC", ascending=False)
df.to_csv(RESULTS / "single_experiment_diagnostic.csv", index=False)
print(df.to_string(index=False))