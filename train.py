import re
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from transformers import CoerceNumeric


def clean_price(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace("TL", "").replace("₺", "")
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def main():
    data_path = Path("data/hackathon_train_set.csv")
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, sep=";")
    df["Price_num"] = df["Price"].apply(clean_price)

    # Krediye uygun ilanlara odaklan
    df = df[df["Available for Loan"].eq("Yes")].copy()

    X = df.drop(columns=["Price", "Price_num"])
    y = np.log1p(df["Price_num"])

    # Kategorik sütunları belirle
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # Bazı sayısallar string geliyor olabilir -> numeric_like
    numeric_like = []
    for c in cat_cols:
        vals = X[c].dropna().astype(str).str.strip()
        if len(vals) and (vals.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.95):
            numeric_like.append(c)

    cat_cols = [c for c in cat_cols if c not in numeric_like]
    num_cols = [c for c in X.columns if c not in cat_cols] + numeric_like

    preprocess = Pipeline(
        steps=[
            ("coerce", CoerceNumeric(numeric_like)),
            (
                "ct",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_cols,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    rmse = float(mean_squared_error(y_test, pred, squared=False))
    r2 = float(r2_score(y_test, pred))

    joblib.dump(
        {
            "pipeline": pipe,
            "train_columns": list(X.columns),
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "metrics": {"rmse_log": rmse, "r2": r2},
        },
        out_dir / "model.joblib",
    )

    # UI seçimleri (çok uzunsa kırp)
    choices = {}
    for c in cat_cols:
        vc = X[c].fillna("Unknown").value_counts()
        top = vc.head(200).index.tolist()
        if "Unknown" not in top:
            top.append("Unknown")
        choices[c] = top
    (out_dir / "choices.json").write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")

    # Default satır şablonu
    template = {}
    for c in X.columns:
        if X[c].dtype == "object":
            template[c] = "Unknown"
        else:
            vals = pd.Series(X[c].dropna().unique())
            if len(vals) and set(vals.tolist()).issubset({0, 1, 0.0, 1.0}):
                template[c] = 0
            else:
                template[c] = None
    (out_dir / "row_template.json").write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Training done ✅")
    print(f"RMSE (log): {rmse:.4f}")
    print(f"R²       : {r2:.4f}")


if __name__ == "__main__":
    main()