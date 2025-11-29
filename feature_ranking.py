"""
Quick feature-screening script for MathScore using the provided X_train.csv / y_train.csv.
Runs numeric correlation and permutation importance from a RandomForestRegressor.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    X = pd.read_csv("X_train.csv")
    y = pd.read_csv("y_train.csv")["MathScore"]

    id_cols = ["Unnamed: 0", "CNTSTUID", "CNTRYID", "CNTSCHID", "SUBNATIO"]
    X = X.drop(columns=[c for c in id_cols if c in X.columns])

    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]

    if not num_cols:
        raise ValueError("No numeric columns detected.")

    print("Top 20 numeric correlations with MathScore:")
    corr = X[num_cols].corrwith(y)
    print(corr.abs().sort_values(ascending=False).head(20))
    print("\nTraining permutation-importance model...")

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preproc = ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)], remainder="drop"
    )
    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    pipe = Pipeline([("prep", preproc), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    perm = permutation_importance(
        pipe, X_val, y_val, n_repeats=3, random_state=42, n_jobs=-1
    )
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    top = (
        pd.Series(perm.importances_mean, index=feature_names)
        .sort_values(ascending=False)
        .head(30)
    )
    print("\nTop 30 features by permutation importance on held-out set:")
    print(top)


if __name__ == "__main__":
    main()
