import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")  # Suppress Pandas warnings

# Load Data
train = pd.read_csv("/kaggle/input/competitions/mlx-2-0-regression/train.csv")
test = pd.read_csv("/kaggle/input/competitions/mlx-2-0-regression/test.csv")
sample_submission = pd.read_csv("/kaggle/input/competitions/mlx-2-0-regression/sample_submission.csv")

# Data Preprocessing
train = train.drop_duplicates()

num_cols = train.select_dtypes(include=["float64", "int64"]).columns.drop("target")
cat_cols = train.select_dtypes(include="object").columns.tolist()

train[num_cols] = train[num_cols].fillna(train[num_cols].mean())
test[num_cols] = test[num_cols].fillna(train[num_cols].mean())

for col in cat_cols:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)


def process_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["publication_timestamp"] = pd.to_datetime(df["publication_timestamp"], errors="coerce")
    df["year"] = df["publication_timestamp"].dt.year
    df["month"] = df["publication_timestamp"].dt.month
    df["day"] = df["publication_timestamp"].dt.day
    df["dayofweek"] = df["publication_timestamp"].dt.dayofweek
    df.drop(columns="publication_timestamp", inplace=True)
    return df


train = process_dates(train)
test = process_dates(test)


# Feature Engineering
def eng_features(df: pd.DataFrame) -> pd.DataFrame:
    bases = [
        "duration_ms",
        "intensity_index",
        "organic_texture",
        "vocal_presence",
        "groove_efficiency",
        "emotional_charge",
        "emotional_resonance",
        "performance_authenticity",
        "instrumental_density",
        "beat_frequency",
        "rhythmic_cohesion",
        "organic_immersion",
    ]

    for base in bases:
        cols = [f"{base}_{i}" for i in range(3)]
        df[f"{base}_avg"] = df[cols].mean(axis=1)
        df[f"{base}_var"] = df[cols].var(axis=1)

    df["duration_x_intensity"] = df["duration_ms_avg"] * df["intensity_index_avg"]
    df["texture_x_groove"] = df["organic_texture_avg"] * df["groove_efficiency_avg"]
    return df


train = eng_features(train)
test = eng_features(test)


# Encoding
def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in cols:
        common = set(train_df[col].unique()) & set(test_df[col].unique())
        train_df[col] = train_df[col].apply(lambda x: x if x in common else "Other")
        test_df[col] = test_df[col].apply(lambda x: x if x in common else "Other")

        for df_ in [train_df, test_df]:
            freq = df_[col].value_counts()
            rare = freq[freq <= 5].index
            df_[col] = df_[col].apply(lambda x: "Rare" if x in rare else x)

        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    return train_df, test_df


cols_to_encode = ["composition_label_0", "composition_label_1", "composition_label_2", "creator_collective"]
train, test = encode_categoricals(train, test, cols_to_encode)

simple_cat_cols = ["weekday_of_release", "season_of_release", "lunar_phase"]
for col in simple_cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train.drop(columns=["target", "track_identifier", "id"], errors="ignore")
y = train["target"]
X_test = test.drop(columns=["track_identifier", "id"], errors="ignore")


# Evaluation Metrics
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSLE": np.sqrt(mean_squared_log_error(y_true, y_pred)),
    }


# K-Fold
def cross_val_predict(
    X_data: pd.DataFrame,
    y_data: pd.Series,
    X_test_data: pd.DataFrame,
    model_func,
    scale: bool = False,
    n_splits: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    oof_preds = np.zeros(len(X_data))
    test_preds = np.zeros(len(X_test_data))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

        if scale:
            pipe = make_pipeline(StandardScaler(), model_func())
        else:
            pipe = model_func()

        pipe.fit(X_train, y_train)
        oof = pipe.predict(X_val)
        oof_preds[val_idx] = oof
        test_preds += pipe.predict(X_test_data) / kf.n_splits

        metrics = evaluate(y_val, oof)
        print(f"Fold {fold + 1}: {metrics}")

        gc.collect()

    return oof_preds, test_preds


# Ensemble Training
models = {
    "XGB": lambda: XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        # tree_method="gpu_hist",
        # predictor="gpu_predictor",
        # gpu_id=0,
        random_state=1,
    ),
    "LGB": lambda: LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        # device="gpu",
        # gpu_platform_id=0,
        # gpu_device_id=0,
        random_state=1,
        verbose=-1,
    ),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.8,
        subsample=0.8,
        # task_type="GPU",
        # devices="0",
        verbose=0,
    ),
    "RF": lambda: RandomForestRegressor(
        n_estimators=1000,
        # max_depth=25,
        # max_features='sqrt',
        n_jobs=-1,
        random_state=1,
    ),
    "ET": lambda: ExtraTreesRegressor(
        n_estimators=1000,
        # max_depth=25,
        # max_features='sqrt',
        n_jobs=-1,
        random_state=1,
    ),
    "Ridge": lambda: Ridge(alpha=0.5),
    "Lasso": lambda: Lasso(alpha=0.05, max_iter=2000),
    # "SVR": lambda: SVR(C=1.0, epsilon=0.1),
}

val_stack_df = pd.DataFrame()
test_stack_df = pd.DataFrame()

for name, model in models.items():
    print(f"Stacking Input from: {name}")
    needs_scaling = name in ["Ridge", "Lasso", "SVR"]
    val_pred, test_pred = cross_val_predict(X, y, X_test, model, scale=needs_scaling)
    val_stack_df[name] = val_pred
    test_stack_df[name] = test_pred

meta_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1,
)
meta_model.fit(val_stack_df, y)

meta_preds = meta_model.predict(val_stack_df)
print("Meta-model performance:", evaluate(y, meta_preds))

final_test_preds = meta_model.predict(test_stack_df)


# Submission
submission = sample_submission.copy()
submission["target"] = final_test_preds
submission.to_csv("ensemble_submission.csv", index=False)

try:
    from IPython.display import FileLink, display

    display(FileLink("ensemble_submission.csv"))
except Exception:
    print("Saved: ensemble_submission.csv")


# Plots for Report
model_scores = {name: np.sqrt(mean_squared_error(y, val_stack_df[name])) for name in val_stack_df.columns}
model_scores.update({"Meta": np.sqrt(mean_squared_error(y, meta_preds))})

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
plt.bar(model_scores.keys(), model_scores.values())
plt.ylabel("RMSE Score")
plt.title("Model Performance on Validation Set")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = y - meta_preds

plt.figure(figsize=(10, 6))
plt.scatter(meta_preds, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Meta-Model")
plt.grid(True)
plt.tight_layout()
plt.show()

importances = meta_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(val_stack_df.columns, importances)
plt.ylabel("Importance")
plt.title("Meta-Model - Base Model Importance")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
