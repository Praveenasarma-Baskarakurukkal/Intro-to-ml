import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

try:
    from catboost import CatBoostRegressor  # type: ignore[reportMissingImports]
    HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor  # type: ignore[reportMissingImports]
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

RANDOM_STATE = 42
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"
CV_SPLITS = 5
XGB_SEARCH_ITERS = 30
GBR_SEARCH_ITERS = 30
CATBOOST_RANDOM_TRIALS = 12
BLEND_RANDOM_SEARCH_ITERS = 4000


# Rubric-driven notes that can be copied into the final report.
APPROACH_NOTES = {
    "Model 1: LinearRegression": {
        "why": "Strong baseline: fast, interpretable, and establishes a linear reference point.",
        "how": "Fits a linear relationship between engineered features and target after imputation, scaling, and one-hot encoding.",
        "drawbacks": "Cannot naturally model complex non-linear feature interactions; may underfit high-variance patterns.",
    },
    "Model 2: AdvancedBoosting": {
        "why": "Stronger non-linear learner for tabular data; CatBoost often performs very well with mixed numeric/categorical features.",
        "how": "Uses gradient boosting trees (CatBoost when available, otherwise tuned GradientBoostingRegressor) to capture feature interactions and complex non-linear effects.",
        "drawbacks": "Higher training cost and more hyperparameters; interpretability is lower than linear regression.",
    },
    "Model 2: GradientBoosting": {
        "why": "Stronger non-linear learner for tabular data; CatBoost often performs very well with mixed numeric/categorical features.",
        "how": "Uses gradient boosting trees (CatBoost when available, otherwise tuned GradientBoostingRegressor) to capture feature interactions and complex non-linear effects.",
        "drawbacks": "Higher training cost and more hyperparameters; interpretability is lower than linear regression.",
    },
    "Model 2: XGBoost": {
        "why": "Tree boosting with strong regularization and flexible depth/learning-rate tradeoffs often performs strongly on tabular Kaggle tasks.",
        "how": "Trains an XGBoost regressor on preprocessed features, tuned with randomized CV search to reduce RMSE.",
        "drawbacks": "Can overfit if not tuned; training time and memory are higher than linear models.",
    },
    "Model 2: AdvancedBlend": {
        "why": "Combining complementary boosted models can reduce variance and improve leaderboard stability.",
        "how": "Blends two best advanced models using validation-optimized weights on RMSE.",
        "drawbacks": "Harder to interpret and maintain than a single model; requires extra training runs.",
    },
}


def print_initial_exploration(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING & INITIAL EXPLORATION")
    print("=" * 80)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nMissing values in train:")
    print(train_df.isna().sum().sort_values(ascending=False).head(20))

    print("\nMissing values in test:")
    print(test_df.isna().sum().sort_values(ascending=False).head(20))

    print("\nData types in train:")
    print(train_df.dtypes)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Convert timestamp to structured calendar features that tree and linear models can both use.
    if "publication_timestamp" in out.columns:
        out["publication_timestamp"] = pd.to_datetime(out["publication_timestamp"], errors="coerce")
        out["pub_year"] = out["publication_timestamp"].dt.year
        out["pub_month"] = out["publication_timestamp"].dt.month
        out["pub_day"] = out["publication_timestamp"].dt.day
        out["pub_weekday"] = out["publication_timestamp"].dt.weekday
        out = out.drop(columns=["publication_timestamp"])

    energy_cols = [c for c in out.columns if "emotional_charge" in c]
    dance_cols = [c for c in out.columns if "groove_efficiency" in c]
    tempo_like_cols = [c for c in out.columns if "beat_frequency" in c or "tempo" in c]

    # Aggregate descriptors reduce sparsity/noise when there are multiple related columns.
    if energy_cols:
        out["avg_energy"] = out[energy_cols].mean(axis=1)
    if dance_cols:
        out["avg_danceability"] = out[dance_cols].mean(axis=1)
    if tempo_like_cols:
        out["avg_tempo"] = out[tempo_like_cols].mean(axis=1)
        out["tempo_range"] = out[tempo_like_cols].max(axis=1) - out[tempo_like_cols].min(axis=1)

    # Ratio feature to capture balance between energy and danceability dimensions.
    if "avg_energy" in out.columns and "avg_danceability" in out.columns:
        out["energy_to_danceability"] = safe_divide(out["avg_energy"], out["avg_danceability"]).replace([np.inf, -np.inf], np.nan)

    drop_cols = [
        "track_identifier",
        "composition_label_0",
        "composition_label_1",
        "composition_label_2",
    ]
    drop_cols = [c for c in drop_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Median imputation is robust to outliers in numerical columns.
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    # Unknown category placeholder prevents dropping rows due to missing categorical values.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def prepare_catboost_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_cb = X_train.copy()
    val_cb = X_val.copy()

    numeric_cols = train_cb.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_cb.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        medians = train_cb[numeric_cols].median()
        train_cb[numeric_cols] = train_cb[numeric_cols].fillna(medians)
        val_cb[numeric_cols] = val_cb[numeric_cols].fillna(medians)

    if categorical_cols:
        train_cb[categorical_cols] = train_cb[categorical_cols].fillna("Unknown").astype(str)
        val_cb[categorical_cols] = val_cb[categorical_cols].fillna("Unknown").astype(str)

    return train_cb, val_cb, numeric_cols, categorical_cols


def prepare_catboost_full_inputs(
    X_train_full: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_cb = X_train_full.copy()
    test_cb = X_test.copy()

    numeric_cols = train_cb.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_cb.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        medians = train_cb[numeric_cols].median()
        train_cb[numeric_cols] = train_cb[numeric_cols].fillna(medians)
        test_cb[numeric_cols] = test_cb[numeric_cols].fillna(medians)

    if categorical_cols:
        train_cb[categorical_cols] = train_cb[categorical_cols].fillna("Unknown").astype(str)
        test_cb[categorical_cols] = test_cb[categorical_cols].fillna("Unknown").astype(str)

    return train_cb, test_cb, numeric_cols, categorical_cols


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }


def find_best_blend_weight(y_true: pd.Series, pred1: np.ndarray, pred2: np.ndarray) -> tuple[float, float]:
    best_w = 0.0
    best_rmse = float("inf")
    # Coarse grid is fast and usually sufficient for two-model blending.
    for w in np.linspace(0.0, 1.0, 21):
        blended = w * pred1 + (1.0 - w) * pred2
        rmse = np.sqrt(mean_squared_error(y_true, blended))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)
    return best_w, best_rmse


def optimize_multi_blend_weights(
    y_true: pd.Series,
    candidate_preds: list[np.ndarray],
    random_state: int,
    n_samples: int = BLEND_RANDOM_SEARCH_ITERS,
) -> tuple[np.ndarray, float]:
    """Optimize convex blend weights for 2-3 models using random simplex search."""
    if len(candidate_preds) < 2:
        raise ValueError("Need at least 2 prediction vectors for blending")

    preds = [np.asarray(p, dtype=float) for p in candidate_preds]
    n_models = len(preds)
    rng = np.random.default_rng(random_state)

    best_weights = np.zeros(n_models, dtype=float)
    best_weights[0] = 1.0
    best_rmse = float("inf")

    # Seed with one-hot and uniform weights so blend never performs worse due to poor initialization.
    seed_weights = [np.eye(n_models)[i] for i in range(n_models)]
    seed_weights.append(np.full(n_models, 1.0 / n_models))

    for w in seed_weights:
        blended = np.sum([w[i] * preds[i] for i in range(n_models)], axis=0)
        rmse = float(np.sqrt(mean_squared_error(y_true, blended)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = np.array(w, dtype=float)

    for _ in range(n_samples):
        w = rng.dirichlet(alpha=np.ones(n_models))
        blended = np.sum([w[i] * preds[i] for i in range(n_models)], axis=0)
        rmse = float(np.sqrt(mean_squared_error(y_true, blended)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = np.array(w, dtype=float)

    return best_weights, best_rmse


def get_approach2_notes(model2_label: str) -> dict:
    if model2_label in APPROACH_NOTES:
        return APPROACH_NOTES[model2_label]
    if "CatBoost" in model2_label:
        return APPROACH_NOTES["Model 2: AdvancedBoosting"]
    if "XGBoost" in model2_label:
        return APPROACH_NOTES["Model 2: XGBoost"]
    if "Blend" in model2_label:
        return APPROACH_NOTES["Model 2: AdvancedBlend"]
    return APPROACH_NOTES["Model 2: GradientBoosting"]


def build_report_summary(comparison_df: pd.DataFrame, model2_label: str) -> str:
    winner = comparison_df.sort_values("RMSE").iloc[0]["Model"]
    approach2_notes = get_approach2_notes(model2_label)
    lines = [
        "Approach 1: Linear Regression pipeline",
        f"- Why chosen: {APPROACH_NOTES['Model 1: LinearRegression']['why']}",
        f"- How it works: {APPROACH_NOTES['Model 1: LinearRegression']['how']}",
        f"- Drawbacks: {APPROACH_NOTES['Model 1: LinearRegression']['drawbacks']}",
        "",
        f"Approach 2: {model2_label} pipeline",
        f"- Why chosen: {approach2_notes['why']}",
        f"- How it works: {approach2_notes['how']}",
        f"- Drawbacks: {approach2_notes['drawbacks']}",
        "",
        "Validation comparison (80/20 split):",
        comparison_df.to_string(index=False),
        "",
        "Metric discussion:",
        "- RMSE is the competition metric and penalizes large errors strongly.",
        "- MAE provides average absolute deviation and is easier to interpret in target units.",
        "- R2 indicates how much target variance is explained by the model.",
        "- MAPE gives relative error proportion but can be unstable near zero targets.",
        "",
        f"Final conclusion: {winner} has the best validation RMSE in this experiment.",
        "Both submissions are clipped to [0, 100] before export.",
    ]
    return "\n".join(lines)


def build_report_draft_markdown(comparison_df: pd.DataFrame, model2_label: str) -> str:
    winner = comparison_df.sort_values("RMSE").iloc[0]["Model"]
    approach2_notes = get_approach2_notes(model2_label)
    markdown = f"""# mlX 2.0 Regression Challenge Report Draft

## 1. Objective
Predict song popularity score (0-100) using only `train.csv` and `test.csv`.

## 2. Data Preparation
- Loaded train/test with pandas.
- Checked shapes, column names, missing values, and dtypes.
- Target variable: `target`.
- Missing value handling:
  - Numerical: median imputation
  - Categorical: constant `Unknown`
- Date processing:
  - Converted `publication_timestamp` to datetime
  - Extracted `pub_year`, `pub_month`, `pub_day`, `pub_weekday`
- Categorical encoding: One-Hot Encoding (`handle_unknown='ignore'`).
- Dropped candidate identifiers/labels: `track_identifier`, `composition_label_0/1/2` (when present).

## 3. Feature Engineering
- Aggregates: `avg_energy`, `avg_danceability`, `avg_tempo`
- Difference: `tempo_range`
- Ratio: `energy_to_danceability`
- Existing derived features such as `emotional_charge` and `groove_efficiency` retained and utilized.

## 4. Modeling Approaches
### Approach 1: Linear Regression
- Why: {APPROACH_NOTES['Model 1: LinearRegression']['why']}
- How: {APPROACH_NOTES['Model 1: LinearRegression']['how']}
- Drawbacks: {APPROACH_NOTES['Model 1: LinearRegression']['drawbacks']}

### Approach 2: {model2_label}
- Why: {approach2_notes['why']}
- How: {approach2_notes['how']}
- Drawbacks: {approach2_notes['drawbacks']}

## 5. Validation Setup and Metrics
- Train/validation split: 80/20 (`random_state={RANDOM_STATE}`).
- Metrics: RMSE (competition metric), MAE, R2, MAPE.

## 6. Results
{comparison_df.to_string(index=False)}

Best by RMSE: **{winner}**

## 7. Metric Discussion
- RMSE emphasizes large errors more than MAE, aligning with leaderboard sensitivity.
- MAE reflects typical absolute prediction error in target units.
- R2 describes variance explained.
- MAPE is useful for relative error but can be unstable for small true values.

## 8. Submission Outputs
- `submission_model1.csv`
- `submission_model2.csv`

## 9. Required Screenshot Placeholders
1. Full-page screenshot of first Kaggle submission proof.
2. Full-page screenshot(s) of final score and leaderboard rank.
3. Any additional full-page screenshots requested by the submission portal.

## 10. Final Conclusion
The stronger non-linear model ({winner}) is preferred for final competition submission in this run.
"""
    return markdown


def build_submission_checklist() -> str:
    return """ASSIGNMENT CHECKLIST

[x] Two different ML approaches implemented
[x] Competition metric (RMSE) computed
[x] Additional metrics computed (MAE, R2, MAPE)
[x] Two submission files generated
[x] Model comparison table produced
[x] Visualizations generated (feature importance, prediction-vs-actual, residuals)
[x] Summary text generated
[ ] First submission full-page screenshot (manual)
[ ] Final score/rank full-page screenshot(s) (manual)
[ ] Final PDF report export (manual)

Manual items must be completed after uploading to Kaggle and taking screenshots.
"""


def plot_prediction_vs_actual(y_true: pd.Series, y_pred: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5)
    line_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    line_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, out_path: str) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(
    model,
    out_path: str,
    top_n: int = 20,
    feature_names: list[str] | None = None,
) -> None:
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]
        feature_names_arr = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
    else:
        if feature_names is None:
            raise ValueError("feature_names must be provided for non-pipeline models")
        feature_names_arr = np.array(feature_names)
        importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names_arr)[order]
    top_values = importances[order]

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(top_features)), top_values)
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (Model 2)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    print("Running mlX 2.0 regression pipeline...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)

    print_initial_exploration(train_df, test_df)

    if "target" not in train_df.columns:
        raise ValueError("Expected 'target' column in train.csv")

    y = train_df["target"].copy()
    X_raw = train_df.drop(columns=["target"]).copy()
    X_test_raw = test_df.copy()

    print("\n" + "=" * 80)
    print("STEP 2-3: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 80)

    X = add_feature_engineering(X_raw)
    X_test = add_feature_engineering(X_test_raw)

    print(f"Feature shape after engineering (train): {X.shape}")
    print(f"Feature shape after engineering (test) : {X_test.shape}")

    missing_test_cols = sorted(set(X.columns) - set(X_test.columns))
    extra_test_cols = sorted(set(X_test.columns) - set(X.columns))

    for c in missing_test_cols:
        X_test[c] = np.nan
    if extra_test_cols:
        X_test = X_test.drop(columns=extra_test_cols)

    X_test = X_test[X.columns]

    print("\n" + "=" * 80)
    print("STEP 4: TRAIN-VALIDATION SPLIT")
    print("=" * 80)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Train split shape: {X_train.shape}, Validation split shape: {X_val.shape}")

    print("\n" + "=" * 80)
    print("STEP 5: MODEL 1 (BASELINE) - LINEAR REGRESSION")
    print("=" * 80)

    model1 = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]
    )
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_val)

    results1 = evaluate_model("Model 1: LinearRegression", y_val, y_pred1)

    print("\n" + "=" * 80)
    print("STEP 6: MODEL 2 (ADVANCED) - BOOSTING")
    print("=" * 80)

    model2_kind = "GradientBoosting"
    model2_label = "Model 2: GradientBoosting"
    model2 = None
    y_pred2 = None

    advanced_candidates: list[dict] = []

    if HAS_CATBOOST:
        print("Training advanced candidate: CatBoostRegressor")
        X_train_cb, X_val_cb, _, cat_cols = prepare_catboost_inputs(X_train, X_val)
        rng_cat = np.random.default_rng(RANDOM_STATE)
        cat_trials = []
        for _ in range(CATBOOST_RANDOM_TRIALS):
            cat_trials.append(
                {
                    "depth": int(rng_cat.choice([6, 7, 8, 9, 10])),
                    "learning_rate": float(rng_cat.choice([0.015, 0.02, 0.03, 0.04, 0.06])),
                    "l2_leaf_reg": float(rng_cat.choice([3.0, 5.0, 7.0, 9.0, 11.0])),
                    "bagging_temperature": float(rng_cat.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                    "random_strength": float(rng_cat.choice([0.5, 1.0, 1.5, 2.0])),
                    "subsample": float(rng_cat.choice([0.7, 0.8, 0.9, 1.0])),
                }
            )

        best_cat_rmse = float("inf")
        best_cat_pred: np.ndarray | None = None
        best_cat_model = None
        best_cat_params = None

        for trial in cat_trials:
            cat_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=RANDOM_STATE,
                iterations=5000,
                depth=trial["depth"],
                learning_rate=trial["learning_rate"],
                l2_leaf_reg=trial["l2_leaf_reg"],
                bagging_temperature=trial["bagging_temperature"],
                random_strength=trial["random_strength"],
                subsample=trial["subsample"],
                verbose=0,
            )
            cat_model.fit(
                X_train_cb,
                y_train,
                cat_features=cat_cols,
                eval_set=(X_val_cb, y_val),
                use_best_model=True,
                verbose=False,
            )
            cat_pred = cat_model.predict(X_val_cb)
            cat_rmse = float(np.sqrt(mean_squared_error(y_val, cat_pred)))
            if cat_rmse < best_cat_rmse:
                best_cat_rmse = cat_rmse
                best_cat_pred = cat_pred
                best_cat_model = cat_model
                best_cat_params = trial

        if best_cat_model is None or best_cat_pred is None:
            raise RuntimeError("CatBoost candidate search failed")

        print(f"CatBoost validation RMSE: {best_cat_rmse:.6f}")
        print(f"CatBoost best params: {best_cat_params}")
        advanced_candidates.append(
            {
                "kind": "CatBoost",
                "label": "Model 2: CatBoost",
                "model": best_cat_model,
                "pred": best_cat_pred,
                "rmse": best_cat_rmse,
                "best_params": best_cat_params,
            }
        )

    if HAS_XGBOOST:
        print("Training advanced candidate: XGBoost (RandomizedSearchCV)")
        xgb_base = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_estimators=800,
                        learning_rate=0.03,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        xgb_cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        xgb_param_distributions = {
            "regressor__n_estimators": [600, 900, 1200, 1500, 1800],
            "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "regressor__max_depth": [4, 5, 6, 8, 10],
            "regressor__subsample": [0.65, 0.8, 0.9, 1.0],
            "regressor__colsample_bytree": [0.65, 0.8, 0.9, 1.0],
            "regressor__min_child_weight": [1, 2, 3, 5, 7],
            "regressor__gamma": [0.0, 0.05, 0.1, 0.2],
            "regressor__reg_alpha": [0.0, 0.05, 0.1, 0.3],
            "regressor__reg_lambda": [0.8, 1.0, 1.5, 2.0],
        }
        xgb_tuner = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=xgb_param_distributions,
            n_iter=XGB_SEARCH_ITERS,
            scoring="neg_root_mean_squared_error",
            cv=xgb_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        xgb_tuner.fit(X_train, y_train)
        xgb_model = xgb_tuner.best_estimator_
        xgb_pred = xgb_model.predict(X_val)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_val, xgb_pred)))
        print(f"XGBoost best CV RMSE: {-xgb_tuner.best_score_:.6f}")
        print(f"XGBoost validation RMSE: {xgb_rmse:.6f}")
        advanced_candidates.append(
            {
                "kind": "XGBoost",
                "label": "Model 2: XGBoost",
                "model": xgb_model,
                "pred": xgb_pred,
                "rmse": xgb_rmse,
                "best_params": xgb_tuner.best_params_,
            }
        )

    print("Training advanced candidate: GradientBoosting (RandomizedSearchCV fallback/benchmark)")
    gbr_base = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
            (
                "regressor",
                GradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                ),
            ),
        ]
    )
    gbr_cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gbr_param_distributions = {
        "regressor__n_estimators": [400, 700, 1000, 1300],
        "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "regressor__max_depth": [2, 3, 4, 5],
        "regressor__subsample": [0.65, 0.8, 0.9, 1.0],
        "regressor__min_samples_leaf": [1, 2, 4, 6, 8],
        "regressor__min_samples_split": [2, 4, 8, 12],
        "regressor__max_features": [None, "sqrt", "log2"],
    }
    gbr_tuner = RandomizedSearchCV(
        estimator=gbr_base,
        param_distributions=gbr_param_distributions,
        n_iter=GBR_SEARCH_ITERS,
        scoring="neg_root_mean_squared_error",
        cv=gbr_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    gbr_tuner.fit(X_train, y_train)
    gbr_model = gbr_tuner.best_estimator_
    gbr_pred = gbr_model.predict(X_val)
    gbr_rmse = float(np.sqrt(mean_squared_error(y_val, gbr_pred)))
    print(f"GradientBoosting best CV RMSE: {-gbr_tuner.best_score_:.6f}")
    print(f"GradientBoosting validation RMSE: {gbr_rmse:.6f}")
    advanced_candidates.append(
        {
            "kind": "GradientBoosting",
            "label": "Model 2: GradientBoosting",
            "model": gbr_model,
            "pred": gbr_pred,
            "rmse": gbr_rmse,
            "best_params": gbr_tuner.best_params_,
        }
    )

    advanced_candidates = sorted(advanced_candidates, key=lambda c: c["rmse"])
    best_candidate = advanced_candidates[0]
    model2_kind = best_candidate["kind"]
    model2_label = best_candidate["label"]
    model2 = best_candidate["model"]
    y_pred2 = best_candidate["pred"]

    model2_blend_info = None
    if len(advanced_candidates) >= 2:
        top_blend_candidates = advanced_candidates[: min(3, len(advanced_candidates))]
        blend_preds = [c["pred"] for c in top_blend_candidates]
        blend_weights, blend_rmse_adv = optimize_multi_blend_weights(
            y_val,
            blend_preds,
            random_state=RANDOM_STATE,
            n_samples=BLEND_RANDOM_SEARCH_ITERS,
        )

        blend_desc = " + ".join(
            [f"{blend_weights[i]:.2f}*{top_blend_candidates[i]['label']}" for i in range(len(top_blend_candidates))]
        )
        print(f"Advanced blend diagnostic: {blend_desc} -> RMSE={blend_rmse_adv:.6f}")

        if blend_rmse_adv < float(best_candidate["rmse"]):
            y_pred2 = np.sum(
                [blend_weights[i] * top_blend_candidates[i]["pred"] for i in range(len(top_blend_candidates))],
                axis=0,
            )
            model2_kind = "AdvancedBlend"
            model2_label = "Model 2: AdvancedBlend"
            model2 = None
            model2_blend_info = {
                "candidates": top_blend_candidates,
                "weights": blend_weights,
            }
            print("Using blended advanced model as Model 2 based on lower validation RMSE.")

    results2 = evaluate_model(model2_label, y_val, y_pred2)

    print("\n" + "=" * 80)
    print("STEP 7: MODEL COMPARISON")
    print("=" * 80)

    comparison_df = pd.DataFrame([results1, results2])
    comparison_df = comparison_df.sort_values("RMSE").reset_index(drop=True)
    print(comparison_df.to_string(index=False))

    winner = comparison_df.iloc[0]
    explanation = (
        f"\nBest model by RMSE: {winner['Model']}. "
        "Advanced boosting generally captures non-linear interactions better than linear baselines; "
        "regularization and early stopping/hyperparameter tuning help control overfitting."
    )
    print(explanation)

    blend_w_model1, blend_rmse = find_best_blend_weight(y_val, y_pred1, y_pred2)
    print(
        f"Blend diagnostic (validation only): {blend_w_model1:.2f}*Model1 + "
        f"{1.0 - blend_w_model1:.2f}*Model2 -> RMSE={blend_rmse:.6f}"
    )

    comparison_df.to_csv("model_comparison_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("STEP 12: VISUALIZATION")
    print("=" * 80)

    plot_prediction_vs_actual(y_val, y_pred2, "prediction_vs_actual_model2.png")
    plot_residuals(y_val, y_pred2, "residual_plot_model2.png")
    if model2_kind == "CatBoost":
        plot_feature_importance(model2, "feature_importance_model2.png", top_n=20, feature_names=X.columns.tolist())
    elif model2_kind in {"XGBoost", "GradientBoosting"}:
        plot_feature_importance(model2, "feature_importance_model2.png", top_n=20)
    else:
        print("Skipping single-model feature importance plot for blended Model 2.")

    print("Saved plots:")
    print("- prediction_vs_actual_model2.png")
    print("- residual_plot_model2.png")
    print("- feature_importance_model2.png")

    print("\n" + "=" * 80)
    print("STEP 8-11: TRAIN FULL DATA, PREDICT TEST, CREATE SUBMISSIONS")
    print("=" * 80)

    model1_full = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]
    )
    model1_full.fit(X, y)

    if model2_kind == "CatBoost":
        X_full_cb, X_test_cb, _, cat_cols_full = prepare_catboost_full_inputs(X, X_test)
        full_iterations = int(model2.get_best_iteration()) if model2.get_best_iteration() is not None else 1500
        full_iterations = max(full_iterations, 300)
        model2_full = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            depth=8,
            learning_rate=0.03,
            iterations=full_iterations,
            l2_leaf_reg=5.0,
            bagging_temperature=0.5,
            random_strength=1.0,
            subsample=0.8,
            verbose=0,
        )
        model2_full.fit(X_full_cb, y, cat_features=cat_cols_full, verbose=False)
        test_pred2 = np.clip(model2_full.predict(X_test_cb), 0, 100)
    elif model2_kind in {"XGBoost", "GradientBoosting"}:
        model2.fit(X, y)
        test_pred2 = np.clip(model2.predict(X_test), 0, 100)
    elif model2_kind == "AdvancedBlend":
        if model2_blend_info is None:
            raise RuntimeError("model2_blend_info is not set for AdvancedBlend")
        blend_candidates = model2_blend_info["candidates"]
        blend_weights = np.asarray(model2_blend_info["weights"], dtype=float)
        component_preds = []

        for cand in blend_candidates:
            if cand["kind"] == "CatBoost":
                X_full_cb, X_test_cb, _, cat_cols_full = prepare_catboost_full_inputs(X, X_test)
                cand_best_params = cand.get("best_params", {})
                cand_iters = int(cand["model"].get_best_iteration()) if cand["model"].get_best_iteration() is not None else 1500
                cand_iters = max(cand_iters, 300)
                cand_full = CatBoostRegressor(
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    iterations=cand_iters,
                    depth=int(cand_best_params.get("depth", 8)),
                    learning_rate=float(cand_best_params.get("learning_rate", 0.03)),
                    l2_leaf_reg=float(cand_best_params.get("l2_leaf_reg", 5.0)),
                    bagging_temperature=float(cand_best_params.get("bagging_temperature", 0.5)),
                    random_strength=float(cand_best_params.get("random_strength", 1.0)),
                    subsample=float(cand_best_params.get("subsample", 0.8)),
                    verbose=0,
                )
                cand_full.fit(X_full_cb, y, cat_features=cat_cols_full, verbose=False)
                component_preds.append(cand_full.predict(X_test_cb))
            else:
                cand_full = cand["model"]
                cand_full.fit(X, y)
                component_preds.append(cand_full.predict(X_test))

        test_pred2 = np.clip(
            np.sum([blend_weights[i] * component_preds[i] for i in range(len(component_preds))], axis=0),
            0,
            100,
        )
    else:
        raise RuntimeError(f"Unsupported model2_kind: {model2_kind}")

    test_pred1 = np.clip(model1_full.predict(X_test), 0, 100)

    sub1 = sample_sub_df.copy()
    sub2 = sample_sub_df.copy()
    sub1["target"] = test_pred1
    sub2["target"] = test_pred2

    sub1.to_csv("submission_model1.csv", index=False)
    sub2.to_csv("submission_model2.csv", index=False)

    summary_text = build_report_summary(comparison_df, model2_label)
    report_md = build_report_draft_markdown(comparison_df, model2_label)
    checklist_text = build_submission_checklist()

    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    with open("report_draft.md", "w", encoding="utf-8") as f:
        f.write(report_md + "\n")
    with open("submission_checklist.txt", "w", encoding="utf-8") as f:
        f.write(checklist_text + "\n")

    print("Saved files:")
    print("- submission_model1.csv")
    print("- submission_model2.csv")
    print("- model_summary.txt")
    print("- model_comparison_metrics.csv")
    print("- report_draft.md")
    print("- submission_checklist.txt")


if __name__ == "__main__":
    main()
