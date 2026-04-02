import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
from lightgbm import LGBMRegressor, early_stopping


# =========================
# CONFIG
# =========================
DATA_DIR = '/kaggle/input/competitions/mlx-2-0-regression'
TRAIN_PATH = f'{DATA_DIR}/train.csv'
TEST_PATH = f'{DATA_DIR}/test.csv'
SUBMISSION_PATH = 'submission_lgb_cv_optimized.csv'

RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 200
MAX_ESTIMATORS = 6000
ENSEMBLE_SEEDS = [42, 2024]

# =========================
# LOAD DATA
# =========================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# =========================
# BASIC CLEANING
# =========================
train.drop('track_identifier', axis=1, inplace=True, errors='ignore')
test.drop('track_identifier', axis=1, inplace=True, errors='ignore')

train['publication_timestamp'] = pd.to_datetime(train['publication_timestamp'])
test['publication_timestamp'] = pd.to_datetime(test['publication_timestamp'])

train['year'] = train['publication_timestamp'].dt.year
train['month'] = train['publication_timestamp'].dt.month
train['day'] = train['publication_timestamp'].dt.day
train['dayofweek'] = train['publication_timestamp'].dt.dayofweek

test['year'] = test['publication_timestamp'].dt.year
test['month'] = test['publication_timestamp'].dt.month
test['day'] = test['publication_timestamp'].dt.day
test['dayofweek'] = test['publication_timestamp'].dt.dayofweek

train.drop('publication_timestamp', axis=1, inplace=True)
test.drop('publication_timestamp', axis=1, inplace=True)

X_raw = train.drop(['target', 'id'], axis=1).copy()
y = train['target'].copy()
X_test_raw = test.drop(['id'], axis=1).copy()

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df = df.copy()

    groups = [
        'emotional_charge','duration_ms','intensity_index',
        'rhythmic_cohesion','groove_efficiency',
        'instrumental_density','organic_texture','organic_immersion'
    ]

    for g in groups:
        cols = [f'{g}_0', f'{g}_1', f'{g}_2']
        if all(col in df.columns for col in cols):
            df[f'{g}_mean'] = df[cols].mean(axis=1)
            df[f'{g}_std'] = df[cols].std(axis=1)
            df[f'{g}_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)

    if {'beat_frequency_0', 'beat_frequency_1'}.issubset(df.columns):
        df['tempo_ratio'] = df['beat_frequency_1'] / (df['beat_frequency_0'] + 1e-6)
        df['tempo_delta'] = df['beat_frequency_1'] - df['beat_frequency_0']

    if {'intensity_index_0', 'intensity_index_1', 'intensity_index_2'}.issubset(df.columns):
        df['energy_combined'] = df['intensity_index_0'] + df['intensity_index_1']
        df['energy_sum_012'] = (
            df['intensity_index_0'] + df['intensity_index_1'] + df['intensity_index_2']
        )

    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    if 'dayofweek' in df.columns:
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    return df


def preprocess_with_fit(X_fit, y_fit, X_apply_list):
    X_fit = X_fit.copy()
    X_apply_list = [df.copy() for df in X_apply_list]

    title_cols = [
        col for col in ['composition_label_0', 'composition_label_1', 'composition_label_2']
        if col in X_fit.columns
    ]

    for col in title_cols:
        freq = X_fit[col].value_counts()
        X_fit[col] = X_fit[col].map(freq)
        for df in X_apply_list:
            df[col] = df[col].map(freq)

    if title_cols:
        X_fit[title_cols] = X_fit[title_cols].fillna(0)
        for df in X_apply_list:
            df[title_cols] = df[title_cols].fillna(0)

    num_cols = X_fit.select_dtypes(include=['number', 'bool']).columns
    cat_cols = X_fit.select_dtypes(include=['object']).columns

    median_vals = X_fit[num_cols].median()
    X_fit[num_cols] = X_fit[num_cols].fillna(median_vals)
    for df in X_apply_list:
        df[num_cols] = df[num_cols].fillna(median_vals)

    X_fit[cat_cols] = X_fit[cat_cols].fillna('Unknown')
    for df in X_apply_list:
        df[cat_cols] = df[cat_cols].fillna('Unknown')

    low_card_cols = [col for col in cat_cols if X_fit[col].nunique(dropna=False) < 12]
    high_card_cols = [col for col in cat_cols if X_fit[col].nunique(dropna=False) >= 12]

    if low_card_cols:
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_fit[low_card_cols] = ord_encoder.fit_transform(X_fit[low_card_cols])
        for df in X_apply_list:
            df[low_card_cols] = ord_encoder.transform(df[low_card_cols])

    if high_card_cols:
        target_encoder = ce.TargetEncoder(cols=high_card_cols, smoothing=30)
        X_fit[high_card_cols] = target_encoder.fit_transform(X_fit[high_card_cols], y_fit)
        for df in X_apply_list:
            df[high_card_cols] = target_encoder.transform(df[high_card_cols])

    X_fit = add_features(X_fit)
    X_apply_list = [add_features(df) for df in X_apply_list]

    clean_cols = X_fit.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    X_fit.columns = clean_cols

    aligned_apply = []
    for df in X_apply_list:
        df = df.set_axis(df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True), axis=1)
        df = df.reindex(columns=X_fit.columns, fill_value=0.0)
        aligned_apply.append(df.astype(np.float32))

    X_fit = X_fit.astype(np.float32)
    return (X_fit, *aligned_apply)


def run_lgbm_cv(X_raw, y, X_test_raw, params, n_splits=5, seeds=(42,)):
    test_preds = np.zeros(len(X_test_raw), dtype=np.float64)
    oof_preds = np.zeros(len(X_raw), dtype=np.float64)

    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(X_raw), dtype=np.float64)
        seed_test = np.zeros(len(X_test_raw), dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw), start=1):
            X_tr_raw = X_raw.iloc[train_idx].copy()
            y_tr = y.iloc[train_idx].copy()
            X_val_raw = X_raw.iloc[val_idx].copy()
            y_val = y.iloc[val_idx].copy()

            X_tr, X_val, X_test_fold = preprocess_with_fit(X_tr_raw, y_tr, [X_val_raw, X_test_raw])

            model = LGBMRegressor(
                **params,
                objective='regression',
                n_estimators=MAX_ESTIMATORS,
                force_col_wise=True,
                n_jobs=-1,
                verbosity=-1,
                random_state=seed + fold,
            )

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
            )

            best_iter = model.best_iteration_ if model.best_iteration_ is not None else MAX_ESTIMATORS
            seed_oof[val_idx] = model.predict(X_val, num_iteration=best_iter)
            seed_test += model.predict(X_test_fold, num_iteration=best_iter) / n_splits

        oof_preds += seed_oof / len(seeds)
        test_preds += seed_test / len(seeds)

    return oof_preds, test_preds


best_params = {
    'learning_rate': 0.025,
    'num_leaves': 96,
    'max_depth': -1,
    'min_child_samples': 24,
    'min_child_weight': 0.001,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.15,
    'reg_lambda': 1.2,
    'min_split_gain': 0.0,
}

oof_preds, test_preds = run_lgbm_cv(
    X_raw=X_raw,
    y=y,
    X_test_raw=X_test_raw,
    params=best_params,
    n_splits=N_SPLITS,
    seeds=ENSEMBLE_SEEDS,
)

rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
mae = float(mean_absolute_error(y, oof_preds))
r2 = float(r2_score(y, oof_preds))

print('OOF RMSE:', rmse)
print('OOF MAE:', mae)
print('OOF R2:', r2)

submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print('Saved:', SUBMISSION_PATH)
