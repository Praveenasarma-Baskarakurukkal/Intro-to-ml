import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
from lightgbm import LGBMRegressor

# =========================
# LOAD DATA
# =========================
train = pd.read_csv('/kaggle/input/competitions/mlx-2-0-regression/train.csv')
test = pd.read_csv('/kaggle/input/competitions/mlx-2-0-regression/test.csv')

# =========================
# BASIC CLEANING
# =========================
train.drop('track_identifier', axis=1, inplace=True)
test.drop('track_identifier', axis=1, inplace=True)

train['publication_timestamp'] = pd.to_datetime(train['publication_timestamp'])
test['publication_timestamp'] = pd.to_datetime(test['publication_timestamp'])

train['year'] = train['publication_timestamp'].dt.year
train['month'] = train['publication_timestamp'].dt.month

test['year'] = test['publication_timestamp'].dt.year
test['month'] = test['publication_timestamp'].dt.month

train.drop('publication_timestamp', axis=1, inplace=True)
test.drop('publication_timestamp', axis=1, inplace=True)

# =========================
# SPLIT
# =========================
X = train.drop(['target', 'id'], axis=1)
y = train['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test = test.drop(['id'], axis=1)

# =========================
# FREQUENCY ENCODING FOR TITLES
# =========================
title_cols = ['composition_label_0', 'composition_label_1', 'composition_label_2']

for col in title_cols:
    freq = X_train[col].value_counts()
    X_train[col] = X_train[col].map(freq)
    X_val[col] = X_val[col].map(freq)
    X_test[col] = X_test[col].map(freq)

X_train[title_cols] = X_train[title_cols].fillna(0)
X_val[title_cols] = X_val[title_cols].fillna(0)
X_test[title_cols] = X_test[title_cols].fillna(0)

# =========================
# COLUMN TYPES
# =========================
num_cols = X_train.select_dtypes(include=['number']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

low_card_cols = [col for col in cat_cols if X_train[col].nunique() < 10]
high_card_cols = [col for col in cat_cols if X_train[col].nunique() >= 10]

# =========================
# MISSING VALUES
# =========================
median_vals = X_train[num_cols].median()

X_train[num_cols] = X_train[num_cols].fillna(median_vals)
X_val[num_cols] = X_val[num_cols].fillna(median_vals)
X_test[num_cols] = X_test[num_cols].fillna(median_vals)

X_train[cat_cols] = X_train[cat_cols].fillna('Unknown')
X_val[cat_cols] = X_val[cat_cols].fillna('Unknown')
X_test[cat_cols] = X_test[cat_cols].fillna('Unknown')

# =========================
# ENCODING
# =========================
ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train[low_card_cols] = ord_encoder.fit_transform(X_train[low_card_cols])
X_val[low_card_cols] = ord_encoder.transform(X_val[low_card_cols])
X_test[low_card_cols] = ord_encoder.transform(X_test[low_card_cols])

target_encoder = ce.TargetEncoder(cols=high_card_cols, smoothing=30)

X_train[high_card_cols] = target_encoder.fit_transform(X_train[high_card_cols], y_train)
X_val[high_card_cols] = target_encoder.transform(X_val[high_card_cols])
X_test[high_card_cols] = target_encoder.transform(X_test[high_card_cols])

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    groups = [
        'emotional_charge','duration_ms','intensity_index',
        'rhythmic_cohesion','groove_efficiency',
        'instrumental_density','organic_texture','organic_immersion'
    ]

    for g in groups:
        cols = [f'{g}_0', f'{g}_1', f'{g}_2']
        if all(col in df.columns for col in cols):
            df[f'{g}_mean'] = df[cols].mean(axis=1)
            df[f'{g}_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)

    df['tempo_ratio'] = df['beat_frequency_1'] / (df['beat_frequency_0'] + 1)
    df['energy_combined'] = df['intensity_index_0'] + df['intensity_index_1']

    return df

X_train = add_features(X_train)
X_val = add_features(X_val)
X_test = add_features(X_test)

# =========================
# CLEAN COLUMN NAMES
# =========================
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X_val.columns = X_val.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

# =========================
# LIGHTGBM MODEL (FAST + EARLY STOPPING)
# =========================
model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    force_col_wise=True,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[]
)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, preds))
mae = mean_absolute_error(y_val, preds)
r2 = r2_score(y_val, preds)

print("Validation RMSE:", rmse)
print("Validation MAE:", mae)
print("Validation R2:", r2)

# =========================
# FINAL TRAINING
# =========================
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

model.fit(X_full, y_full)

# =========================
# PREDICT TEST
# =========================
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'target': test_preds
})

submission.to_csv('submission_lgb_fast.csv', index=False)
