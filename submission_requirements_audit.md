# Submission Requirements Audit (March 31, 2026)

## Implemented in `train_and_submit.py`

- [x] Loads `train.csv`, `test.csv`, `sample_submission.csv`
- [x] Prints shape, columns, missing values, dtypes
- [x] Separates features and target (`target`)
- [x] Handles missing values
  - [x] Numerical -> median imputation
  - [x] Categorical -> `Unknown` + one-hot encoding
- [x] Parses `publication_timestamp` and extracts `pub_year`, `pub_month`, `pub_day`, `pub_weekday`
- [x] Drops `track_identifier`, `composition_label_0/1/2` if present
- [x] Feature scaling with `StandardScaler` for linear model
- [x] Feature engineering (`avg_energy`, `avg_danceability`, `avg_tempo`, `tempo_range`, `energy_to_danceability`)
- [x] 80/20 train-validation split with fixed random state
- [x] Two distinct models
  - [x] Model 1: Linear Regression
  - [x] Model 2: Gradient Boosting Regressor
- [x] Metrics computed
  - [x] RMSE (competition metric)
  - [x] MAE
  - [x] R2
  - [x] MAPE
- [x] Model comparison table printed and saved (`model_comparison_metrics.csv`)
- [x] Retrains both models on full training data
- [x] Predicts test set and clips predictions to [0, 100]
- [x] Writes two submission files
  - [x] `submission_model1.csv`
  - [x] `submission_model2.csv`
- [x] Visualization outputs
  - [x] Feature importance plot for tree model
  - [x] Prediction vs actual plot
  - [x] Residual plot
- [x] Report-support outputs
  - [x] `model_summary.txt`
  - [x] `report_draft.md`
  - [x] `submission_checklist.txt`

## Manual Deliverables Still Required (Cannot Be Auto-generated)

- [ ] Upload at least one submission on Kaggle and capture a full-page proof screenshot
- [ ] Capture final leaderboard score/rank full-page screenshot(s)
- [ ] Prepare final 3-4 page PDF report (use `report_draft.md` as base)
- [ ] Upload PDF to Moodle submission link
- [ ] Upload screenshots/leaderboard details to separate link (if required by instructor)

## Notes

- The code is prepared to generate all required files once executed successfully in the local environment.
- Screenshots and final PDF are inherently manual steps and must be completed by the student after running and submitting on Kaggle.
