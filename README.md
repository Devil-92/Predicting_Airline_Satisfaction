# Predicting Airline Satisfaction

Gradient-boosting pipeline for [Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10): predict whether an airline passenger was satisfied, scored by ROC-AUC. Built in Colab; the full workflow is in [`Predicting_Airline_Satisfaction.ipynb`](Predicting_Airline_Satisfaction.ipynb).

## Data

- 699,635 training rows, 299,844 test rows, 21 features plus a binary target (about 44% positive).
- 13 ordinal service ratings (wifi, online boarding, seat comfort, ...), age, flight distance, departure and arrival delay, and 4 categoricals (gender, customer type, type of travel, class).
- No missing values.

## Results

All CV scores use 5-fold stratified CV with seed 42.

| Model | CV AUC | Public LB |
|---|---|---|
| LightGBM, defaults | 0.95771 | 0.95721 |
| LightGBM + categorical combinations | 0.95790 | 0.95740 |
| XGBoost, early stopping | 0.95864 | 0.95829 |
| LightGBM + XGBoost, rank blend | 0.95888 | 0.95832 |
| LightGBM + XGBoost + CatBoost, rank blend | 0.95897 | 0.95841 |

CV and the public leaderboard differ by a steady ~0.0005 across every stage, so local CV was a reliable guide for comparing changes.

## Approach

1. **Baseline.** Default LightGBM with object columns cast to `category`, no other preprocessing.
2. **Categorical combinations.** All 2-way and 4-way concatenations of the four categorical columns, added as new categorical features (+0.0002 CV, improving every fold).
3. **XGBoost and LightGBM with early stopping.** Native categorical support, test predictions averaged across folds.
4. **CatBoost.** Trained on the raw categoricals with `max_ctr_complexity=1`.
5. **Blending.** Rank-average of out-of-fold predictions, with a weight sweep for the two-model blend.


## Observations

- Online boarding carries about 51% of LightGBM's gain importance, followed by inflight wifi (13%) and type of travel (12%).
- Out-of-fold predictions from the three models correlate at about 0.998, so blending adds only ~0.0002 CV. Further gains need more diverse inputs, not more models of the same kind.
- Early stopping on the validation fold makes out-of-fold AUC slightly optimistic; the constant CV to LB gap suggests it does not distort model comparisons.

## Reproducing

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost kagglehub seaborn matplotlib
```

The notebook downloads the data with `kagglehub`, which needs Kaggle credentials. Set the `KAGGLEHUB_CACHE` path in the first cells to a location you can write to. Each stage writes a submission CSV to the working directory.

## Next steps

- Lower learning rate (0.03) with early stopping.
- Multi-seed averaging for the XGBoost and LightGBM runs.
- Check whether the original public airline satisfaction dataset can be appended as extra training rows.
