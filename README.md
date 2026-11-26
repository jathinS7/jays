# Jays ML Projects

Two independent machine learning projects in one repository.

| Script | Problem | Main methods |
|---|---|---|
| `jays.py` | Customer churn classification | Imputation (KNN), encoding pipeline, KNN, Naive Bayes, decision tree, logistic regression |
| `cropdisease.py` | Crop disease severity classification and yield-loss regression | EDA, Random Forest classifier and regressor, outbreak-risk forecasts |

## Setup

```bash
pip install -r requirements.txt
```

## `jays.py`: customer churn

Expects `ml_practice_dataset_100MB.csv` in the same folder. It drops `customer_id`, `churn` and `review_text` from the features, then compares four classifiers on a 70/30 split. It prints accuracy only.

```bash
python jays.py
```

## `cropdisease.py`: crop disease risk

Looks for its CSV files in `/kaggle/input/...` or the current folder. The required files are:

- `global_crop_disease_risk_prediction_2026_2100.csv` (main data)
- `disease_knowledge_base_symptoms_controls.csv`
- `crop_disease_solution_library.csv`
- `future_disease_forecasts_2030_2050_2100.csv`
- `crop_vulnerability_rankings.csv`

```bash
python cropdisease.py
```

It prints accuracy (classification) and MAE / R² (yield-loss regression), and shows EDA plots.

## Known limitations

- `cropdisease.py` classifies `disease_severity` with features that include `outbreak_probability_*` columns. Check the dataset documentation for leakage before treating the accuracy as real.
- `jays.py` uses the target `churn` as a column name but does not check for class imbalance.
- Results have not been validated with cross-validation.

## Author

Chillagattu Jathin · [LinkedIn](https://www.linkedin.com/in/jathins7)
