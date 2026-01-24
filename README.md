# Detection of Abnormal User Behavior Based on Time and Usage Patterns

An integrated User Behavior Analysis (UBA) system that detects anomalous behavior using temporal and behavioral patterns.

The project processes data chronologically to avoid time leakage and target leakage. It applies unsupervised clustering before classification and uses the resulting `Cluster_ID` as a model feature. The project also includes a modular Streamlit interface for interactive, real-time inference.

## Dataset and Leakage Controls

- **Dataset:** `logs_SPEDIA_annotated_en.csv`
- Data is segmented and processed in chronological order.
- Look-ahead bias and target leakage are avoided.
- Raw text is not used directly as a model feature.

## Feature Engineering

`feature_engineering.py` creates features without using future data:

- **Time:** `hour`, `weekday`, `is_weekend`, `is_off_hours`, `is_night`
- **Behavior:** `user_event_index`, `daily_event_index`, `activity_rate`, `daily_activity_rate`
- **User age:** `days_since_first_activity`, `is_new_user`, `is_first_activity`
- **Missing-value indicators:** `activity_is_missing`, `action_is_missing`, `decoder_is_missing`, `level_is_high`

## Clustering and Feature Selection

The clustering stage identifies patterns and adds `Cluster_ID` before classification. It supports DBSCAN, K-Means, and Agglomerative Clustering; uses Silhouette Score for selection; and can use PCA for dimensionality reduction. PCA and t-SNE projections are available for visualization.

Feature selection follows this order:

1. Domain security importance
2. Statistical tests
3. Model importance

The selected feature list is stored in `artifacts/metadata/selected_features.csv` to keep training and evaluation consistent.

## Evaluation

Run `evaluation.py` to load reference data and test splits, apply `Cluster_ID`, align features with the selected-feature metadata, and calculate metrics. Reports are saved under `artifacts/metadata/`:

- `model_test_metrics.csv`
- `model_validation_metrics.csv`

## Streamlit Interface

Launch the application with `streamlit run app.py`. The interface provides:

- A cyber-themed login screen.
- Manual or derived feature input.
- Normal, Medium, and High quick presets.
- Real-time anomaly probability and risk-band display.
- Significance, feature-impact, and clustering reports, including interactive PCA and t-SNE charts.
- CSV batch scanning with downloadable results.

> **Demo credentials:** `admin` / `admin`. Change or remove these credentials before deploying the application beyond a local demonstration.

## Installation and Setup

### 1. Install the requirements

```bash
pip install -r note/requirements.txt
```

### 2. Extract the model archive

The trained models are provided in `artifacts/models.rar`. **Extract its contents into `artifacts/models/` before running the application.** If the `models` folder does not exist, create it first.

On Windows, use 7-Zip or WinRAR:

1. Right-click `artifacts/models.rar`.
2. Choose **Extract files…** (or the equivalent extraction option).
3. Set the destination to the project's `artifacts/models/` folder.
4. Confirm that the model files (such as `decision_tree.pkl`, `random_forest.pkl`, and the other `.pkl` files) are directly inside `artifacts/models/`, not inside an extra nested folder.

The resulting layout should look like this:

```text
artifacts/
├── metadata/
└── models/
    ├── decision_tree.pkl
    ├── gradient_boosting.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

### 3. Extract the dataset archive

The dataset is provided as `data/logs_SPEDIA_annotated_en.rar`. Extract it into the `data/` folder. After extraction, make sure the CSV file is directly available at:

```text
data/logs_SPEDIA_annotated_en.csv
```

### 4. Run the application

From the project root, start the Streamlit interface:

```bash
streamlit run app.py
```

## Quick Manual Test

Run `test-manualy.py` for headless verification. It loads a selected model, adds `Cluster_ID` automatically, and outputs the predicted class and anomaly probability.

## Project Structure

```text
project-root/
├── app.py
├── config.py
├── data_loader.py
├── evaluation.py
├── feature_engineering.py
├── load_and_explore.py
├── model_training.py
├── scoring_service.py
├── test-manualy.py
├── ui_components.py
├── README.md
├── artifacts/
│   ├── models.rar
│   ├── metadata/
│   └── models/                 
├── assets/
│   ├── css/
│   │   ├── login_styles.css
│   │   └── styles.css
│   └── image/
│       └── shadow1.png
├── data/
│   └── logs_SPEDIA_annotated_en.rar
└── note/
    ├── requirements.txt
    └── ملاحظات للبيانات.txt
```

## Reproducibility

- **Random state:** `RANDOM_STATE = 42`
- Models, clustering artifacts, and metrics are stored under `artifacts/` to support reproducibility.

## Developer

**RASHEED AL-RAJEHI** · @ Shadow Wolf Security
