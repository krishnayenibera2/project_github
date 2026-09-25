# Books Scraper & Relational ETL Pipeline

A complete end-to-end Python web scraping, data cleaning, relational modeling, and SQL analytical execution pipeline using `requests`, `BeautifulSoup`, `pandas`, and `sqlite3`.

---

## 1. Project Overview & Features

- **Web Scraping:** Extract title, rating, price, availability, and category for 100 books from [Books to Scrape](http://books.toscrape.com) across 5 paginated listing pages.
- **Robust Cleaning & Imputation:**
  - Standardized ratings to numeric values (1–5).
  - Converted price strings to float values (`price_gbp`).
  - Converted availability text to boolean indicators (`in_stock`).
  - **Missing Data Handling Policy:** Median imputation for numeric parsing anomalies; drop rows missing critical non-numeric attributes.
- **Fixed Currency Conversion:** Converts GBP to INR using a fixed baseline conversion rate of **1 GBP = 105.50 INR**.
- **Normalized SQLite Schema:** Two-table relational structure with primary/foreign key enforcement (`categories` and `books`).
- **SQL Analytics:** Demonstrates `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and explicit `JOIN`.
- **Pandas Verification:** Compares relational SQL join execution with native Pandas `pd.merge()` operations to confirm data parity.

---

## 2. Database Schema

### `categories` Table
 Column Name    Data Type  Constraints               

 `category_id`  INTEGER    PRIMARY KEY AUTOINCREMENT 
 `category_name` TEXT      UNIQUE, NOT NULL          

### `books` Table
 Column Name   | Data Type | Constraints                 

---

## 3. SQL Queries Covered

1. **`SELECT / WHERE / BETWEEN`**: Filters books priced between £20.00 and £30.00 GBP.
2. **`ORDER BY / LIMIT`**: Retrieves the top 5 most expensive books in INR.
3. **`DISTINCT`**: Returns distinct star ratings available across the catalog.
4. **`IN` Operator**: Filters high-rated books with ratings of 4 or 5.
5. **`JOIN`**: Joins `books` and `categories` tables to list the 10 highest-priced 5-star books alongside their category names.

---

## 4. Execution & Verification

Run the Python script directly:

```bash
python scraper_pipeline.py





-------------------------------------------------------------------------------------------------------------

## MODULE_2

-------------------------------------------------------------------------------------------------------------
# Titanic End-to-End Analytics & Machine Learning Pipeline

## 1. Project Overview & Offline Fallback Commitment

This project implements an end-to-end data engineering, exploratory analysis, and predictive modeling pipeline on the classic Titanic dataset. 

To ensure complete reproducibility and grading reliability without external internet dependency, the pipeline executes an offline fallback save immediately upon data load:
- **Offline Artifact Path:** `/analytics/titanic.csv`
- **Execution Mechanism:** `df.to_csv("analytics/titanic.csv", index=False)`

All exploratory data analysis, feature engineering, and predictive model pipelines build deterministically from this local CSV fallback.

---

## 2. Dataset Profiling & Missing-Value Strategy

### Dataset Profile Summary
- **Total Records:** 891 rows, 15 columns
- **Class Distribution:** `0` (Died): 549 (61.62%), `1` (Survived): 342 (38.38%)

### Measured Column Missing Rates & Applied Handling Strategies

| Column Name | Measured Missing Rate | Assigned Handling Strategy | Technical Justification |
| :--- | :--- | :--- | :--- |
| `embarked` / `embark_town` | **0.22%** (2 rows) | **Drop Rows** | Below the 5% threshold; dropping 2 rows preserves dataset integrity without introducing artificial noise. |
| `age` | **19.87%** (177 rows) | **Impute (Median)** | Falls cleanly within the 5%–30% threshold range; imputed using training set median within leak-free pipelines. |
| `deck` | **77.22%** (688 rows) | **Drop Column** | Missingness vastly exceeds 30%. Imputation or categorical dummy encoding would introduce severe noise and cause model overfitting. |

*Redundant columns dropped:* `embark_town` (duplicate of `embarked`), `class` (duplicate of `pclass`), `alive` (duplicate of `survived`), `who`, `adult_male`, and `alone` (derived features).

---

## 3. Univariate & Bivariate Exploratory Analysis

### Univariate Analysis (Age & Fare)
- **Age Outliers:** **15 outliers** detected outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ (bounds: $[-6.62, 54.88]$).
- **Fare Outliers:** **114 outliers** detected outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ (bounds: $[-26.76, 65.63]$).
- **Fare Central Tendencies:**
  - **Mean:** £32.20
  - **Median:** £14.45
  - **Mode:** £8.05
- **Skewness Conclusion:** Because $\text{Mean } (32.20) > \text{Median } (14.45) > \text{Mode } (8.05)$, the Fare distribution exhibits extreme **right-skewness** with a long right tail driven by high-value luxury tickets.

### Bivariate Analysis & Correlation Breakdown
- **Survival Rate by Sex:** Female: **74.04%** | Male: **18.89%**
- **Survival Rate by Pclass:** 1st Class: **62.62%** | 2nd Class: **47.28%** | 3rd Class: **24.24%**
- **Survival Rate by Sex & Class:**
  - 1st Class Female: **96.74%** | 1st Class Male: **36.89%**
  - 3rd Class Female: **50.00%** | 3rd Class Male: **13.54%**

#### Top 2 Strongest Off-Diagonal Correlations (6x6 Matrix)
1. **`pclass` and `fare` ($r = -0.55$):** Strong inverse relationship reflecting that 1st class tickets (`pclass=1`) carried significantly higher purchase prices.
2. **`pclass` and `age` ($r = -0.37$):** Moderate inverse correlation indicating older passengers were more likely to afford 1st class accommodations.

---

## 4. Multivariate Data Story: Survival Mechanics

1. **Survival by Class and Sex:** Displays the compound effect of the "women and children first" maritime code alongside socioeconomic status. 1st-class females achieved a near-perfect **96.74%** survival rate, whereas 3rd-class males suffered a dismal **13.54%** survival rate.
2. **Log Fare Distribution by Class and Survival:** Reveals that across every passenger class, individuals who survived paid a higher median fare than those who perished, proving that cabin proximity to lifeboat decks favored high-tariff passengers.
3. **Age Distribution by Sex and Survival:** Shows clear survival prioritization for young children (under age 10) across both sexes, while adult males between 18 and 40 suffered the highest mortality rates.
4. **Survival Rate by Family Size:** Exhibits an inverted-U survival curve. Solo travelers (`family_size = 1`) had low survival rates (~30%), moderate families of 2–4 members peaked at over ~55–70% survival due to group coordination, and large families ($\ge 5$) suffered sharply reduced survival due to evacuation bottlenecks.

---

## 5. Preprocessing & Leak-Free Pipeline Design

To guarantee zero data leakage from test data into training representations:
- Split dataset first using `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`. Stratification ensures train and test splits strictly preserve the baseline **38.38%** target class ratio.
- All imputation, standard scaling, and one-hot encoding operations are encapsulated inside a scikit-learn `ColumnTransformer` within a master `Pipeline`.
- The pipeline calls `.fit()` exclusively on `X_train` and transforms `X_test` strictly in transform-only mode.

---

## 6. Model Evaluation & Comparison

### Classification Metrics Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7865 | 0.7385 | 0.7059 | 0.7218 | 0.8529 |
| **Decision Tree (Max Depth = 3)** | 0.7921 | 0.7719 | 0.6471 | 0.7040 | 0.8351 |
| **Random Forest (Baseline)** | 0.8146 | 0.7761 | 0.7353 | 0.7552 | 0.8601 |
| **Tuned Random Forest** | **0.8258** | **0.7846** | **0.7500** | **0.7669** | **0.8687** |

### Imbalance Strategy Comparison (Logistic Regression)

| Strategy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **(a) Baseline (No Handling)** | 0.7385 | 0.7059 | 0.7218 |
| **(b) Class Weight Balanced** | 0.7164 | **0.7059** | 0.7111 |
| **(c) Train-Fold SMOTE** | **0.7206** | **0.7206** | **0.7206** |

*Conclusion:* Given the mild class balance (~38% positive class), baseline training yields strong results. Applying **Train-Fold SMOTE** slightly improves recall equality without suffering the precision degradation observed under simple class-weight balancing.

### Hyperparameter Tuning & Out-of-Bag (OOB) Reporting
- **Tuned Hyperparameters:** `n_estimators=100`, `max_depth=5`, `max_features='sqrt'`
- **Out-of-Bag (OOB) Score:** **0.8169** (evaluated via `RandomForestClassifier(oob_score=True)` on the training fold).

---

## 7. Regression Side-Task: Predicting Fare

### Metric Performance Table

| Model | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Fare)** | 19.3821 | 33.4158 | 0.3852 | 0.3598 |

### Heteroscedasticity Analysis
The residual plot exhibits severe **heteroscedasticity** (a non-random fan-shaped spread of residuals). At low predicted fare values, residuals are tightly clustered near zero, whereas at high predicted values, residual variance explodes up to $\pm 300$. This indicates that linear models violate the homoscedasticity assumption on raw ticket fares, further justifying log-transformations for monetary targets.

---

## 8. Final Deployment Recommendation

**Deployment Recommendation:** The **Tuned Random Forest Classifier** is selected for production deployment. 

**Justification:** It achieves superior performance across all evaluation metrics, reaching an **Accuracy of 82.58%**, an **F1 Score of 0.7669**, and an **AUC of 0.8687**. Compared to Logistic Regression and Decision Trees, the Random Forest effectively captures non-linear interactions between class, family size, and gender while maintaining high generalization stability verified by its **0.8169 OOB Score**.

---

## 9. Model Serialization & Live Pipeline Verification

The final production model is serialized as an end-to-end scikit-learn Pipeline incorporating preprocessor steps and trained estimator weights:
- **Saved Pipeline Path:** `analytics/titanic_best_pipeline.joblib`

### Verification Script Excerpt
```python
import joblib
import pandas as pd

# Load pipeline from disk
pipeline = joblib.load("analytics/titanic_best_pipeline.joblib")

# Pass raw, unpreprocessed data directly
raw_data = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S",
        }
    ]
)

prediction = pipeline.predict(raw_data)
probability = pipeline.predict_proba(raw_data)[:, 1]
print(
    f"Prediction: {prediction[0]} | Survival Probability: {probability[0]:.4f}"
)



