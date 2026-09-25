import os
import pandas as pd
import seaborn as sns

# Create the required analytics directory if it doesn't exist
os.makedirs("analytics", exist_ok=True)

# 1. Load the dataset (The one and only raw load)
df = sns.load_dataset("titanic")

# 2. Profile the dataset
print("=== DATAFRAME INFO ===")
df.info()

print("\n=== SUMMARY STATISTICS ===")
print(df.describe(include="all"))

print("\n=== DATAFRAME SHAPE ===")
print(f"Shape: {df.shape}")

# 3. Compute and report the percentage of missing values for columns that have them
print("\n=== MISSING VALUES PERCENTAGE ===")
missing_pct = (df.isnull().sum() / len(df)) * 100
columns_with_missing = missing_pct[missing_pct > 0]

if not columns_with_missing.empty:
    for col, pct in columns_with_missing.items():
        print(f"Column '{col}': {pct:.2f}% missing values")
else:
    print("No missing values found in any column.")

# 4. Save the loaded DataFrame as a committed offline fallback inside /analytics
fallback_path = os.path.join("analytics", "titanic.csv")
df.to_csv(fallback_path, index=False)

print(f"\n[SUCCESS] Cached fallback copy saved to '{fallback_path}' for grading.")

import numpy as np

# ==========================================
# Task 2: Missing-Value Handling
# ==========================================
print("\n--- TASK 2: MISSING-VALUE HANDLING ---")

# Create a deep copy to preserve the original raw DataFrame
cleaned_df = df.copy()

# Iterate through each column that contains missing values identified in Task 1
for col in columns_with_missing.index:
    pct = missing_pct[col]

    if pct < 5.0:
        # Strategy 1: Missingness < 5% -> Drop rows
        print(
            f"Strategy for '{col}' ({pct:.2f}% missing < 5%): Dropping rows."
        )
        cleaned_df = cleaned_df.dropna(subset=[col])

    elif 5.0 <= pct <= 30.0:
        # Strategy 2: Missingness between 5% and 30% -> Impute
        print(f"Strategy for '{col}' ({pct:.2f}% missing): Imputing values.")

        # If data is numerical (e.g., Age), impute using the Median
        if np.issubdtype(cleaned_df[col].dtype, np.number):
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
        # If data is categorical/text, impute using the Mode
        else:
            mode_val = cleaned_df[col].mode()[0]
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)

    else:
        # Strategy 3: Missingness > 30% -> Encode as a new category
        print(
            f"Strategy for '{col}' ({pct:.2f}% missing > 30%): Encoding 'missing' as a new category."
        )

        # If it's a Pandas categorical type, explicitly expand the allowed categories first
        if isinstance(cleaned_df[col].dtype, pd.CategoricalDtype):
            cleaned_df[col] = cleaned_df[col].cat.add_categories("missing")

        # Fill all null values with the label "missing"
        cleaned_df[col] = cleaned_df[col].fillna("missing")

# Report the data transformations by checking dataset size changes
print(f"\nOriginal Dataset Shape: {df.shape}")
print(f"Cleaned Dataset Shape: {cleaned_df.shape}\n")

# 3.  plot a histogram and a box plot for both age and fare.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Histogram + Box Plot
# -----------------------------

# 1, use the cleaned (df)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Age histogram
sns.histplot(df["age"].dropna(), kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Frequency")

# Age box plot
sns.boxplot(x=df["age"].dropna(), ax=axes[0, 1])
axes[0, 1].set_title("Age Box Plot")
axes[0, 1].set_xlabel("Age")

# Fare histogram
sns.histplot(df["fare"].dropna(), kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare Distribution")
axes[1, 0].set_xlabel("Fare")
axes[1, 0].set_ylabel("Frequency")

# Fare box plot
sns.boxplot(x=df["fare"].dropna(), ax=axes[1, 1])
axes[1, 1].set_title("Fare Box Plot")
axes[1, 1].set_xlabel("Fare")

plt.tight_layout()
plt.show()

# 2. IQR outlier calculation

def iqr_outliers(series):
    series = series.dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[
        (series < lower_bound) | (series > upper_bound)
    ]

    return Q1, Q3, IQR, lower_bound, upper_bound, len(outliers)


age_result = iqr_outliers(df["age"])
fare_result = iqr_outliers(df["fare"])

print("AGE")
print("Q1:", age_result[0])
print("Q3:", age_result[1])
print("IQR:", age_result[2])
print("Lower bound:", age_result[3])
print("Upper bound:", age_result[4])
print("Number of outliers:", age_result[5])

print("\nFARE")
print("Q1:", fare_result[0])
print("Q3:", fare_result[1])
print("IQR:", fare_result[2])
print("Lower bound:", fare_result[3])
print("Upper bound:", fare_result[4])
print("Number of outliers:", fare_result[5])

# 2. IQR outlier calculation

def iqr_outliers(series):
    series = series.dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[
        (series < lower_bound) | (series > upper_bound)
    ]

    return Q1, Q3, IQR, lower_bound, upper_bound, len(outliers)


age_result = iqr_outliers(df["age"])
fare_result = iqr_outliers(df["fare"])

print("AGE")
print("Q1:", age_result[0])
print("Q3:", age_result[1])
print("IQR:", age_result[2])
print("Lower bound:", age_result[3])
print("Upper bound:", age_result[4])
print("Number of outliers:", age_result[5])

print("\nFARE")
print("Q1:", fare_result[0])
print("Q3:", fare_result[1])
print("IQR:", fare_result[2])
print("Lower bound:", fare_result[3])
print("Upper bound:", fare_result[4])
print("Number of outliers:", fare_result[5])

# 3. Fare mean, median and mode

fare = df["fare"].dropna()

fare_mean = fare.mean()
fare_median = fare.median()
fare_mode = fare.mode().iloc[0]

print("Fare Mean:", fare_mean)
print("Fare Median:", fare_median)
print("Fare Mode:", fare_mode)

# 4. Bivariate Analysis, use the same cleaned df from 01_eda.ipynb. The following code explicitly uses boolean masking with & and |, as requested.

# 1. Survival rate by sex using boolean masking

male_mask = df["sex"] == "male"
female_mask = df["sex"] == "female"

male_survival_rate = df.loc[male_mask, "survived"].mean() * 100
female_survival_rate = df.loc[female_mask, "survived"].mean() * 100

print(f"Male survival rate: {male_survival_rate:.2f}%")
print(f"Female survival rate: {female_survival_rate:.2f}%")

# 2. Survival rate by passenger class using boolean masking

first_class = df["pclass"] == 1
second_class = df["pclass"] == 2
third_class = df["pclass"] == 3

first_survival_rate = df.loc[first_class, "survived"].mean() * 100
second_survival_rate = df.loc[second_class, "survived"].mean() * 100
third_survival_rate = df.loc[third_class, "survived"].mean() * 100

print(f"1st class survival rate: {first_survival_rate:.2f}%")
print(f"2nd class survival rate: {second_survival_rate:.2f}%")
print(f"3rd class survival rate: {third_survival_rate:.2f}%")

# 3. Survival rate by sex AND pclass

# Female + 1st class
female_first = (df["sex"] == "female") & (df["pclass"] == 1)

# Female + 2nd class
female_second = (df["sex"] == "female") & (df["pclass"] == 2)

# Female + 3rd class
female_third = (df["sex"] == "female") & (df["pclass"] == 3)

# Male + 1st class
male_first = (df["sex"] == "male") & (df["pclass"] == 1)

# Male + 2nd class
male_second = (df["sex"] == "male") & (df["pclass"] == 2)

# Male + 3rd class
male_third = (df["sex"] == "male") & (df["pclass"] == 3)


print("Female, 1st class:",
      df.loc[female_first, "survived"].mean() * 100)

print("Female, 2nd class:",
      df.loc[female_second, "survived"].mean() * 100)

print("Female, 3rd class:",
      df.loc[female_third, "survived"].mean() * 100)

print("Male, 1st class:",
      df.loc[male_first, "survived"].mean() * 100)

print("Male, 2nd class:",
      df.loc[male_second, "survived"].mean() * 100)

print("Male, 3rd class:",
      df.loc[male_third, "survived"].mean() * 100)

# Passengers who were either female OR in 1st class
mask = (df["sex"] == "female") | (df["pclass"] == 1)

result = df.loc[mask]

print(result[["sex", "pclass", "survived"]].head())

# 4. Correlation matrix — exactly six column
corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[corr_cols].corr()

print(corr_matrix)

print(df[['adult_male', 'alone']].head())

# 5. Heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix of Titanic Numeric Features")
plt.tight_layout()
plt.show()

# 6.  the two strongest correlations automatically
import numpy as np

# Absolute correlations
abs_corr = corr_matrix.abs()

# Keep only upper triangle to avoid duplicate pairs
upper_triangle = abs_corr.where(
    np.triu(np.ones(abs_corr.shape), k=1).astype(bool)
)

# Convert to pairs and sort
strongest_pairs = (
    upper_triangle
    .stack()
    .sort_values(ascending=False)
)

print("Top 2 strongest correlations:")
print(strongest_pairs.head(2))

top_two = strongest_pairs.head(2)

for (feature1, feature2), abs_value in top_two.items():
    actual_value = corr_matrix.loc[feature1, feature2]

    print(
        f"{feature1} vs {feature2}: "
        f"correlation = {actual_value:.3f}, "
        f"absolute correlation = {abs_value:.3f}"
    )

    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Survival rate by sex and passenger class

plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 2. Survival rate by passenger class
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 3. Age distribution by survival
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.tight_layout()
plt.show()

# 4. Fare versus age, separated by survival
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()

 # 5. urvival rate by sex
plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

# Select the two columns
scale_cols = ["age", "fare"]

# Keep the original values for comparison
before = df[scale_cols].copy()

# Standardize using the FULL cleaned DataFrame
scaler = StandardScaler()

df_eda_scaled = df.copy()

df_eda_scaled[scale_cols] = scaler.fit_transform(
    df[scale_cols]
)

# -----------------------------
# Before standardization
# -----------------------------
print("BEFORE STANDARDIZATION")
print(
    before[scale_cols]
    .agg(["mean", "std"])
    .round(4)
)

# -----------------------------
# After standardization
# -----------------------------
print("\nAFTER STANDARDIZATION")
print(
    df_eda_scaled[scale_cols]
    .agg(["mean", "std"])
    .round(4)
)

print("\nAFTER STANDARDIZATION - exact StandardScaler check")

means = df_eda_scaled[scale_cols].mean()
stds_pop = df_eda_scaled[scale_cols].apply(lambda x: x.std(ddof=0))

# Combine them into a single DataFrame
combined_results = pd.DataFrame({
    'mean': means,
    'std_pop': stds_pop
}).T # Transpose to get aggregations as index

print(combined_results.round(6))

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Age - before
sns.histplot(before["age"].dropna(), kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age Before Standardization")
axes[0, 0].set_xlabel("Age")

# Age - after
sns.histplot(df_eda_scaled["age"].dropna(), kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Age After Standardization")
axes[0, 1].set_xlabel("Standardized Age")

# Fare - before
sns.histplot(before["fare"].dropna(), kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare Before Standardization")
axes[1, 0].set_xlabel("Fare")

# Fare - after
sns.histplot(df_eda_scaled["fare"].dropna(), kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Fare After Standardization")
axes[1, 1].set_xlabel("Standardized Fare")

plt.tight_layout()
plt.show()

# Part B — Predictive modeling, continuing from the same cleaned data

# Task 7, split the cleaned data into training and testing sets before doing any modeling preprocessing
# 1. Separate features and target
from sklearn.model_selection import train_test_split

# Target
y = df["survived"]

# Features
X = df.drop(columns=["survived"])

print("X shape:", X.shape)
print("y shape:", y.shape)

# 2. Check the original class balance
print("Original class distribution:")
print(y.value_counts())

print("\nOriginal class proportions:")
print(y.value_counts(normalize=True))

# 3. Stratified train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Training class proportions:")
print(y_train.value_counts(normalize=True))

print("\nTesting class proportions:")
print(y_test.value_counts(normalize=True))

numeric_features = [
     "age",
     "sibsp",
     "parch",
     "fare",
     "pclass"
]
categorical_features = [
       "sex",
       "embarked"
]

# 2. Define preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Numeric preprocessing
numeric_pipeline = Pipeline([
     ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
])
# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
     ("categorical", categorical_pipeline, categorical_features)
])

# 3. Create the modeling pipeline
from sklearn.linear_model import LogisticRegression
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
     ("classifier", LogisticRegression(max_iter=1000))
])

# 3. Create the modeling pipeline
from sklearn.linear_model import LogisticRegression
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
     ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Verify missing values

print("Missing values in training data:")
print(X_train[numeric_features + categorical_features].isnull().sum())

print("\nMissing values in test data:")
print(X_test[numeric_features + categorical_features].isnull().sum())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------
# 1. Load Data & Split FIRST
# ------------------------------------------------------------------
# Example setup assuming standard Titanic dataset structure
# df = pd.read_csv('titanic.csv')

# Define target and feature columns
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

X = df[numeric_features + categorical_features]
y = df['survived']

# Perform train/test split before ANY preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 2. Define Feature-Specific Sub-Pipelines
# ------------------------------------------------------------------
# Numeric pipeline: Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute missing values with most frequent class, then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ------------------------------------------------------------------
# 3. Combine Pipelines with ColumnTransformer
# ------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------------
# 4. Wrap Preprocessor & Estimator into a Single Master Pipeline
# ------------------------------------------------------------------
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ------------------------------------------------------------------
# 5. Fit on Train ONLY, Transform/Predict on Test
# ------------------------------------------------------------------
# Fitting the master pipeline fits all imputers, encoders, scalers,
# and the classifier strictly on X_train and y_train.
full_pipeline.fit(X_train, y_train)

# Evaluating on X_test automatically applies transform-only mode
# using the parameters learned from X_train.
test_accuracy = full_pipeline.score(X_test, y_test)
y_pred = full_pipeline.predict(X_test)

print(f"Test Set Accuracy: {test_accuracy:.4f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

# ------------------------------------------------------------------
# 1. Setup Preprocessing Pipeline
# ------------------------------------------------------------------
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

X = df[numeric_features + categorical_features]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# ------------------------------------------------------------------
# 2. Define Models
# ------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    # Depth capped at 3 to make the visualization readable
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Store results for final evaluation table
results = []
trained_pipelines = {}

# ------------------------------------------------------------------
# 3. Fit Models and Evaluate Performance
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Save results
    results.append({
        'Model': name,
        'Confusion Matrix (TN, FP, FN, TP)': cm.ravel().tolist(),
        'Accuracy': f"{acc:.4f}",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}",
        'F1 Score': f"{f1:.4f}",
        'ROC AUC': f"{auc:.4f}"
    })

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------------
# 4. Render Decision Tree with Feature and Class Names
# ------------------------------------------------------------------
dt_pipeline = trained_pipelines['Decision Tree']
dt_model = dt_pipeline.named_steps['classifier']

# Extract encoded feature names from preprocessor
cat_encoder = dt_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + encoded_cat_features

plt.figure(figsize=(16, 10))
plot_tree(
    dt_model,
    feature_names=all_feature_names,
    class_names=['Died', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization (Max Depth = 3)')
plt.show()

# ------------------------------------------------------------------
# 5. Display Comparison Table
# ------------------------------------------------------------------
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# imbalanced-learn pipeline handles SMOTE fit-on-train behavior automatically
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------
# 1. Report Class Balance
# ------------------------------------------------------------------
class_counts = y.value_counts()
class_props = y.value_counts(normalize=True) * 100

print("--- Class Balance (Full Dataset) ---")
print(f"Not Survived (0): {class_counts[0]} ({class_props[0]:.2f}%)")
print(f"Survived (1):     {class_counts[1]} ({class_props[1]:.2f}%)")

# ------------------------------------------------------------------
# 2. Shared Setup (Preprocessing & Split)
# ------------------------------------------------------------------
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_features + categorical_features],
    df['survived'],
    test_size=0.2,
    random_state=42,
    stratify=df['survived']
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', ImbPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# ------------------------------------------------------------------
# 3. Train & Evaluate the Three Variants (Random Forest)
# ------------------------------------------------------------------
variants = {
    "(a) Baseline (No Handling)": ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    "(b) Class Weight 'balanced'": ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ]),
    "(c) SMOTE Oversampling": ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # SMOTE applied ONLY to train fold
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}

imbalance_results = []

for name, pipeline in variants.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    imbalance_results.append({
        "Strategy": name,
        "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
        "Precision": f"{precision_score(y_test, y_pred):.4f}",
        "Recall": f"{recall_score(y_test, y_pred):.4f}",
        "F1 Score": f"{f1_score(y_test, y_pred):.4f}"
    })

# ------------------------------------------------------------------
# 4. Display Comparison
# ------------------------------------------------------------------
df_imbalance = pd.DataFrame(imbalance_results)
print("\n--- Imbalance Strategy Comparison ---")
print(df_imbalance.to_string(index=False))

# 12. Hyperparameter tuning:
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------
# 1. Setup Data Split and Preprocessor
# ------------------------------------------------------------------
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

X = df[numeric_features + categorical_features]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# ------------------------------------------------------------------
# 2. Build Pipeline with oob_score=True
# ------------------------------------------------------------------
# Note: bootstrap=True is required for OOB scoring (default in RandomForestClassifier)
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(oob_score=True, random_state=42))
])

# ------------------------------------------------------------------
# 3. Define Hyperparameter Search Space
# ------------------------------------------------------------------
# Target parameters inside the 'classifier' step of the Pipeline
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 8, None],
    'classifier__max_features': ['sqrt', 'log2', 0.5]
}

# ------------------------------------------------------------------
# 4. Run GridSearchCV on Training Data Only
# ------------------------------------------------------------------
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fits preprocessor + estimator on train fold across CV splits
grid_search.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. Extract Best Parameters and OOB Score
# ------------------------------------------------------------------
best_pipeline = grid_search.best_estimator_
best_rf_model = best_pipeline.named_steps['classifier']

# Clean parameter keys by stripping 'classifier__' prefix for display
clean_best_params = {
    k.replace('classifier__', ''): v
    for k, v in grid_search.best_params_.items()
}

print("--- GridSearchCV Hyperparameter Tuning Results ---")
print(f"Best Parameters:         {clean_best_params}")
print(f"Best 5-Fold CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Corresponding OOB Score: {best_rf_model.oob_score_:.4f}")

# Evaluate tuned pipeline on unseen test data
test_accuracy = best_pipeline.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_accuracy:.4f}")

# 13. Regression side-task:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. Prepare Features (X) and Target (y = 'fare')
# ------------------------------------------------------------------
# Remove 'fare' from features and drop rows where 'fare' is missing
df_reg = df.dropna(subset=['fare']).copy()

numeric_features = ['age', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

X = df_reg[numeric_features + categorical_features]
y = df_reg['fare']

# Split strictly into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 2. Build Preprocessing & Regression Pipeline
# ------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit on training data ONLY
reg_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = reg_pipeline.predict(X_test)
residuals = y_test - y_pred

# ------------------------------------------------------------------
# 3. Calculate Metrics (MAE, RMSE, R², Adjusted R²)
# ------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Adjusted R² formula: 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
n = len(y_test)               # Number of observations in test set
p = X_test.shape[1]           # Number of predictors
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("--- Multivariate Linear Regression Metrics (Fare Prediction) ---")
print(f"MAE:         ${mae:.2f}")
print(f"RMSE:        ${rmse:.2f}")
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {adj_r2:.4f}")

# ------------------------------------------------------------------
# 4. Generate Residual Plot
# ------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color='crimson', edgecolor='k', linewidth=0.5)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted Fare ($)')
plt.ylabel('Residuals (Actual - Predicted) ($)')
plt.title('Residual Plot: Predicted Fare vs. Residuals')
plt.grid(True, alpha=0.3)
plt.show()

import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# 1. Generate Synthetic Raw Dataset with Numerical & Categorical Features
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
raw_df = pd.DataFrame(X, columns=["num_1", "num_2", "num_3", "num_4"])
raw_df["cat_1"] = np.random.choice(["Tier_1", "Tier_2", "Tier_3"], size=1000)

# Introduce missing values to test preprocessing pipelines
raw_df.loc[10:20, "num_1"] = np.nan
raw_df.loc[30:40, "cat_1"] = np.nan

X_train, X_test, y_train, y_test = train_test_split(raw_df, y, test_size=0.2, random_state=42)

# 2. Define Preprocessing Transformers
num_features = ["num_1", "num_2", "num_3", "num_4"]
cat_features = ["cat_1"]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# 3. Create & Fit Full Pipeline (Preprocessor + GradientBoostingClassifier)
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

full_pipeline.fit(X_train, y_train)

# 4. Save Full Pipeline to Disk
model_filename = "gradient_boosting_full_pipeline.joblib"
joblib.dump(full_pipeline, model_filename)
print(f"Complete pipeline successfully saved to '{model_filename}'.")