# ============================================================
# 📊 EXPLORATORY DATA ANALYSIS (EDA) – INITIAL INSPECTION
# ============================================================

# Import required libraries
# pandas → data manipulation
# seaborn → statistical visualization
# matplotlib → plotting graphs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, precision_score, recall_score
)

# ============================================================
# 1️⃣ LOAD DATASET
# ============================================================

# Load the dataset from CSV file
# Make sure the file path is correct relative to your project directory
df = pd.read_csv("data/boston.csv")
# df = pd.read_csv("data/housing.csv", delim_whitespace=True)
# df = pd.read_csv("data/boston.csv", sep=r"\s+")

# ============================================================
# 2️⃣ BASIC STRUCTURE OVERVIEW
# ============================================================

# Display first 5 rows
# Helps understand column names and sample data
print("Head of the dataset:")
print(df.head())

# Check dataset shape
# shape → (number_of_rows, number_of_columns)
print("\nDataset Shape:")
print(df.shape)

# Display column names
# Helps understand feature list
print("\nDataset Columns:")
print(df.columns)

# ============================================================
# 3️⃣ DATA TYPES & NULL CHECK
# ============================================================

# df.info() shows:
# - Data types of each column
# - Non-null counts
# - Memory usage (summary level)
# Helps detect missing values and incorrect data types
print("\nDataset Info:")
print(df.info())

# ============================================================
# 4️⃣ MEMORY USAGE ANALYSIS
# ============================================================

# Check detailed memory usage of each column
# deep=True ensures object columns are fully calculated
# Important for large datasets
print("\nMemory Usage (in bytes):")
print(df.memory_usage(deep=True))

# ============================================================
# 📂 FEATURE TYPE IDENTIFICATION
# ============================================================

# ------------------------------------------------------------
# 1️⃣ Identify Numerical Columns
# ------------------------------------------------------------

# select_dtypes() filters columns based on datatype
# include=['int64','float64'] selects numeric columns
# .columns returns column names only
# These columns will be used for:
# - Scaling
# - Outlier detection
# - Correlation analysis
num_cols = df.select_dtypes(include=['int64','float64']).columns


# ------------------------------------------------------------
# 2️⃣ Identify Categorical Columns
# ------------------------------------------------------------

# include=['object'] selects string/text columns
# These columns typically require encoding
cat_cols = df.select_dtypes(include=['object']).columns

# ------------------------------------------------------------
# 3️⃣ Display Results
# ------------------------------------------------------------

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)

# ============================================================
# 🔎 MISSING VALUES ANALYSIS
# ============================================================

print("\nMissing Values Analysis")
print("="*60)

# ------------------------------------------------------------
# 1️⃣ Count of Missing Values
# ------------------------------------------------------------

# df.isna() returns True for missing values
# .sum() counts how many missing values per column
print("\nMissing Values (Count):")
print(df.isna().sum())

# ------------------------------------------------------------
# 2️⃣ Missing Values Percentage (Manual Calculation)
# ------------------------------------------------------------

# Divide missing count by total rows
# Multiply by 100 to convert to percentage
print("\nMissing Values Percentage (Manual):")
print(df.isna().sum() / len(df) * 100)

# ------------------------------------------------------------
# 3️⃣ Missing Values Percentage (Using Mean)
# ------------------------------------------------------------

# df.isna() converts:
# True  → 1
# False → 0
# .mean() calculates proportion of missing values
# Multiply by 100 for percentage
print("\nMissing Values Percentage (Using Mean):")
print(df.isna().mean() * 100)

print("="*60)

# ============================================================
# 🎯 TARGET VARIABLE ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1️⃣ Target Distribution (Count)
# ------------------------------------------------------------

# value_counts() counts how many samples belong to each class
# Used to check class imbalance
print("\nTarget Variable Distribution:")
print(df['MEDV'].value_counts())


# ------------------------------------------------------------
# 2️⃣ Target Distribution (Percentage)
# ------------------------------------------------------------

# normalize=True converts counts to proportions
# Multiply by 100 for percentage format
print("\nTarget Variable Distribution (Percentage):")
print(df['MEDV'].value_counts(normalize=True) * 100)

# ============================================================
# 📊 NUMERICAL FEATURE SUMMARY
# ============================================================

# describe() gives:
# - count
# - mean
# - std
# - min
# - 25%, 50%, 75%
# - max
# Helps detect extreme values and scale differences
# df["Amount_log"] = np.log1p(df["Amount"])
# columns_to_describe = "Amount_log"
print(df[num_cols].describe())

# ============================================================
# 📈 VISUALIZATION & DISTRIBUTION ANALYSIS
# ============================================================

# # Loop through each numerical column
# for col in num_cols:

#     # --------------------------------------------------------
#     # Boxplot
#     # --------------------------------------------------------
#     # Used to detect outliers via IQR method
#     plt.figure()
#     sns.boxplot(x=df[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

#     # --------------------------------------------------------
#     # Histogram + KDE
#     # --------------------------------------------------------
#     # Histogram shows distribution
#     # KDE shows smooth probability density curve
#     plt.figure()
#     sns.histplot(df[col], kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.show()

#     # --------------------------------------------------------
#     # KDE Plot Using hue Parameter
#     # --------------------------------------------------------
#     # KDE shows smooth probability density curve
#     # 'hue' automatically separates Normal (0) and Fraud (1)
#     # Useful to visually compare overlap between classes
#     plt.figure()
#     sns.kdeplot(data=df, x=col, hue='MEDV')
#     plt.title(f"{col} Distribution by MEDV")
#     plt.show()

#     # --------------------------------------------------------
#     # Manual KDE Plot (Separate Class Filtering)
#     # --------------------------------------------------------
#     # Explicitly filter Normal transactions (Class = 0)
#     # Explicitly filter Fraud transactions (Class = 1)
#     # Gives more control if custom styling is needed
#     plt.figure()
#     sns.kdeplot(data=df[df['Class']==0], x=col, label='Normal')
#     sns.kdeplot(data=df[df['Class']==1], x=col, label='Fraud')
#     plt.legend()
#     plt.title(f"{col} Distribution by Class")
#     plt.show()

#     # --------------------------------------------------------
#     # Skewness
#     # --------------------------------------------------------
#     # Measures symmetry of distribution
#     # Positive → Right skew
#     # Negative → Left skew
#     # Near 0 → Symmetric
#     print(f"{col} Skew:", df[col].skew())

#     # --------------------------------------------------------
#     # Mean Comparison by Class
#     # --------------------------------------------------------
#     # Calculates average value of feature for each class
#     # Helps identify direction of shift (which class has higher/lower mean)
#     print(f"{col} Mean:", df.groupby("MEDV")[col].mean())


# Loop through each numerical column
for col in num_cols:

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------
    # Used to detect outliers via IQR method
    # plt.figure()
    # sns.boxplot(x=df[col])
    # plt.title(f"Boxplot of {col}")
    # plt.show()

    # --------------------------------------------------------
    # Histogram + KDE
    # --------------------------------------------------------
    # Histogram shows distribution
    # KDE shows smooth probability density curve
    # plt.figure()
    # sns.histplot(df[col], kde=True)
    # plt.title(f"Distribution of {col}")
    # plt.show()

    # --------------------------------------------------------
    # Scatter Plot vs Target (VERY IMPORTANT in regression)
    # --------------------------------------------------------
    # Shows relationship between feature and target
    # Helps detect:
    # - Linear relationship
    # - Non-linearity
    # - Heteroscedasticity
    # - Outliers
    # plt.figure()
    # sns.scatterplot(x=df[col], y=df["MEDV"])
    # #sns.regplot(x=df["CRIM"], y=df["MEDV"], scatter_kws={"alpha":0.5})
    # plt.title(f"{col} vs MEDV")
    # plt.xlabel(col)
    # plt.ylabel("MEDV")
    # plt.show()

    # plt.figure()
    # #sns.scatterplot(x=df[col], y=df["MEDV"])
    # sns.regplot(x=df[col], y=df["MEDV"], scatter_kws={"alpha":0.5})
    # plt.title(f"{col} vs MEDV")
    # plt.xlabel(col)
    # plt.ylabel("MEDV")
    # plt.show()

    # --------------------------------------------------------
    # Correlation with Target
    # --------------------------------------------------------
    # Measures linear relationship strength
    print(f"{col} Correlation with MEDV:",
          df[col].corr(df["MEDV"]))

    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------
    # Measures symmetry of distribution
    print(f"{col} Skew:", df[col].skew())

    print("-" * 60)


# ============================================================
# 🚨 OUTLIER DETECTION USING IQR METHOD
# ============================================================
# This section identifies potential outliers in all numerical columns
# using the Interquartile Range (IQR) method.
#
# IQR Method Logic:
# 1. Compute Q1 (25th percentile)
# 2. Compute Q3 (75th percentile)
# 3. Calculate IQR = Q3 - Q1
# 4. Define lower bound = Q1 - 1.5 * IQR
# 5. Define upper bound = Q3 + 1.5 * IQR
# 6. Any values outside these bounds are considered potential outliers
#
# NOTE:
# This does NOT remove outliers.
# It only measures their percentage for analysis purposes.
# ============================================================

for col in num_cols:

    # Calculate 1st Quartile (25th percentile)
    Q1 = df[col].quantile(0.25)

    # Calculate 3rd Quartile (75th percentile)
    Q3 = df[col].quantile(0.75)

    # Compute Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define lower boundary for outlier detection
    # Any value below this will be considered an outlier
    lower = Q1 - 1.5 * IQR

    # Define upper boundary for outlier detection
    # Any value above this will be considered an outlier
    upper = Q3 + 1.5 * IQR

    # Identify rows where values fall outside the lower and upper bounds
    # These rows are potential outliers
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    # Calculate and print percentage of outliers in this column
    # (number of outlier rows divided by total dataset size)
    print(col, "Outlier %:", len(outliers)/len(df)*100)

# ============================================================
# ✅ END OF OUTLIER DETECTION SECTION
# ============================================================


# ============================================================
# 📊 CORRELATION MATRIX ANALYSIS (Numerical vs Numerical)
# ============================================================
# This section computes and visualizes the correlation matrix
# for all numerical features in the dataset.
#
# Purpose:
# 1. Measure linear relationship between numerical variables
# 2. Detect strong positive or negative correlations
# 3. Identify potential multicollinearity issues
# 4. Check correlation strength with the target variable
#
# Correlation Value Range:
# +1   → Perfect positive linear relationship
#  0   → No linear relationship
# -1   → Perfect negative linear relationship
#
# What to Check:
# - Any correlation > 0.8 or < -0.8? (Possible multicollinearity)
# - Which features strongly correlate with target?
# - Are some features redundant?
#
# IMPORTANT:
# Correlation ≠ Causation
# A strong correlation does NOT mean one variable causes the other.
# ============================================================

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# TARGET COLUMN NAME
# --------------------------------------------------------
# Specify your target column (change if needed)
target_col = "MEDV"


# --------------------------------------------------------
# SELECT NUMERIC FEATURES (EXCLUDING TARGET)
# --------------------------------------------------------
# Select all numeric columns (int64, float64)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Remove target column from numeric feature list
# We don't want to compare target with itself
num_cols = num_cols.drop(target_col)


# --------------------------------------------------------
# CORRELATION WITH TARGET
# --------------------------------------------------------
# Compute correlation of all numeric features with target
# abs() is used because:
#   - We care about strength, not direction (+/-)
#   - Both strong positive and strong negative are useful

corr_with_target = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)


# --------------------------------------------------------
# SELECT TOP 5 MOST CORRELATED FEATURES
# --------------------------------------------------------
# index[0] is the target itself (correlation = 1)
# So we skip it using [1:6]
# This selects top 5 features most correlated with target

top_features = corr_with_target.index[1:6]


# --------------------------------------------------------
# BOX PLOT: TOP 5 FEATURES VS TARGET
# --------------------------------------------------------
# Purpose:
# - Focus only on strongest predictors
# - Clear visual comparison
# - Helps understand which features drive prediction most

for col in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f"{col} vs {target_col}")
    plt.tight_layout()
    plt.show()



# print("\nTarget Variable Summary Statistics:")
# print(df['MEDV'].describe())
# print("\nTarget Variable Unique Values:")
# print(df['MEDV'].unique())
# print("\nTarget Variable Value Counts:")
# print(df['MEDV'].value_counts())
# print("\nTarget Variable Value Counts (Percentage):")
# print(df['MEDV'].value_counts(normalize=True) * 100)
# print("\nTarget Variable Distribution (Histogram):")
# sns.histplot(df['MEDV'], bins=30, kde=True)
# plt.title('Distribution of Target Variable (MEDV)')
# plt.xlabel('MEDV')
# plt.ylabel('Frequency') 
# plt.show()
# print("\nTarget Variable Distribution (Boxplot):")
# sns.boxplot(x=df['MEDV'])
# plt.title('Boxplot of Target Variable (MEDV)')      
# plt.xlabel('MEDV')
# plt.show()
# print("\nTarget Variable Distribution (Violin Plot):")
