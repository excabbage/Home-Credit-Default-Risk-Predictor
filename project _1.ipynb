"""
Goal:
    Convert the raw monthly POS/loan records (many rows per client)
    into a clean, aggregated table (one row per client) suitable for modeling.
"""

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import kurtosis, skew, entropy
import numpy.linalg as la

# 2. Load Dataset
file_path = "POS_CASH_balance.csv"
pos = pd.read_csv(file_path)

print("Data loaded successfully!")
print(f"Shape: {pos.shape}")
print("Columns:", list(pos.columns))
print(pos.head())

# 3. Basic Information and Missing Values
print("\n--- Missing Value Percentage ---")
missing_ratio = pos.isnull().mean().sort_values(ascending=False)
print(missing_ratio.head(15))

# Drop columns with >95% missing values
cols_to_drop = missing_ratio[missing_ratio > 0.95].index.tolist()
if cols_to_drop:
    pos.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns with >95% missing: {cols_to_drop}")

# Drop rows missing essential keys
pos.dropna(subset=['SK_ID_CURR', 'SK_ID_PREV'], inplace=True)

# 4. Handle Data Types and Outliers
# Ensure integer columns have correct types
pos['MONTHS_BALANCE'] = pos['MONTHS_BALANCE'].astype(int)

# Clip invalid values (SK_DPD, SK_DPD_DEF should be >= 0)
for col in ['SK_DPD', 'SK_DPD_DEF']:
    if col in pos.columns:
        pos[col] = pos[col].clip(lower=0)

# Replace negative CNT_INSTALMENT_FUTURE if exists
if 'CNT_INSTALMENT_FUTURE' in pos.columns:
    pos['CNT_INSTALMENT_FUTURE'] = pos['CNT_INSTALMENT_FUTURE'].clip(lower=0)

# 5. Encode Categorical Feature
# NAME_CONTRACT_STATUS is categorical (e.g., Active, Completed, etc.)
if 'NAME_CONTRACT_STATUS' in pos.columns:
    print("\nEncoding categorical variable: NAME_CONTRACT_STATUS")
    pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na=False)

# 6. Account-Level Aggregation (by SK_ID_PREV)
print("\n Aggregating at SK_ID_PREV level ")
prev_agg = pos.groupby('SK_ID_PREV').agg({
    'MONTHS_BALANCE': ['max', 'min', 'count'],
    'SK_DPD': ['mean', 'max', 'sum'],
    'SK_DPD_DEF': ['mean', 'max', 'sum'],
    'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'min']
})

# Flatten column names
prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
prev_agg.reset_index(inplace=True)

# Merge SK_ID_CURR from original table
prev_agg = prev_agg.merge(pos[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(),
                          on='SK_ID_PREV', how='left')

# 7. Client-Level Aggregation (by SK_ID_CURR)
print("\n Aggregating at SK_ID_CURR level")
client_agg = prev_agg.groupby('SK_ID_CURR').agg(['mean', 'max', 'min', 'std'])
client_agg.columns = ['POS_' + '_'.join(col).upper() for col in client_agg.columns]
client_agg.reset_index(inplace=True)

# 8. Additional Derived Features
# Number of POS accounts per client
n_accounts = pos.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().reset_index()
n_accounts.columns = ['SK_ID_CURR', 'POS_NUM_ACCOUNTS']

# Average DPD per client
avg_dpd = pos.groupby('SK_ID_CURR')['SK_DPD'].mean().reset_index()
avg_dpd.columns = ['SK_ID_CURR', 'POS_AVG_DPD']

# Merge additional features
client_agg = client_agg.merge(n_accounts, on='SK_ID_CURR', how='left')
client_agg = client_agg.merge(avg_dpd, on='SK_ID_CURR', how='left')

# 9. Final Check and Export
print("\n Final dataset ready!")
print(f"Shape after aggregation: {client_agg.shape}")
print(client_agg.head())

# POS_CASH_balance Feature Selection, Collinearity Check, and Feature Importance

# 1. Define interpretable candidate features
candidate_features = {
    'POS_NUM_ACCOUNTS': "Number of POS/loan accounts -> measures credit activity and exposure",
    'POS_AVG_DPD': "Average days past due -> reflects repayment discipline (higher = riskier)",
    'POS_PREV_MONTHS_BALANCE_MAX_MEAN': "Average of max months_balance per account -> account recency / activity level",
    'POS_PREV_MONTHS_BALANCE_MIN_MEAN': "Average of min months_balance per account -> earliest record (credit history depth)",
    'POS_PREV_CNT_INSTALMENT_FUTURE_MAX_MEAN': "Mean of maximum remaining instalments -> average repayment burden",
    'POS_PREV_CNT_INSTALMENT_FUTURE_MAX_MAX': "Max of maximum remaining instalments -> largest single loan obligation",
    'POS_PREV_CNT_INSTALMENT_FUTURE_MIN_MEAN': "Mean of minimum remaining instalments -> typical completed loan length",
    'POS_PREV_CNT_INSTALMENT_FUTURE_MIN_MIN': "Minimum of minimum remaining instalments -> shortest loan duration"
}

# Keep only available columns
available_cols = [c for c in candidate_features.keys() if c in client_agg.columns]
missing_cols = list(set(candidate_features.keys()) - set(available_cols))
if missing_cols:
    print(f" Missing expected columns skipped: {missing_cols}")

selected_features = ['SK_ID_CURR'] + available_cols
POS_CASH_balance_sel = client_agg[selected_features].copy()

print("\n Selected Features & Explanations")
for f in available_cols:
    print(f"{f:45s} -> {candidate_features[f]}")

# 2. Correlation heatmap visualization
corr = POS_CASH_balance_sel.drop(columns='SK_ID_CURR').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of POS_CASH_balance Features", fontsize=14, weight='bold')
plt.show()

# 3. Detect multicollinearity (|corr| > 0.8)
high_corr_pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr_pairs.append((cols[i], cols[j], corr.iloc[i, j]))

if high_corr_pairs:
    print("\n Highly correlated feature pairs (>0.8):")
    for a, b, cval in high_corr_pairs:
        print(f"  {a} <-> {b} : corr = {cval:.3f}")
else:
    print("\n No high correlations above 0.8 detected.")

# 4. Drop redundant variables to reduce overfitting risk
to_drop = set()
for a, b, _ in high_corr_pairs:
    # Prefer keeping features with broader interpretability
    drop_candidate = b if ('MAX_MAX' in b or 'MIN_MIN' in b) else a
    to_drop.add(drop_candidate)

if to_drop:
    print(f"\nDropping redundant features due to collinearity: {list(to_drop)}")
    POS_CASH_balance_sel.drop(columns=list(to_drop), inplace=True)

# 5. Log-transform heavily skewed variables
skew_threshold = 1.0
numeric_cols = POS_CASH_balance_sel.drop(columns='SK_ID_CURR').select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    skewness = POS_CASH_balance_sel[col].skew()
    if skewness > skew_threshold:
        POS_CASH_balance_sel[col] = np.log1p(POS_CASH_balance_sel[col])
        print(f"Applied log1p transform to {col} (skew={skewness:.2f})")

# 6. Variance Inflation Factor (VIF) for multicollinearity
print("\n Variance Inflation Factor (VIF) Analysis")
X_vif = POS_CASH_balance_sel.drop(columns='SK_ID_CURR').dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False))

# Drop features with VIF > 10
high_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
if high_vif:
    print(f"\nDropping high-VIF features (>10): {high_vif}")
    POS_CASH_balance_sel.drop(columns=high_vif, inplace=True)
    
# 7. Final Summary
print("\n Final POS_CASH_balance_features ready for modeling:")
print(POS_CASH_balance_sel.info())
print(POS_CASH_balance_sel.describe(percentiles=[0.05, 0.5, 0.95]))

"""
Goal:
    1. Clean the raw previous_application.csv (row level)
    2. Aggregate to customer-level (SK_ID_CURR) features
"""
# 1. Load Dataset
file_path = "previous_application.csv"  
prev = pd.read_csv(file_path)
print(f" Loaded previous_application: {prev.shape[0]:,} rows, {prev.shape[1]} columns")

# 2. Drop high-missing columns (>95%)
missing_ratio = prev.isnull().mean()
to_drop = missing_ratio[missing_ratio > 0.95].index.tolist()
if to_drop:
    prev.drop(columns=to_drop, inplace=True)
    print(f" Dropped {len(to_drop)} columns with >95% missing values")

# 3. Handle Missing Values
# Numeric to median
num_cols = prev.select_dtypes(include=[np.number]).columns
prev[num_cols] = prev[num_cols].fillna(prev[num_cols].median())

# Categorical to "Unknown"
cat_cols = prev.select_dtypes(exclude=[np.number]).columns
prev[cat_cols] = prev[cat_cols].fillna("Unknown")

# 4. Outlier Clipping
for col in num_cols:
    low, high = prev[col].quantile([0.01, 0.99])
    prev[col] = prev[col].clip(lower=low, upper=high)

# 5. Skewness Correction
skew_threshold = 1.5
for col in num_cols:
    skewness = prev[col].skew()
    if skewness > skew_threshold:
        prev[col] = np.log1p(prev[col])
        print(f"Applied log1p transform to {col} (skew={skewness:.2f})")


# 6. Auto Encoding Function for Categorical Variables
def encode_categoricals(df, max_onehot=10, drop_first=True):
    """
    Automatically encodes categorical variables:
      - Uses One-Hot Encoding if unique categories <= max_onehot
      - Uses Frequency Encoding otherwise
    """
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"\n Found {len(cat_cols)} categorical columns.")

    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        print(f"\n Encoding '{col}' ({n_unique} unique categories)")

        # Small number of categories → One-Hot Encoding
        if n_unique <= max_onehot:
            print(f"  → Using One-Hot Encoding")
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

        # Large number of categories → Frequency Encoding
        else:
            print(f"  → Using Frequency Encoding")
            freq_map = df_encoded[col].value_counts(normalize=True)
            df_encoded[col + '_FE'] = df_encoded[col].map(freq_map)
            df_encoded.drop(columns=[col], inplace=True)

    print(f"\n Encoding complete. Final shape: {df_encoded.shape}")
    return df_encoded

prev = encode_categoricals(prev, max_onehot=10, drop_first=True)

# 7. Derive ratio / useful numeric features
prev["APPLICATION_TO_CREDIT_RATIO"] = np.where(
    prev["AMT_CREDIT"] != 0, prev["AMT_APPLICATION"] / prev["AMT_CREDIT"], 0
)
prev["CREDIT_TO_GOODS_RATIO"] = np.where(
    prev["AMT_GOODS_PRICE"] != 0, prev["AMT_CREDIT"] / prev["AMT_GOODS_PRICE"], 0
)
prev["DOWN_PAYMENT_RATIO"] = np.where(
    prev["AMT_CREDIT"] != 0, prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"], 0
)
prev["AVG_PAYMENT_SIZE"] = np.where(
    prev["CNT_PAYMENT"] != 0, prev["AMT_CREDIT"] / prev["CNT_PAYMENT"], 0
)

# 8. Aggregation to customer-level (SK_ID_CURR)
# Select numeric columns (exclude identifiers)
numeric_cols = prev.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ["SK_ID_PREV", "SK_ID_CURR"]]

# Aggregate statistics
agg_dict = {col: ["mean", "max", "min", "std"] for col in numeric_cols}
prev_agg = prev.groupby("SK_ID_CURR").agg(agg_dict)

# Flatten multi-level columns
prev_agg.columns = ["PREV_" + "_".join(col).upper() for col in prev_agg.columns]
prev_agg.reset_index(inplace=True)

# 9. Derived Approval & Refusal Rates
status_cols = [c for c in prev.columns if "NAME_CONTRACT_STATUS" in c]
if status_cols:
    prev_agg["PREV_APPROVAL_RATE"] = prev.groupby("SK_ID_CURR")[status_cols[0]].mean()

# 10. Skewness correction (post-aggregation)
skew_threshold = 1.0
for col in prev_agg.select_dtypes(include=[np.number]).columns:
    if col != "SK_ID_CURR":
        skewness = prev_agg[col].skew()
        if skewness > skew_threshold:
            prev_agg[col] = np.log1p(prev_agg[col])
            print(f"Applied log1p to {col} (skew={skewness:.2f}) after aggregation")

# 11. Display results
print("\n Final aggregated dataset preview:")
print(prev_agg.head(10))
print(f"\nTotal customers: {prev_agg.shape[0]:,}")
print(f"Total features: {prev_agg.shape[1]:,}")

print("\n Summary statistics:")
print(prev_agg.describe(percentiles=[0.05, 0.5, 0.95]).T.head(10))

"""
Goal:
    Select and validate previous_application features from an economic perspective.
    Steps include:
        1. Economic feature selection and reasoning
        2. Derived feature construction
        3. Correlation filtering
        4. VIF (iterative elimination)
        5. Statistical diagnostics (Variance, PCA, KMO)
"""

# 1. Economic Feature Selection and Derived Variables
# Economic features from previous_application_agg
economic_features = [
    'PREV_AMT_CREDIT_MEAN', 'PREV_AMT_CREDIT_MAX', 'PREV_AMT_APPLICATION_MEAN',
    'PREV_AMT_ANNUITY_MEAN', 'PREV_CNT_PAYMENT_MEAN', 'PREV_RATE_DOWN_PAYMENT_MEAN',
    'PREV_CREDIT_TO_GOODS_RATIO_MEAN', 'PREV_APPLICATION_TO_CREDIT_RATIO_MEAN',
    'PREV_AMT_DOWN_PAYMENT_MEAN', 'PREV_AVG_PAYMENT_SIZE_MEAN',
    'PREV_DAYS_DECISION_MIN', 'PREV_APPROVAL_RATE'
]
available = [f for f in economic_features if f in prev_agg.columns]
prev_sel = prev_agg[['SK_ID_CURR'] + available].copy()

# Derived variables 
if set(['PREV_AMT_CREDIT_MEAN','PREV_AMT_CREDIT_MAX']).issubset(prev_sel.columns):
    prev_sel["PREV_CREDIT_UTILIZATION"] = (
        prev_sel["PREV_AMT_CREDIT_MEAN"] / (prev_sel["PREV_AMT_CREDIT_MAX"] + 1e-6)
    )
if set(['PREV_AVG_PAYMENT_SIZE_MEAN','PREV_AMT_CREDIT_MEAN']).issubset(prev_sel.columns):
    prev_sel["PREV_PAYMENT_TO_CREDIT_RATIO"] = (
        prev_sel["PREV_AVG_PAYMENT_SIZE_MEAN"] / (prev_sel["PREV_AMT_CREDIT_MEAN"] + 1e-6)
    )
if set(['PREV_AMT_CREDIT_MEAN','PREV_CNT_PAYMENT_MEAN']).issubset(prev_sel.columns):
    prev_sel["PREV_CREDIT_TO_DURATION_RATIO"] = (
        prev_sel["PREV_AMT_CREDIT_MEAN"] / (prev_sel["PREV_CNT_PAYMENT_MEAN"] + 1e-6)
    )
if "PREV_DAYS_DECISION_MIN" in prev_sel.columns:
    prev_sel["PREV_RECENT_ACTIVITY"] = 1 / (1 + prev_sel["PREV_DAYS_DECISION_MIN"])

print(f" Selected {len(available)} economic base features + derived indicators")

# 2. Correlation Analysis
corr = prev_sel.drop(columns="SK_ID_CURR").corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Economic Features")
plt.show()

# Drop one feature per correlated pair (|corr| > 0.8)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [col for col in upper.columns if any(upper[col].abs() > 0.8)]
if drop_corr:
    print(f" Dropping {len(drop_corr)} highly correlated features: {drop_corr}")
    prev_sel.drop(columns=drop_corr, inplace=True)

# 3. Robust VIF Iterative Elimination
X = prev_sel.drop(columns=["SK_ID_CURR"]).fillna(0.0)
X = X.replace([np.inf, -np.inf], 0)

# Remove near-constant columns
vt = VarianceThreshold(threshold=1e-8)
_ = vt.fit_transform(X)
low_var_cols = list(X.columns[~vt.get_support()])
if low_var_cols:
    X = X.drop(columns=low_var_cols)
    print(f" Dropped near-constant columns: {low_var_cols}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def compute_vif_df(X_scaled, cols):
    vals = []
    for i in range(X_scaled.shape[1]):
        try:
            vals.append(variance_inflation_factor(X_scaled, i))
        except la.LinAlgError:
            vals.append(np.nan)
    return pd.DataFrame({"Feature": cols, "VIF": vals}).sort_values("VIF", ascending=False)

# Iterative elimination
removed = []
threshold = 10
for i in range(30):
    X_scaled = scaler.fit_transform(X)
    vif_df = compute_vif_df(X_scaled, X.columns.tolist())
    if vif_df["VIF"].isna().any():
        bad = vif_df[vif_df["VIF"].isna()]["Feature"].tolist()
        X = X.drop(columns=bad)
        removed += bad
        continue
    worst = vif_df.iloc[0]
    if worst.VIF <= threshold:
        break
    X = X.drop(columns=[worst.Feature])
    removed.append(worst.Feature)

final_vif_df = compute_vif_df(scaler.fit_transform(X), X.columns.tolist())
print("\n Final VIF table:")
print(final_vif_df)
if removed:
    print("\n Removed features due to high VIF:", removed)

# Align prev_sel
prev_sel = prev_sel[['SK_ID_CURR'] + X.columns.tolist()]

# 4. Additional Statistical Diagnostics (Variance, PCA, KMO)
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4.1 Variance / CV
var_info = pd.DataFrame({
    "Feature": X_df.columns,
    "Variance": X_df.var(),
    "Std": X_df.std(),
    "MeanAbs": np.abs(X_df).mean()
})
var_info["CV"] = var_info["Std"] / (var_info["MeanAbs"] + 1e-6)
print("\n Variance & Coefficient of Variation:")
print(var_info.sort_values("Variance", ascending=False))

# 4.2 PCA explained variance
pca = PCA()
pca.fit(X_df)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance (PCA)")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.show()

# 4.3 KMO test
kmo_all, kmo_model = calculate_kmo(X_df)
print(f"\n KMO overall measure = {kmo_model:.3f}")
if kmo_model < 0.6:
    print(" Low KMO (<0.6) — weak factor structure, consider removing noisy features.")
else:
    print(" KMO > 0.6 — structure suitable for factor modeling / PCA.")

# 4.4 Skewness & Kurtosis
dist_stats = pd.DataFrame({
    "Feature": X_df.columns,
    "Skew": [skew(X_df[c]) for c in X_df.columns],
    "Kurtosis": [kurtosis(X_df[c]) for c in X_df.columns]
})
print("\n Skewness & Kurtosis diagnostics:")
print(dist_stats)

# 5. Final Summary
print("\n Final Economic Feature Set (after correlation, VIF, and diagnostics):")
for f in prev_sel.columns:
    if f != "SK_ID_CURR":
        print("•", f)
print(f"\nFinal shape: {prev_sel.shape}")

"""
Goal:
     Clean raw data (missing values, outliers, skewness)
     Create meaningful financial variables
     Aggregate to customer-level features (SK_ID_CURR)
"""

# 1. Load dataset
file_path = "installments_payments.csv"  
ins = pd.read_csv(file_path)
print(f" Loaded installments_payments: {ins.shape[0]:,} rows × {ins.shape[1]} columns")

# 2. Drop columns with too many missing values (>95%)
missing_ratio = ins.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.95].index
if len(drop_cols) > 0:
    print(f" Dropped {len(drop_cols)} columns with >95% missing: {drop_cols.tolist()}")
    ins.drop(columns=drop_cols, inplace=True)

# 3. Handle missing values
num_cols = ins.select_dtypes(include=[np.number]).columns
ins[num_cols] = ins[num_cols].fillna(ins[num_cols].median())

# 4. Outlier clipping
for c in num_cols:
    low, high = ins[c].quantile([0.01, 0.99])
    ins[c] = ins[c].clip(low, high)

# 5. Skewness correction (log1p)
for c in num_cols:
    skewness = ins[c].skew()
    if abs(skewness) > 1.5:
        ins[c] = np.log1p(ins[c] - ins[c].min() + 1)
        print(f"Applied log1p transform to {c} (skew={skewness:.2f})")

# 6. Auto Encoding Function for Categorical Variables

ins = encode_categoricals(ins, max_onehot=10, drop_first=True)

# 7. Derived features (with economic meaning)
ins["PAYMENT_DIFF"] = ins["AMT_PAYMENT"] - ins["AMT_INSTALMENT"]
ins["PAYMENT_RATIO"] = np.where(ins["AMT_INSTALMENT"] != 0,
                                ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"], 1)
ins["PAYMENT_DELAY"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]

# Economic interpretation:
# PAYMENT_DIFF — Over/under payment (negative → underpaid)
# PAYMENT_RATIO — Payment completeness indicator (<1 = missed payments)
# PAYMENT_DELAY — Payment timeliness (positive = delayed)

# 8. Aggregate to customer-level (SK_ID_CURR)
agg_dict = {
    "AMT_PAYMENT": ["mean", "max", "min", "std"],
    "AMT_INSTALMENT": ["mean", "max", "min", "std"],
    "PAYMENT_DIFF": ["mean", "max", "min", "std"],
    "PAYMENT_RATIO": ["mean", "max", "min", "std"],
    "PAYMENT_DELAY": ["mean", "max", "min", "std"]
}

ins_agg = ins.groupby("SK_ID_CURR").agg(agg_dict)
ins_agg.columns = ["INS_" + "_".join(col).upper() for col in ins_agg.columns]
ins_agg.reset_index(inplace=True)

print(f"\n Aggregated to customer level: {ins_agg.shape[0]} customers, {ins_agg.shape[1]} features")

# 9. Preview results
print("\n Preview of aggregated installments_payments features:")
print(ins_agg.head(10))

"""
Goal:
     Select economically meaningful repayment behavior features
     Perform correlation and VIF analysis
     Conduct statistical quality diagnostics (Variance, PCA, Entropy, Stability)
     Output final clean feature set ready for modeling
"""

# 1. ECONOMIC FEATURE SELECTION
# Choose features that capture repayment discipline and financial stability
economic_features = [
    'INS_AMT_PAYMENT_MEAN', 'INS_AMT_PAYMENT_MAX', 'INS_AMT_PAYMENT_MIN', 'INS_AMT_PAYMENT_STD',
    'INS_AMT_INSTALMENT_MEAN', 'INS_AMT_INSTALMENT_MAX', 'INS_AMT_INSTALMENT_MIN', 'INS_AMT_INSTALMENT_STD',
    'INS_PAYMENT_DIFF_MEAN', 'INS_PAYMENT_DIFF_MAX', 'INS_PAYMENT_DIFF_MIN', 'INS_PAYMENT_DIFF_STD',
    'INS_PAYMENT_RATIO_MEAN', 'INS_PAYMENT_RATIO_STD',
    'INS_PAYMENT_DELAY_MEAN', 'INS_PAYMENT_DELAY_STD'
]

available = [f for f in economic_features if f in ins_agg.columns]
ins_sel = ins_agg[['SK_ID_CURR'] + available].copy()
print(f" Selected {len(available)} repayment-related features for diagnostics")

# Economic reasoning:
# - AMT_PAYMENT: real cash outflows (customer repayment capacity)
# - AMT_INSTALMENT: expected obligations (loan burden)
# - PAYMENT_DIFF: actual - expected → measures over/under-payment
# - PAYMENT_RATIO: repayment completeness (<1 = underpayment)
# - PAYMENT_DELAY: lateness of payment (proxy for delinquency behavior)

# 2. CORRELATION ANALYSIS
corr = ins_sel.drop(columns='SK_ID_CURR').corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap - installments_payments features")
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col].abs() > 0.8)]
if to_drop_corr:
    print(f" Dropping {len(to_drop_corr)} highly correlated features: {to_drop_corr}")
    ins_sel.drop(columns=to_drop_corr, inplace=True)

# 3. MULTICOLLINEARITY CHECK (VIF)
X = ins_sel.drop(columns=['SK_ID_CURR']).fillna(0.0)
X = X.replace([np.inf, -np.inf], 0)

# Remove near-constant columns
vt = VarianceThreshold(threshold=1e-8)
_ = vt.fit_transform(X)
low_var_cols = list(X.columns[~vt.get_support()])
if low_var_cols:
    print(f" Dropping near-constant features: {low_var_cols}")
    X.drop(columns=low_var_cols, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def compute_vif(X_scaled, cols):
    vif_vals = []
    for i in range(X_scaled.shape[1]):
        try:
            vif_vals.append(variance_inflation_factor(X_scaled, i))
        except la.LinAlgError:
            vif_vals.append(np.nan)
    return pd.DataFrame({"Feature": cols, "VIF": vif_vals}).sort_values("VIF", ascending=False)

# Iterative elimination (VIF > 10)
removed = []
threshold = 10
for _ in range(20):
    X_scaled = scaler.fit_transform(X)
    vif_df = compute_vif(X_scaled, X.columns)
    if vif_df["VIF"].isna().any():
        bad = vif_df[vif_df["VIF"].isna()]["Feature"].tolist()
        X.drop(columns=bad, inplace=True)
        removed += bad
        continue
    worst = vif_df.iloc[0]
    if worst.VIF <= threshold:
        break
    X.drop(columns=[worst.Feature], inplace=True)
    removed.append(worst.Feature)

print("\n Removed due to high VIF:", removed)
print("\n Final VIF table:")
print(compute_vif(scaler.fit_transform(X), X.columns))

ins_sel = ins_sel[['SK_ID_CURR'] + X.columns.tolist()]

# 4. STATISTICAL QUALITY DIAGNOSTICS
X_std = scaler.fit_transform(X)
X_df = pd.DataFrame(X_std, columns=X.columns)

# Variance, Skewness, Kurtosis
stats_table = pd.DataFrame({
    "Feature": X.columns,
    "Variance": X.var(),
    "Skewness": [skew(X[c]) for c in X.columns],
    "Kurtosis": [kurtosis(X[c]) for c in X.columns]
})
print("\n Variance / Skewness / Kurtosis Summary:")
print(stats_table.sort_values("Variance", ascending=False))

# PCA: Explained variance structure
pca = PCA()
pca.fit(X_std)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance (PCA) - installments_payments")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.grid(True)
plt.show()

print("\n PCA explained variance ratios (first 8):")
for i, ev in enumerate(pca.explained_variance_ratio_[:8]):
    print(f"  PC{i+1}: {ev:.3f}")

# Information Entropy 
entropy_vals = {}
for c in X.columns:
    hist, _ = np.histogram(X[c], bins=20, density=True)
    p = hist[hist > 0]
    entropy_vals[c] = entropy(p)
entropy_df = pd.DataFrame(list(entropy_vals.items()), columns=['Feature', 'Entropy'])
print("\n Information Entropy (Feature Information Richness):")
print(entropy_df.sort_values('Entropy', ascending=False))

# Stability Test (PSI between random halves) 
np.random.seed(42)
idx = np.random.rand(len(X)) < 0.5
X1, X2 = X[idx], X[~idx]

def psi(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)
    return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

psi_values = {c: psi(X1[c], X2[c]) for c in X.columns}
psi_df = pd.DataFrame(list(psi_values.items()), columns=['Feature', 'PSI'])
print("\n Feature Stability (PSI test between random splits):")
print(psi_df.sort_values("PSI"))

# Interpretation:
# PSI < 0.1 → stable
# PSI 0.1–0.25 → mildly shifting
# PSI > 0.25 → unstable (likely time-variant)

# 5. FINAL OUTPUT
print("\n Final Selected Features (installments_payments):")
for f in ins_sel.columns:
    if f != "SK_ID_CURR":
        print("•", f)

print(f"\nFinal shape: {ins_sel.shape}")
print("\n Summary Statistics (first few):")
print(ins_sel.describe(percentiles=[0.05, 0.5, 0.95]).T.head(10))

"""
Goal:
    1. Clean the raw credit_card_balance.csv (row level)
    2. Aggregate to customer-level (SK_ID_CURR) features and create meaningful features
"""

#Import packages 
#suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import gc

# Define one-hot encoding functions for converting categorical data into one-hots 
def encode_categorical(df, include_na=False):
    initial_cols = df.columns.tolist() # Store initial column names
    cat_cols = [col for col in df.columns if df[col].dtype == 'object'] # Identify columns with categorical data type (object) 
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=include_na) # Perform one-hot encoding on categorical columns
    added_cols = [col for col in df.columns if col not in initial_cols] # Get new columns created by encoding
    return df, added_cols

#feature engineering on the credit_card_balance tables: 
def credit_bal_feature_processing(cc):

    print(f" Loaded credit_balance: {cc.shape[0]:,} rows × {cc.shape[1]} columns")

    #Add indication of late payment (Yes if DPD (days past due) > 0 else No)
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: "Y" if x > 0 else "N")

    #Drop columns with too many missing values (>95%)
    missing_ratio = cc.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > 0.95].index
    if len(drop_cols) > 0:
        print(f" Dropped {len(drop_cols)} columns with >95% missing: {drop_cols.tolist()}")
        cc.drop(cc=drop_cols, inplace=True)

    #Handle missing values
    num_cols = cc.select_dtypes(include=[np.number]).columns
    cc[num_cols] = cc[num_cols].fillna(cc[num_cols].median())

    #Outlier clipping
    for c in num_cols:
        low, high = cc[c].quantile([0.01, 0.99])
        cc[c] = cc[c].clip(low, high)
        
    #Skewness correction (log1p)
    for c in num_cols:
        skewness = cc[c].skew()
        if abs(skewness) > 1.5:
            cc[c] = np.log1p(cc[c] - cc[c].min() + 1)
            print(f"Applied log1p transform to {c} (skew={skewness:.2f})")

    #Encode categorical features with one-hot coding
    cc, cat_cols = encode_categorical(cc, include_na = False)

    # Create features with economic reasonings
    #ratio of credit amounts used from credit limit 
    cc['CREDIT_LIMIT_USAGE'] = np.where(cc['AMT_CREDIT_LIMIT_ACTUAL'] !=0, cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'], 0)

    # ratio of current payment to min payment 
    cc['PAYMENT_TO_MIN_PAY'] = np.where(cc['AMT_INST_MIN_REGULARITY'] !=0, cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY'], 1)

    # rato of drawings to the credit limit
    cc['DRAWING_TO_LIMIT'] = np.where(cc['AMT_CREDIT_LIMIT_ACTUAL'] !=0, cc['AMT_DRAWINGS_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL'], 0)  

    #Aggregations by SK_ID_CURR
    #only the four above features are utilized here
    cc_agg_dict = {
    "LATE_PAYMENT_Y": ["mean"],
    "CREDIT_LIMIT_USAGE": ["mean", "max", "min", "std"],
    "PAYMENT_TO_MIN_PAY": ["mean", "max", "min", "std"],
    "DRAWING_TO_LIMIT": ["mean", "max", "min", "std"]
    }

    cc_agg = cc.groupby('SK_ID_CURR').agg(cc_agg_dict)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg.reset_index(inplace= True)

    del cc
    gc.collect()

    print(f"\n Aggregated to customer level: {cc_agg.shape[0]} customers, {cc_agg.shape[1]} features")

    #9. Preview results
    print("\n Preview of aggregated credit_balances features:")
    print(cc_agg.head(10))

    return cc_agg

#Import the credit_card_balance table
credit_bal = pd.read_csv(r"credit_card_balance.csv")

#Apply the process function to the credit_card_balance table
credit_bal_copy=credit_bal.copy()
credit_bal_sel=credit_bal_feature_processing(credit_bal_copy)

"""
Goal:
    1. Clean and join the raw bureau_balance.csv and bureau.csv 
    2. Aggregate to customer-level (SK_ID_CURR) features and create meaningful features
"""
#Import packages 
#suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import gc

#Create a function for doing the feature engineering work of bureau and bureau_balance tables
def bureau_feature_processing(bb, bureau):

    print(f' Loaded bureau_balance: {bb.shape[0]:,} rows × {bb.shape[1]} columns')
    print(f' Loaded bureau: {bureau.shape[0]:,} rows × {bureau.shape[1]} columns')

    #Drop columns with too many missing values (>95%)

    bureau_missing_ratio = bureau.isnull().mean()
    bureau_drop_cols = bureau_missing_ratio[bureau_missing_ratio > 0.95].index
    if len(bureau_drop_cols) > 0:
        print(f' Dropped {len(bureau_drop_cols)} columns with >95% missing: {bureau_drop_cols.tolist()}')
        bureau.drop(bureau=bureau_drop_cols, inplace=True)

    bb_missing_ratio = bb.isnull().mean()
    bb_drop_cols = bb_missing_ratio[bb_missing_ratio > 0.95].index
    if len(bb_drop_cols) > 0:
        print(f' Dropped {len(bb_drop_cols)} columns with >95% missing: {bb_drop_cols.tolist()}')
        bb.drop(bb=bb_drop_cols, inplace=True)


    #Handle missing values
    bureau_num_cols = bureau.select_dtypes(include=[np.number]).columns
    bureau[bureau_num_cols] = bureau[bureau_num_cols].fillna(bureau[bureau_num_cols].median())

    bb_num_cols = bb.select_dtypes(include=[np.number]).columns
    bb[bb_num_cols] = bb[bb_num_cols].fillna(bb[bb_num_cols].median())


    #Outlier clipping
    for c in bureau_num_cols:
        low, high = bureau[c].quantile([0.01, 0.99])
        bureau[c] = bureau[c].clip(low, high)

    for c in bb_num_cols:
        low, high = bb[c].quantile([0.01, 0.99])
        bb[c] = bb[c].clip(low, high)
        
    #Skewness correction (log1p)
    for c in bureau_num_cols:
        skewness = bureau[c].skew()
        if abs(skewness) > 1.5:
            bureau[c] = np.log1p(bureau[c] - bureau[c].min() + 1)
            print(f'Applied log1p transform to {c} (skew={skewness:.2f})')

    for c in bb_num_cols:
        skewness = bb[c].skew()
        if abs(skewness) > 1.5:
            bb[c] = np.log1p(bb[c] - bb[c].min() + 1)
            print(f'Applied log1p transform to {c} (skew={skewness:.2f})')
    
    #Creating one-hots from the categorical variables of both bureau_balance table and bureau table
    bb, bb_cat = encode_categorical(bb, include_na= False)
    bureau, bureau_cat = encode_categorical(bureau, include_na= False)

    #Create aggregate methods based on the loan id SK_ID_BUREAU
    #set-up for finding min, max and count(size) of the month of balance relative to the app date
    bb_agg = {'MONTHS_BALANCE': ['min', 'max', 'size']} 
    for col in bb_cat:
        bb_agg[col] = ['mean'] #set-up for finding the mean on the loan status dummy variables
    bb_agg_cols = bb.groupby('SK_ID_BUREAU').agg(bb_agg) #getting the aggregated columns
    #name the columns with the aggregate type 
    bb_agg_cols.columns = pd.Index([e[0] + '_' + e[1].upper() for e in bb_agg_cols.columns.tolist()]) 
    del bb, bb_agg
    gc.collect()
    
    #Left join the bb_aggregate_cols with bureau on the loan id SK_ID_BUREAU
    bureau = bureau.join(bb_agg_cols, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    #Feature Creation
    #The following 3 measures gauge the utilization of credit limits from the perspective of debt amounts and annuities.
    #Higher utilization rates may imply higher default risks.
    bureau['DEBT_TO_CREDIT'] = np.where(bureau['AMT_CREDIT_SUM'] !=0, bureau['AMT_CREDIT_SUM_DEBT']/bureau['AMT_CREDIT_SUM'], 0) 
    bureau['CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['ANNUITY_TO_CREDIT'] = np.where(bureau['AMT_CREDIT_SUM'] !=0, bureau['AMT_ANNUITY']/bureau['AMT_CREDIT_SUM'],0)


    #Aggregations by SK_ID_CURR
    #Only selected features with clearer economic meanings are covered here
    bureau_agg_dict = {
        # Numerical features 
        'AMT_CREDIT_SUM_OVERDUE': ['mean'], 
        # Derived rato and difference features
        'DEBT_TO_CREDIT': ['mean', 'max', 'min', 'std'],
        'CREDIT_DEBT_DIFF': ['mean', 'max', 'min', 'std'],
        'ANNUITY_TO_CREDIT': ['mean', 'max', 'min', 'std'],  
    }

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_agg_dict)
    bureau_agg.columns = pd.Index(['Bureau_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau_agg.reset_index(inplace= True)
    

    print(f"\n Aggregated to customer level: {bureau_agg.shape[0]} customers, {bureau_agg.shape[1]} features")

    #9. Preview results
    print("\n Preview of aggregated bureau features:")
    print(bureau_agg.head(10))

    return bureau_agg


#Import the bureau_balance and bureau tables
bureau_bal = pd.read_csv(r"bureau_balance.csv")
bureau = pd.read_csv(r"bureau.csv")

#Apply the process function to the tables
bureau_bal_copy=bureau_bal.copy()
bureau_copy = bureau.copy()

bureau_sel=bureau_feature_processing(bureau_bal_copy,bureau_copy)





#feature engineering on main tables: app_train and app_test


app_train = pd.read_csv("application_train.csv")
app_test = pd.read_csv("application_test.csv")

#Import packages 
#suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import gc


#Create a function for doing the feature engineering work
def app_feature_processing(df):

    #Create ratio features driven by domain knowledge on credit related measures
    new_features_df = df[['SK_ID_CURR']].copy()

    #ratio of credit amount of the loan (AMT_CREDIT) to Loan annuity (AMT_ANNUITY) 
    #loans with a higher loan-to-annuity ratio may experience easier repayments due to improved affordability
    new_features_df['CREDIT_TO_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] 

    #ratio of credit amount of the loan (AMT_CREDIT) to the price of the goods for which the loan is given (AMT_GOODS_PRICE) 
    #loans with lower credit amount against the price of the goods may imply better affordability of the client
    new_features_df['CREDIT_TO_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    #ratio of Loan annuity (AMT_ANNUITY) to income of the client
    #lower ratio may imply higer affordability of the client
    new_features_df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    #ratio of credit amount of the loan (AMT_CREDIT) to income of the client
    #lower ratio may imply higer affordability of the client
    new_features_df['CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    #days employed at the current role adjusted with age
    #more days of current employment with respect to age may imply higher job stability and hence credibility of the client
    new_features_df['EMPLOYED_TO_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']


    del df
    gc.collect()

    #Preview results
    print("\n Preview of table features:")
    print(new_features_df.head(10))

    return new_features_df

#Applying the function to both the app_train and app_test table
app_train_ratio_sel=app_feature_processing(app_train)
app_test_ratio_sel=app_feature_processing(app_test)



# Merge Selected Features into Application Train/Test

# 1. Ensure SK_ID_CURR exists in all auxiliary tables
for name, df_aux in [('POS', POS_CASH_balance_sel), ('PREV', prev_sel), ('INS', ins_sel), ('CC', credit_bal_sel), ('BUR', bureau_sel)]:
    if 'SK_ID_CURR' not in df_aux.columns:
        raise KeyError(f"{name}_agg missing SK_ID_CURR!")

# 2. Merge aggregated features
train_merge = app_train.copy()
test_merge  = app_test.copy()

# POS features
train_merge = train_merge.merge(POS_CASH_balance_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(POS_CASH_balance_sel, on='SK_ID_CURR', how='left')

# PREVIOUS_APPLICATION features
train_merge = train_merge.merge(prev_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(prev_sel, on='SK_ID_CURR', how='left')

# NSTALLMENTS_PAYMENTS features
train_merge = train_merge.merge(ins_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(ins_sel, on='SK_ID_CURR', how='left')

# CREDIT_BALANCE features
train_merge = train_merge.merge(credit_bal_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(credit_bal_sel, on='SK_ID_CURR', how='left')

# BUREAU AND BUREAU_BALANCE features
train_merge = train_merge.merge(bureau_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(bureau_sel, on='SK_ID_CURR', how='left')

# Application table ratio features
train_merge = train_merge.merge(app_train_ratio_sel, on='SK_ID_CURR', how='left')
test_merge  = test_merge.merge(app_test_ratio_sel, on='SK_ID_CURR', how='left')


print(f" After merging: train {train_merge.shape}, test {test_merge.shape}")

"""
Goal:
     Clean merged train/test data
"""
# Make copies to avoid overwriting originals
train_clean = train_merge.copy()
test_clean  = test_merge.copy()
#extract the SK_ID_CURR column
ID = test_clean['SK_ID_CURR'].copy().to_numpy()

print(f"Before cleaning: train {train_clean.shape}, test {test_clean.shape}")

#  1.Remove all-null and constant columns
null_cols = [c for c in train_clean.columns if train_clean[c].isna().all()]
const_cols = [c for c in train_clean.columns if train_clean[c].nunique() <= 1]

drop_cols = list(set(null_cols + const_cols))
if drop_cols:
    print(f" Dropping {len(drop_cols)} useless columns (all null or constant).")
    train_clean.drop(columns=drop_cols, inplace=True)
    test_clean.drop(columns=[c for c in drop_cols if c in test_clean.columns], inplace=True)

#  2.Handle missing values
# Identify numeric vs categorical
num_cols = train_clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in train_clean.columns if c not in num_cols and c != 'TARGET']

# Fill numeric NaNs with median
for c in num_cols:
    if c not in train_clean.columns:
        continue
    if c == 'TARGET':
        continue

    med = train_clean[c].median()
    train_clean[c].fillna(med, inplace=True)
    if c in test_clean.columns:
        test_clean[c].fillna(med, inplace=True)

# Fill categorical NaNs with "Unknown"
for c in cat_cols:
    train_clean[c].fillna("Unknown", inplace=True)
    test_clean[c].fillna("Unknown", inplace=True)

print(f" Missing values filled (median for numeric, 'Unknown' for categorical).")

#  3.Clip numeric outliers (1st–99th percentile)
for c in num_cols:
    if c == 'TARGET':  
        continue
    if c not in train_clean.columns:
        continue

    low, high = train_clean[c].quantile([0.01, 0.99])
    train_clean[c] = train_clean[c].clip(low, high)

    if c in test_clean.columns:
        test_clean[c] = test_clean[c].clip(low, high)

print(" Outliers clipped at 1%–99% quantiles.")

#  4.Skewness correction

skewed_cols = []
for c in num_cols:
    if c == 'TARGET':  # skip label
        continue
    skewness = train_clean[c].skew()
    if abs(skewness) > 1.5 and train_clean[c].min() >= 0:  # only nonnegative features
        train_clean[c] = np.log1p(train_clean[c])
        test_clean[c]  = np.log1p(test_clean[c])
        skewed_cols.append(c)

print(f" Applied log1p transform to {len(skewed_cols)} skewed numeric columns.")

#  5.Standardize numeric features

scaler = StandardScaler()

num_cols_aligned = [c for c in num_cols if c in test_clean.columns]

X_train_num = pd.DataFrame(
    scaler.fit_transform(train_clean[num_cols_aligned]),
    columns=num_cols_aligned,
    index=train_clean.index
)
X_test_num = pd.DataFrame(
    scaler.transform(test_clean[num_cols_aligned]),
    columns=num_cols_aligned,
    index=test_clean.index
)

# Recombine scaled numeric + categorical + target
cat_cols = [c for c in train_clean.columns if c not in num_cols + ['TARGET']]
train_clean = pd.concat([train_clean[cat_cols + ['TARGET']], X_train_num], axis=1)
test_clean  = pd.concat([test_clean[cat_cols], X_test_num], axis=1)

print(f" After scaling: train {train_clean.shape}, test {test_clean.shape}")

#  6.Auto Encoding Function for Categorical Variables

train_clean = encode_categoricals(train_clean, max_onehot=10, drop_first=True)
test_clean = encode_categoricals(test_clean, max_onehot=10, drop_first=True)

#  7.Final check
print("Final check:")
print("Train NaNs:", train_clean.isna().sum().sum(), "| Test NaNs:", test_clean.isna().sum().sum())
print("Train numeric columns:", len(num_cols), "| Categorical columns:", len(cat_cols))

"""
Goal:
     Statistical Feature Selection (with target)
"""


# Split into X / y
target_col = 'TARGET'
X = train_clean.drop(columns=[target_col, 'SK_ID_CURR'])
y = train_clean[target_col]
test_X = test_clean.drop(columns=['SK_ID_CURR'])

#  1.Variance Threshold — remove low variance features
vt = VarianceThreshold(threshold=1e-5)
vt.fit(X)
low_var_cols = X.columns[~vt.get_support()].tolist()
if low_var_cols:
    print(f" Dropping {len(low_var_cols)} low-variance features.")
    X.drop(columns=low_var_cols, inplace=True)
    cols_in_test = [c for c in low_var_cols if c in test_clean.columns]
    if cols_in_test:
        test_clean.drop(columns=cols_in_test, inplace=True)
        
#  2.Correlation-based filtering (|corr| > 0.8)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [col for col in upper.columns if any(upper[col] > 0.8)]

if high_corr:
    print(f" Dropping {len(high_corr)} highly correlated features.")
    X.drop(columns=high_corr, inplace=True)
    test_clean.drop(columns=[c for c in high_corr if c in test_clean.columns], inplace=True)


#  3.VIF Filtering — Optimized (using sample subset)
num_cols = X.select_dtypes(include=[np.number]).columns
sample_idx = np.random.choice(X.shape[0], min(5000, X.shape[0]), replace=False)
X_sample = X.iloc[sample_idx][num_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
X_scaled = np.nan_to_num(X_scaled)

vif_df = pd.DataFrame({
    "Feature": num_cols,
    "VIF": [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
}).sort_values("VIF", ascending=False)

high_vif_cols = vif_df.loc[vif_df["VIF"] > 20, "Feature"].tolist()
print(f" Dropping {len(high_vif_cols)} features with VIF > 20 (sample-based).")

X.drop(columns=high_vif_cols, inplace=True, errors='ignore')
test_X.drop(columns=[c for c in high_vif_cols if c in test_X.columns], inplace=True)

#  4.Target-based filtering Statistical & Model-based

# 4.1 ANOVA F-test 

selector = SelectKBest(f_classif, k='all')
selector.fit(X, y)
anova_scores = pd.DataFrame({
    "Feature": X.columns,
    "F_score": selector.scores_,
    "p_value": selector.pvalues_
}).sort_values("F_score", ascending=False)

# 4.2 Mutual Information (Sampled or Proxy)
try:
    sample_idx = np.random.choice(X.shape[0], min(5000, X.shape[0]), replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]
    mi = mutual_info_classif(X_sample, y_sample, random_state=42)
    anova_scores["Mutual_Info"] = mi
    print(f" Computed Mutual Information on sample of {len(sample_idx)} rows.")
except Exception as e:
    print(" MI computation failed, using point-biserial proxy:", e)
    mi_proxy = []
    for col in X.columns:
        try:
            r, _ = pointbiserialr(y, X[col])
            mi_proxy.append(abs(r))
        except Exception:
            mi_proxy.append(0)
    anova_scores["Mutual_Info"] = mi_proxy

# 4.3 Model-based importance 
model = LogisticRegression(max_iter=300, solver='lbfgs')
model.fit(X, y)
importance = np.abs(model.coef_[0])
anova_scores["LR_importance"] = importance

# 4.4 Combine and rank 
anova_scores["RankScore"] = (
    0.5 * anova_scores["F_score"].rank(ascending=False) +
    0.3 * anova_scores["Mutual_Info"].rank(ascending=False) +
    0.2 * anova_scores["LR_importance"].rank(ascending=False)
)
anova_scores.sort_values("RankScore", ascending=True, inplace=True)

# Select top N features (example: top 80)
topN = 80
selected_features = anova_scores.tail(topN)["Feature"].tolist()

print(f"\n Selected Top {topN} statistically significant features.")
print(anova_scores.tail(15)[["Feature", "F_score", "Mutual_Info", "LR_importance"]])

# Subset data
common_features = [f for f in selected_features if f in test_X.columns]
X_final = X[common_features]
test_final = test_X[common_features]

#  5.Final Output
print(f"\nFinal feature set shape: {X_final.shape}")
print("Top few selected features:")
print(X_final.columns[:15].tolist())

print(f"\nFinal test feature set shape: {test_final.shape}")

# For modeling:
# X_final → train features
# y → target
# test_final → test features

"""
Goal
    Model training
    Split the data into training and testing set
"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import random
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y  
)

print("Traing set shape:", X_train.shape)  
print("Testing set shape:", X_test.shape) 
print("Traing Target distribute:", y_train.value_counts()) 

# Evaluate some simple Models: Logistic, Linear, Tree
def evaluate_simple_estimators(X_train, y_train, X_test, y_test):

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
    }
    
    # evaluation using ROC AUC
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  #  model Training
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # prediction
        auc_score = roc_auc_score(y_test, y_pred_proba)  #compute ROC AUC
        results[name] = auc_score  
        print(f"{name}: AUC = {auc_score:.3f}")  
    
    return results


print("Compare the simple models' results:")
simple_results = evaluate_simple_estimators(X_train, y_train, X_test, y_test)  

# Evaluate some ensemble Models: Random Forest, XGBoost, GBoost, CatBoost
def evaluate_ensemble_methods(X_train, y_train, X_test, y_test):

    # Define Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    # evaluation using ROC AUC
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  #  model Training
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # prediction
        auc_score = roc_auc_score(y_test, y_pred_proba)  #compute ROC AUC
        results[name] = auc_score  
        print(f"{name}: AUC = {auc_score:.3f}") 
    
    return results


print("Compare the ensemble models' results:")
ensemble_results = evaluate_ensemble_methods(X_train, y_train, X_test, y_test)  

"""
Goal
    Tune the hyperparameters of selected method: CatBoost
    Analyse the final result
"""
def tune_catboost(X_train, y_train, X_test, y_test):

    # Define parameters space
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'depth': [4, 6, 8, 10, 12],
        'l2_leaf_reg': [1, 3, 5, 7, 10],
        'iterations': [500, 1000, 1500, 2000],
        'border_count': [32, 64, 128, 254],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # create CatBoost classifier
    catboost = CatBoostClassifier(
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'  
    )
    
    # Tune
    random_search = RandomizedSearchCV(
        estimator=catboost,
        param_distributions=param_dist,
        n_iter=20,  
        cv=3,  # fole-3 CV
        scoring='roc_auc',  # evaluate the ROC AUC result
        random_state=42,
        n_jobs=-1  # use all CPU Cores
    )
    
    
    random_search.fit(X_train, y_train)  
    
    
    print("Optimal hyperparameters:", random_search.best_params_)  
    print("Optimal Training Socre:", random_search.best_score_)  
    
    # Use optimal model to train the data
    best_model = random_search.best_estimator_  
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  
    test_auc = roc_auc_score(y_test, y_pred_proba)  
    print("Testing AUC Socre:", test_auc)  
    
    return best_model, random_search.best_params_

# tune_catboost
print("tune_catboost:")
best_catboost, best_params = tune_catboost(X_train, y_train, X_test, y_test)  

# evaluate_final_model
def evaluate_final_model(model, X_train, y_train, X_test, y_test):

    # traing set prediction
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]  
    train_auc = roc_auc_score(y_train, y_train_pred_proba)  
    
    # Testing set prediction
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]  
    test_auc = roc_auc_score(y_test, y_test_pred_proba)  
    
    print(f"Traing set AUC: {train_auc:.3f}")  
    print(f"Testing set AUC: {test_auc:.3f}") 
    
    # plot ROC AUC curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)  
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)  
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.3f})', color='blue') 
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.3f})', color='red')  
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('FP rate')  
    plt.ylabel('TP rate') 
    plt.title('ROC curve')  
    plt.legend()  
    plt.grid(True)  
    plt.show()
    
    # draw confusion matrix
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)  
    cm = confusion_matrix(y_test, y_test_pred)  
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.title('Confusion Matrix')  
    plt.xlabel('Prediction label') 
    plt.ylabel('Real label')  
    plt.show()
    

    print("\n classification report:")
    print(classification_report(y_test, y_test_pred))  
    
    return train_auc, test_auc, y_test_pred_proba


print("evaluate_final_model:")
train_auc, test_auc, y_pred_proba = evaluate_final_model(best_catboost, X_train, y_train, X_test, y_test)  

# 6. feature importance analysis
def plot_feature_importance(model, feature_names, top_n=40):

    # obtaion feature importance result
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_  # obtaion feature importance result
    else:
        print("Model doesn't provide feature importance result")
        return
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)  
    
    top_features = feature_importance_df.head(20)  
    
    # plot top 20 important features
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_features['importance'])  
    plt.yticks(range(len(top_features)), top_features['feature'])  
    plt.xlabel('feature importance score') 
    plt.title('Top 20 important features')  
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()
    

    
    return feature_importance_df

print("feature importance analysis:")
feature_importance_df = plot_feature_importance(best_catboost, X_train.columns)  

"""
Goal
    After validating the model, it is evaluated using the test dataset
"""

# use test dataset to predict default risk.
test_pred_proba = best_catboost.predict_proba(test_final)[:, 1]

submission = pd.DataFrame({"SK_ID_CURR":ID, "TARGET":test_pred_proba})


# save the prediction results as csv
submission.to_csv("submission-catboost.csv", index = False)
