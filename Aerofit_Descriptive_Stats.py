# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🏋️ Aerofit Treadmill — Descriptive Statistics & Probability Analysis
#
# **Author:** Pritam Palit
#
# **Objective:** Identify target audience characteristics for each Aerofit treadmill product
# through descriptive statistics, visual analysis, and probability computations to derive
# actionable business recommendations.
#
# ---
#
# ## Product Portfolio
#
# | Product | Price | Segment |
# |---------|-------|---------|
# | **KP281** | \$1,500 | Entry-level |
# | **KP481** | \$1,750 | Mid-level |
# | **KP781** | \$2,500 | Advanced |
#
# ---

# %% [markdown]
# ## 1. Problem Statement & Basic Metrics (10 pts)
#
# **Business Problem:** Aerofit, a leading brand in fitness equipment, wants to study
# the characteristics of its treadmill customers to provide better product recommendations
# for new buyers. The market research team wants to investigate whether there are
# differences across the product lines (KP281, KP481, KP781) with respect to customer
# demographics and behavioral attributes.
#
# **Goal:** Build customer profiles for each treadmill product by analyzing descriptive
# statistics and computing conditional & marginal probabilities.

# %%
# ── Imports ──
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Styling ──
sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ── Paths ──
IMG_DIR = 'images'
DATA_PATH = os.path.join('data', 'aerofit_treadmill.csv')
os.makedirs(IMG_DIR, exist_ok=True)

print("✅ Setup complete")

# %% [markdown]
# ### 1.1 Load & Inspect the Dataset

# %%
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
df.head(10)

# %%
print("── Column Data Types ──")
print(df.dtypes)
print(f"\n── Dataset Info ──")
df.info()

# %%
print("── Missing Values ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df)
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# %%
print("── Duplicate Rows ──")
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows: {dup_count}")
if dup_count > 0:
    print("Dropping duplicates...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after dropping duplicates: {df.shape}")

# %%
print("── Statistical Summary (Numerical) ──")
df.describe().T.round(2)

# %%
print("── Statistical Summary (Categorical) ──")
df.describe(include='object').T

# %% [markdown]
# ### 1.2 Convert Categoricals to `category` dtype

# %%
cat_cols = ['Product', 'Gender', 'MaritalStatus']
for col in cat_cols:
    df[col] = df[col].astype('category')

print("Dtypes after conversion:")
print(df[cat_cols].dtypes)

# %% [markdown]
# ---
# ## 2. Non-Graphical Analysis (10 pts)

# %% [markdown]
# ### 2.1 Value Counts for Categorical Variables

# %%
for col in cat_cols:
    print(f"\n── {col} ──")
    vc = df[col].value_counts()
    vc_pct = df[col].value_counts(normalize=True).mul(100).round(2)
    summary = pd.DataFrame({'Count': vc, 'Percentage (%)': vc_pct})
    print(summary)

# %% [markdown]
# ### 2.2 Unique Attributes

# %%
print("── Unique Values per Column ──")
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique <= 15:
        print(f"  {col:20s} → {n_unique:3d} unique → {sorted(df[col].unique())}")
    else:
        print(f"  {col:20s} → {n_unique:3d} unique → range [{df[col].min()}, {df[col].max()}]")

# %% [markdown]
# ### 2.3 Grouped Statistics by Product

# %%
print("── Mean statistics by Product ──")
df.groupby('Product').mean(numeric_only=True).round(2)

# %%
print("── Median statistics by Product ──")
df.groupby('Product').median(numeric_only=True).round(2)

# %%
print("── Standard Deviation by Product ──")
df.groupby('Product').std(numeric_only=True).round(2)

# %%
print("── Detailed describe by Product ──")
for product in ['KP281', 'KP481', 'KP781']:
    print(f"\n{'='*60}")
    print(f"  Product: {product}")
    print(f"{'='*60}")
    subset = df[df['Product'] == product]
    print(f"  Count: {len(subset)} customers ({len(subset)/len(df)*100:.1f}%)")
    print(subset.describe().T.round(2))

# %% [markdown]
# ---
# ## 3. Visual Analysis (30 pts)

# %% [markdown]
# ### 3.1 Univariate Analysis — Continuous Variables (10 pts)

# %%
# ── Histograms + KDE for continuous variables ──
continuous_cols = ['Age', 'Income', 'Usage', 'Fitness', 'Miles']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    sns.histplot(df[col], kde=True, bins=20, color=sns.color_palette('Set2')[i],
                 edgecolor='white', alpha=0.7, ax=ax)
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.1f}')
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.legend(fontsize=9)

# Remove unused subplot
axes[5].set_visible(False)
fig.suptitle('Univariate Distribution — Continuous Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_univariate_continuous.png'))
plt.show()

# %%
# ── Countplots for categorical variables ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(cat_cols):
    ax = axes[i]
    order = df[col].value_counts().index
    palette = sns.color_palette('Set2', n_colors=len(order))
    sns.countplot(data=df, x=col, order=order, palette=palette, edgecolor='black',
                  linewidth=0.8, ax=ax)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}\n({height/total*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.set_ylabel('Count')

fig.suptitle('Univariate Distribution — Categorical Variables', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_univariate_categorical.png'))
plt.show()

# %%
# ── Countplot: Product split by Gender ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='Product', hue='Gender', palette='Set2',
              edgecolor='black', linewidth=0.8, ax=axes[0])
axes[0].set_title('Product Distribution by Gender', fontweight='bold')
axes[0].legend(title='Gender')
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

sns.countplot(data=df, x='Product', hue='MaritalStatus', palette='Set1',
              edgecolor='black', linewidth=0.8, ax=axes[1])
axes[1].set_title('Product Distribution by Marital Status', fontweight='bold')
axes[1].legend(title='Marital Status')
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '03_product_by_gender_marital.png'))
plt.show()

# %% [markdown]
# ### 3.2 Bivariate Analysis — Boxplots by Product (10 pts)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

box_vars = ['Age', 'Income', 'Miles', 'Usage', 'Fitness']
palette = {'KP281': '#66c2a5', 'KP481': '#fc8d62', 'KP781': '#8da0cb'}

for i, col in enumerate(box_vars):
    ax = axes[i]
    sns.boxplot(data=df, x='Product', y=col, palette=palette,
                order=['KP281', 'KP481', 'KP781'],
                linewidth=1.2, ax=ax)
    # overlay strip plot
    sns.stripplot(data=df, x='Product', y=col,
                  order=['KP281', 'KP481', 'KP781'],
                  color='black', alpha=0.3, size=3, jitter=True, ax=ax)
    ax.set_title(f'{col} by Product', fontweight='bold')

axes[5].set_visible(False)
fig.suptitle('Boxplots — Continuous Variables by Treadmill Product',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '04_boxplots_by_product.png'))
plt.show()

# %%
# ── Violin plots as complement ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
violin_vars = ['Income', 'Miles', 'Fitness']

for i, col in enumerate(violin_vars):
    ax = axes[i]
    sns.violinplot(data=df, x='Product', y=col, palette=palette,
                   order=['KP281', 'KP481', 'KP781'],
                   inner='quartile', linewidth=1.2, ax=ax)
    ax.set_title(f'{col} by Product (Violin)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '05_violinplots_by_product.png'))
plt.show()

# %% [markdown]
# ### 3.3 Correlation Analysis (10 pts)

# %%
# ── Correlation Heatmap ──
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
corr_matrix = df[numeric_cols].corr().round(3)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            center=0, vmin=-1, vmax=1, linewidths=0.8,
            square=True, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Heatmap — Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '06_correlation_heatmap.png'))
plt.show()

print("\n── Top Correlations ──")
corr_unstacked = corr_matrix.where(~np.triu(np.ones_like(corr_matrix, dtype=bool))).unstack()
corr_sorted = corr_unstacked.dropna().sort_values(ascending=False)
print(corr_sorted.head(10).to_string())

# %%
# ── Pairplot colored by Product ──
pair_cols = ['Age', 'Income', 'Usage', 'Fitness', 'Miles', 'Product']
g = sns.pairplot(df[pair_cols], hue='Product', palette=palette,
                 diag_kind='kde', plot_kws={'alpha': 0.5, 's': 30},
                 height=2.2, aspect=1.1)
g.figure.suptitle('Pairplot — Colored by Treadmill Product', fontsize=16,
                   fontweight='bold', y=1.02)
plt.savefig(os.path.join(IMG_DIR, '07_pairplot_by_product.png'))
plt.show()

# %% [markdown]
# ---
# ## 4. Missing Value & Outlier Detection (10 pts)

# %% [markdown]
# ### 4.1 Missing Value Check (Recap)

# %%
print("── Missing Values Summary ──")
print(df.isnull().sum())
print(f"\n✅ Total missing: {df.isnull().sum().sum()} — Dataset is clean!")

# %% [markdown]
# ### 4.2 Outlier Detection using IQR Method

# %%
def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return {
        'Column': column,
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outlier Count': len(outliers),
        'Outlier %': round(len(outliers) / len(data) * 100, 2)
    }


outlier_results = []
for col in continuous_cols:
    outlier_results.append(detect_outliers_iqr(df, col))

outlier_df = pd.DataFrame(outlier_results)
outlier_df

# %%
# ── Boxplots for outlier visualization ──
fig, axes = plt.subplots(1, len(continuous_cols), figsize=(20, 5))

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    sns.boxplot(data=df, y=col, color=sns.color_palette('Set2')[i],
                linewidth=1.2, ax=ax)
    ax.set_title(f'{col}', fontweight='bold')

    # Mark outlier bounds
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    ax.axhline(Q1 - 1.5 * IQR, color='red', linestyle='--', alpha=0.7, label='Lower Bound')
    ax.axhline(Q3 + 1.5 * IQR, color='red', linestyle='--', alpha=0.7, label='Upper Bound')
    ax.legend(fontsize=7)

fig.suptitle('Outlier Detection — Boxplots with IQR Bounds',
             fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_outlier_boxplots.png'))
plt.show()

# %%
# ── Skewness & Kurtosis ──
print("── Skewness & Kurtosis ──")
sk_data = []
for col in continuous_cols:
    sk_data.append({
        'Feature': col,
        'Skewness': round(df[col].skew(), 3),
        'Kurtosis': round(df[col].kurtosis(), 3),
        'Interpretation': 'Right-skewed' if df[col].skew() > 0.5 else
                          'Left-skewed' if df[col].skew() < -0.5 else 'Approx. symmetric'
    })
pd.DataFrame(sk_data)

# %% [markdown]
# ---
# ## 5. Probability Analysis (20 pts)

# %% [markdown]
# ### 5.1 Marginal Probabilities

# %%
print("=" * 60)
print("  MARGINAL PROBABILITIES — Product Purchase")
print("=" * 60)
product_counts = df['Product'].value_counts()
product_probs = df['Product'].value_counts(normalize=True).round(4)
marginal_df = pd.DataFrame({
    'Count': product_counts,
    'P(Product)': product_probs,
    'Percentage (%)': (product_probs * 100).round(2)
})
print(marginal_df)
print()

# Gender marginal
print("── Marginal Probability: Gender ──")
gender_probs = df['Gender'].value_counts(normalize=True).round(4)
print(gender_probs)
print()

# Marital status marginal
print("── Marginal Probability: Marital Status ──")
marital_probs = df['MaritalStatus'].value_counts(normalize=True).round(4)
print(marital_probs)

# %% [markdown]
# ### 5.2 Conditional Probability Tables (Crosstabs)

# %%
# ── Product × Gender ──
print("=" * 60)
print("  CONTINGENCY TABLE: Product × Gender")
print("=" * 60)

ct_pg = pd.crosstab(df['Product'], df['Gender'], margins=True)
print("\n── Raw Counts ──")
print(ct_pg)

ct_pg_row = pd.crosstab(df['Product'], df['Gender'], normalize='index').round(4)
print("\n── Row-normalized (P(Gender | Product)) ──")
print(ct_pg_row)

ct_pg_col = pd.crosstab(df['Product'], df['Gender'], normalize='columns').round(4)
print("\n── Column-normalized (P(Product | Gender)) ──")
print(ct_pg_col)

ct_pg_all = pd.crosstab(df['Product'], df['Gender'], normalize='all').round(4)
print("\n── Overall-normalized (Joint Probability) ──")
print(ct_pg_all)

# %%
# ── Product × Marital Status ──
print("=" * 60)
print("  CONTINGENCY TABLE: Product × Marital Status")
print("=" * 60)

ct_pm = pd.crosstab(df['Product'], df['MaritalStatus'], margins=True)
print("\n── Raw Counts ──")
print(ct_pm)

ct_pm_row = pd.crosstab(df['Product'], df['MaritalStatus'], normalize='index').round(4)
print("\n── Row-normalized (P(MaritalStatus | Product)) ──")
print(ct_pm_row)

ct_pm_col = pd.crosstab(df['Product'], df['MaritalStatus'], normalize='columns').round(4)
print("\n── Column-normalized (P(Product | MaritalStatus)) ──")
print(ct_pm_col)

# %%
# ── Product × Fitness ──
print("=" * 60)
print("  CONTINGENCY TABLE: Product × Fitness Level")
print("=" * 60)

ct_pf = pd.crosstab(df['Product'], df['Fitness'], margins=True)
print("\n── Raw Counts ──")
print(ct_pf)

ct_pf_row = pd.crosstab(df['Product'], df['Fitness'], normalize='index').round(4)
print("\n── Row-normalized (P(Fitness | Product)) ──")
print(ct_pf_row)

# %% [markdown]
# ### 5.3 Key Conditional Probabilities

# %%
print("=" * 60)
print("  KEY CONDITIONAL PROBABILITIES")
print("=" * 60)

total = len(df)

# P(Male | KP781)
kp781 = df[df['Product'] == 'KP781']
p_male_given_kp781 = (kp781['Gender'] == 'Male').mean()
print(f"\n  P(Male | KP781)      = {p_male_given_kp781:.4f}  ({p_male_given_kp781*100:.2f}%)")

# P(Female | KP781)
p_female_given_kp781 = (kp781['Gender'] == 'Female').mean()
print(f"  P(Female | KP781)    = {p_female_given_kp781:.4f}  ({p_female_given_kp781*100:.2f}%)")

# P(KP781 | Male)
males = df[df['Gender'] == 'Male']
p_kp781_given_male = (males['Product'] == 'KP781').mean()
print(f"\n  P(KP781 | Male)      = {p_kp781_given_male:.4f}  ({p_kp781_given_male*100:.2f}%)")

# P(KP781 | Female)
females = df[df['Gender'] == 'Female']
p_kp781_given_female = (females['Product'] == 'KP781').mean()
print(f"  P(KP781 | Female)    = {p_kp781_given_female:.4f}  ({p_kp781_given_female*100:.2f}%)")

# P(KP281 | Female)
p_kp281_given_female = (females['Product'] == 'KP281').mean()
print(f"\n  P(KP281 | Female)    = {p_kp281_given_female:.4f}  ({p_kp281_given_female*100:.2f}%)")

# P(KP281 | Male)
p_kp281_given_male = (males['Product'] == 'KP281').mean()
print(f"  P(KP281 | Male)      = {p_kp281_given_male:.4f}  ({p_kp281_given_male*100:.2f}%)")

# P(Partnered | KP781)
p_partnered_kp781 = (kp781['MaritalStatus'] == 'Partnered').mean()
print(f"\n  P(Partnered | KP781) = {p_partnered_kp781:.4f}  ({p_partnered_kp781*100:.2f}%)")

# P(Single | KP781)
p_single_kp781 = (kp781['MaritalStatus'] == 'Single').mean()
print(f"  P(Single | KP781)    = {p_single_kp781:.4f}  ({p_single_kp781*100:.2f}%)")

# %%
# ── Comprehensive Conditional Probability Summary Table ──
print("\n── Complete Conditional Probability Matrix: P(Product | Attribute) ──\n")

products = ['KP281', 'KP481', 'KP781']
summary_rows = []

for gender in ['Male', 'Female']:
    row = {'Attribute': f'Gender={gender}'}
    subset = df[df['Gender'] == gender]
    for prod in products:
        prob = (subset['Product'] == prod).mean()
        row[f'P({prod})'] = f'{prob:.4f}'
    summary_rows.append(row)

for ms in ['Single', 'Partnered']:
    row = {'Attribute': f'MaritalStatus={ms}'}
    subset = df[df['MaritalStatus'] == ms]
    for prod in products:
        prob = (subset['Product'] == prod).mean()
        row[f'P({prod})'] = f'{prob:.4f}'
    summary_rows.append(row)

for fit in sorted(df['Fitness'].unique()):
    row = {'Attribute': f'Fitness={fit}'}
    subset = df[df['Fitness'] == fit]
    for prod in products:
        prob = (subset['Product'] == prod).mean()
        row[f'P({prod})'] = f'{prob:.4f}'
    summary_rows.append(row)

cond_prob_summary = pd.DataFrame(summary_rows).set_index('Attribute')
print(cond_prob_summary.to_string())

# %% [markdown]
# ### 5.4 Visual: Conditional Probability Heatmaps

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Product × Gender heatmap
sns.heatmap(ct_pg_row, annot=True, fmt='.2%', cmap='YlOrRd',
            linewidths=0.8, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title('P(Gender | Product)', fontweight='bold')

# Product × Marital Status heatmap
sns.heatmap(ct_pm_row, annot=True, fmt='.2%', cmap='YlGnBu',
            linewidths=0.8, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('P(MaritalStatus | Product)', fontweight='bold')

# Product × Fitness heatmap
sns.heatmap(ct_pf_row, annot=True, fmt='.2%', cmap='PuBuGn',
            linewidths=0.8, ax=axes[2], vmin=0, vmax=1)
axes[2].set_title('P(Fitness | Product)', fontweight='bold')

fig.suptitle('Conditional Probability Heatmaps', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '09_conditional_prob_heatmaps.png'))
plt.show()

# %% [markdown]
# ### 5.5 Customer Profiling per Product

# %%
print("=" * 70)
print("  CUSTOMER PROFILES BY TREADMILL PRODUCT")
print("=" * 70)

for product in products:
    subset = df[df['Product'] == product]
    n = len(subset)
    pct = n / len(df) * 100

    print(f"\n{'─'*70}")
    print(f"  📦 {product} (n={n}, {pct:.1f}% of customers)")
    print(f"{'─'*70}")

    # Demographics
    print(f"  Age       : {subset['Age'].mean():.1f} ± {subset['Age'].std():.1f}  "
          f"(range: {subset['Age'].min()}-{subset['Age'].max()})")
    print(f"  Income    : ${subset['Income'].mean():,.0f} ± ${subset['Income'].std():,.0f}  "
          f"(range: ${subset['Income'].min():,}-${subset['Income'].max():,})")
    print(f"  Education : {subset['Education'].mean():.1f} years")

    # Gender split
    gender_split = subset['Gender'].value_counts(normalize=True).mul(100).round(1)
    gender_str = ' / '.join([f"{g}: {v}%" for g, v in gender_split.items()])
    print(f"  Gender    : {gender_str}")

    # Marital status
    ms_split = subset['MaritalStatus'].value_counts(normalize=True).mul(100).round(1)
    ms_str = ' / '.join([f"{m}: {v}%" for m, v in ms_split.items()])
    print(f"  Marital   : {ms_str}")

    # Behavioral
    print(f"  Usage     : {subset['Usage'].mean():.1f} times/week")
    print(f"  Fitness   : {subset['Fitness'].mean():.1f} / 5")
    print(f"  Miles     : {subset['Miles'].mean():.0f} miles/week")

# %% [markdown]
# ---
# ## 6. Business Insights (10 pts)

# %% [markdown]
# ### 6.1 Product-wise Customer Profiles — Summary Table

# %%
profile_data = []
for product in products:
    s = df[df['Product'] == product]
    profile_data.append({
        'Product': product,
        'Count': len(s),
        'Share (%)': round(len(s)/len(df)*100, 1),
        'Avg Age': round(s['Age'].mean(), 1),
        'Avg Income ($)': round(s['Income'].mean(), 0),
        'Avg Education (yrs)': round(s['Education'].mean(), 1),
        'Male %': round((s['Gender']=='Male').mean()*100, 1),
        'Partnered %': round((s['MaritalStatus']=='Partnered').mean()*100, 1),
        'Avg Usage/wk': round(s['Usage'].mean(), 1),
        'Avg Fitness': round(s['Fitness'].mean(), 1),
        'Avg Miles/wk': round(s['Miles'].mean(), 0),
    })

profile_df = pd.DataFrame(profile_data).set_index('Product')
profile_df

# %%
# ── Visual comparison of key metrics across products ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

metrics = ['Age', 'Income', 'Usage', 'Fitness', 'Miles', 'Education']
for i, col in enumerate(metrics):
    ax = axes[i]
    means = df.groupby('Product')[col].mean().reindex(['KP281', 'KP481', 'KP781'])
    bars = ax.bar(means.index, means.values,
                  color=[palette[p] for p in means.index],
                  edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(f'Average {col} by Product', fontweight='bold')
    ax.set_ylabel(col)

fig.suptitle('Key Metrics Comparison Across Products', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '10_metrics_comparison.png'))
plt.show()

# %% [markdown]
# ### 6.2 Key Differentiators

# %%
# ── Income distribution by product (detailed) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Income histogram by product
for product in products:
    subset = df[df['Product'] == product]
    axes[0].hist(subset['Income'], bins=15, alpha=0.5, label=product,
                 color=palette[product], edgecolor='black', linewidth=0.5)
axes[0].set_title('Income Distribution by Product', fontweight='bold')
axes[0].set_xlabel('Annual Income ($)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Miles histogram by product
for product in products:
    subset = df[df['Product'] == product]
    axes[1].hist(subset['Miles'], bins=15, alpha=0.5, label=product,
                 color=palette[product], edgecolor='black', linewidth=0.5)
axes[1].set_title('Miles/Week Distribution by Product', fontweight='bold')
axes[1].set_xlabel('Miles per Week')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '11_income_miles_by_product.png'))
plt.show()

# %%
# ── Fitness & Usage patterns ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fitness by Gender & Product
ct_fit = pd.crosstab([df['Product'], df['Gender']], df['Fitness'])
ct_fit_pct = ct_fit.div(ct_fit.sum(axis=1), axis=0)
ct_fit_pct.plot(kind='bar', stacked=True, colormap='YlOrRd', ax=axes[0],
                edgecolor='black', linewidth=0.5)
axes[0].set_title('Fitness Level Distribution by Product & Gender', fontweight='bold')
axes[0].set_ylabel('Proportion')
axes[0].legend(title='Fitness', bbox_to_anchor=(1.05, 1))
axes[0].tick_params(axis='x', rotation=45)

# Usage by Product
sns.boxplot(data=df, x='Product', y='Usage', palette=palette,
            order=['KP281', 'KP481', 'KP781'], ax=axes[1])
axes[1].set_title('Usage (times/week) by Product', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '12_fitness_usage_patterns.png'))
plt.show()

# %% [markdown]
# ### 6.3 Statistical Tests — Are Product Segments Truly Different?

# %%
print("── ANOVA Test: Do product groups differ significantly? ──\n")

anova_results = []
for col in continuous_cols:
    groups = [df[df['Product'] == p][col].values for p in products]
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results.append({
        'Feature': col,
        'F-statistic': round(f_stat, 3),
        'p-value': f'{p_val:.6f}',
        'Significant (α=0.05)': '✅ Yes' if p_val < 0.05 else '❌ No'
    })

anova_df = pd.DataFrame(anova_results)
print(anova_df.to_string(index=False))

# %% [markdown]
# ### 6.4 Consolidated Business Insights

# %% [markdown]
# #### 📊 Key Findings
#
# **1. KP281 (Entry-Level — \$1,500)**
# - Most popular product overall
# - Attracts budget-conscious customers with lower income levels
# - Users have moderate fitness levels and usage patterns
# - Fairly balanced gender distribution
#
# **2. KP481 (Mid-Level — \$1,750)**
# - Second most popular product
# - Customers are similar to KP281 in demographics
# - Slightly higher income and usage compared to KP281
# - Acts as a "step-up" product for slightly more committed users
#
# **3. KP781 (Advanced — \$2,500)**
# - Least purchased but serves a distinct segment
# - Customers have significantly higher income levels
# - Users are more fitness-oriented (higher fitness ratings)
# - They run more miles per week — serious fitness enthusiasts
# - Skews more heavily male
#
# **4. Gender Patterns**
# - Males dominate KP781 purchases — strong gender bias
# - KP281 and KP481 have more balanced gender distributions
# - Marketing for KP781 to women represents a growth opportunity
#
# **5. Income & Fitness are Key Differentiators**
# - Income is the strongest predictor of product choice
# - Fitness level and weekly miles strongly correlate with premium product selection
# - Education years show moderate correlation with product tier

# %% [markdown]
# ---
# ## 7. Recommendations (10 pts)

# %% [markdown]
# ### 🎯 Actionable Recommendations for Aerofit
#
# ---
#
# #### **A. Target Demographics by Product**
#
# | Product | Target Segment | Key Message |
# |---------|---------------|-------------|
# | **KP281** | Budget-conscious beginners, casual joggers, wide age range | "Start your fitness journey without breaking the bank" |
# | **KP481** | Regular exercisers looking to upgrade, mid-income professionals | "Level up your workout with advanced features" |
# | **KP781** | High-income fitness enthusiasts, marathon runners, serious athletes | "Professional-grade performance for serious athletes" |
#
# ---
#
# #### **B. Marketing Strategies**
#
# 1. **KP281 — Mass Market Approach**
#    - Advertise on social media platforms targeting 25-45 age group
#    - Emphasize affordability and ease of use
#    - Partner with beginner fitness programs and wellness apps
#    - Run seasonal discount campaigns to drive volume
#
# 2. **KP481 — Upgrade & Retain**
#    - Target existing KP281 owners with upgrade offers after 6-12 months
#    - Highlight advanced features over KP281 at only \$250 more
#    - Position as "best value" treadmill in marketing
#    - Use testimonials from users who transitioned from KP281
#
# 3. **KP781 — Premium Positioning**
#    - Partner with gym chains and professional trainers
#    - Sponsor marathons and fitness competitions
#    - **Target female athletes specifically** — currently underrepresented
#    - Offer premium financing options (0% EMI)
#    - Emphasize durability, tracking features, and professional specs
#
# ---
#
# #### **C. Product Development Insights**
#
# 1. **Gender Gap in KP781**: Develop female-specific marketing campaigns; consider design
#    adjustments (size options, color choices) to appeal to women
#
# 2. **Income-based Bundling**: Offer KP281 + fitness accessories bundle to compete with
#    KP481 price point; offer KP781 + premium subscription bundle
#
# 3. **Fitness Level Bridge**: Users with fitness level 3-4 are swing customers between
#    KP481 and KP781 — targeted promotions could drive upgrades
#
# ---
#
# #### **D. Data-Driven Next Steps**
#
# 1. Collect customer satisfaction scores post-purchase for retention analysis
# 2. Track upgrade paths (KP281 → KP481 → KP781) over customer lifetime
# 3. A/B test marketing messages based on the conditional probability insights
# 4. Build a predictive classification model to recommend products to new customers

# %% [markdown]
# ---
#
# ## 📋 Summary
#
# | Section | Key Takeaway |
# |---------|-------------|
# | **Basic Metrics** | 180 customers, 9 features, no missing values |
# | **Non-Graphical** | KP281 is most popular; males slightly outnumber females |
# | **Visual Analysis** | Income & Miles clearly differentiate KP781 buyers |
# | **Outliers** | Income has some high-end outliers; Miles shows right skew |
# | **Probability** | P(Male \| KP781) is high; KP781 buyers are high-income fitness enthusiasts |
# | **Insights** | Three distinct customer segments aligned with product tiers |
# | **Recommendations** | Product-specific marketing; address KP781 gender gap; upgrade paths |
#
# ---
#
# **Author:** Pritam Palit
#
# **Case Study:** Aerofit Treadmill — Descriptive Statistics & Probability Analysis
