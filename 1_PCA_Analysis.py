import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Load cleaned socio-economic datasets for each London borough
work_df = pd.read_csv("work_data_c.csv")
age_df = pd.read_csv("age_18_64_data_c.csv")
disability_df = pd.read_csv("disability_data_c.csv")
economic_df = pd.read_csv("economic_activity_data_c.csv")
income_df = pd.read_csv("income_data_c.csv")

# Standardise column names to ensure consistency across datasets
work_df.rename(columns={
    'Works mainly from home': 'Work_From_Home',
    'Total: All usual residents aged 16 years and over in employment': 'Employed'
}, inplace=True)
age_df.rename(columns={'Age 18 to 64 Total': 'Age_18_64'}, inplace=True)
disability_df.rename(columns={'Disabled People': 'Disabled', 'Total population': 'Total_Pop'}, inplace=True)
economic_df.rename(columns={'Total Economically Active Population': 'Economically_Active'}, inplace=True)
income_df.rename(columns={'GDHI per Head (£)': 'Income_per_Head'}, inplace=True)

# Merge all datasets on the common 'Borough' field
dfs = [work_df, age_df[['Borough', 'Age_18_64']],
       disability_df[['Borough', 'Disabled', 'Total_Pop']],
       economic_df[['Borough', 'Economically_Active']],
       income_df[['Borough', 'Income_per_Head']]]

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='Borough', how='inner')

# Create percentage indicators
merged_df['Age_18_64_%'] = (merged_df['Age_18_64'] / merged_df['Total_Pop']) * 100
merged_df['Economically_Active_%'] = (merged_df['Economically_Active'] / merged_df['Total_Pop']) * 100
merged_df['Disabled_%'] = (merged_df['Disabled'] / merged_df['Total_Pop']) * 100
merged_df['Work_From_Home_%'] = (merged_df['Work_From_Home'] / merged_df['Employed']) * 100

# Select features for PCA
features = ['Age_18_64_%', 'Economically_Active_%', 'Income_per_Head', 'Work_From_Home_%', 'Disabled_%']

# KMO and Bartlett’s test
kmo_all, kmo_model = calculate_kmo(merged_df[features])
chi_sq, p_value = calculate_bartlett_sphericity(merged_df[features])
print(f"KMO Score: {kmo_model:.2f}")
print(f"Bartlett’s Test p-value: {p_value:.4f}")

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(merged_df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("1_pca_feature_correlation_heatmap.png")

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(merged_df[features])

# Boxplot for Z-scores
plt.figure(figsize=(10, 5))
boxplot_data = pd.DataFrame(X_scaled, columns=features)
sns.boxplot(data=boxplot_data)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.text(len(features) - 0.5, 0.1, 'Mean = 0', color='black', fontsize=9, ha='right')
plt.title("Boxplot of Standardised Socio-Economic Features by London Borough")
plt.xlabel("Features")
plt.ylabel("Standardised Values (Z-scores)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_pca_standardised_feature_boxplot.png")

# PCA analysis
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
merged_df['PC1'] = X_pca[:, 0]
merged_df['PC2'] = X_pca[:, 1]

explained_var = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_var)
print(f"Total Variance Captured: {explained_var.sum() * 100:.2f}%")

# Scree plot
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_var = pca_full.explained_variance_ratio_
components = range(1, 6)
cumulative_variance = np.cumsum(explained_var[:5]) * 100
plt.figure(figsize=(6, 4))
plt.plot(components, cumulative_variance, marker='o', linestyle='--')
plt.xticks(components)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("Scree Plot: Cumulative Variance by PCA Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_pca_scree_plot.png")

# Loadings table
loadings = pd.DataFrame(pca.components_.T, columns=['PC1_Loading', 'PC2_Loading'], index=features)

# PC1 contribution bar chart
loadings['PC1_Abs_Loading'] = loadings['PC1_Loading'].abs()
loadings['Direction'] = loadings['PC1_Loading'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
loadings_sorted_pc1 = loadings.sort_values(by='PC1_Abs_Loading', ascending=False)
top_features_pc1 = loadings_sorted_pc1.index[:2].tolist()
pc1_label = f"{top_features_pc1[0].replace('_', ' ')} and {top_features_pc1[1].replace('_', ' ')} Index"
plt.figure(figsize=(10, 6))
colors_pc1 = ['green' if d == 'Positive' else 'red' for d in loadings_sorted_pc1['Direction']]
plt.bar(loadings_sorted_pc1.index, loadings_sorted_pc1['PC1_Abs_Loading'], color=colors_pc1)
plt.title(f"Feature Contribution to PC1 ({pc1_label})")
plt.ylabel("Absolute PC1 Loading")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_pca_pc1_feature_contributions.png")

print("Top PC1 Feature Contributions:")
print(loadings_sorted_pc1[['PC1_Loading', 'PC1_Abs_Loading', 'Direction']])
print(f"PC1 is interpreted as: {pc1_label}")

# PC2 contribution bar chart
loadings['PC2_Abs_Loading'] = loadings['PC2_Loading'].abs()
loadings['PC2_Direction'] = loadings['PC2_Loading'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
loadings_sorted_pc2 = loadings.sort_values(by='PC2_Abs_Loading', ascending=False)
top_features_pc2 = loadings_sorted_pc2.index[:2].tolist()
pc2_label = f"{top_features_pc2[0].replace('_', ' ')} and {top_features_pc2[1].replace('_', ' ')} Index"
plt.figure(figsize=(10, 6))
colors_pc2 = ['green' if d == 'Positive' else 'red' for d in loadings_sorted_pc2['PC2_Direction']]
plt.bar(loadings_sorted_pc2.index, loadings_sorted_pc2['PC2_Abs_Loading'], color=colors_pc2)
plt.title(f"Feature Contribution to PC2 ({pc2_label})")
plt.ylabel("Absolute PC2 Loading")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_pca_pc2_feature_contributions.png")

print("\nTop PC2 Feature Contributions:")
print(loadings_sorted_pc2[['PC2_Loading', 'PC2_Abs_Loading', 'PC2_Direction']])
print(f"PC2 is interpreted as: {pc2_label}")

# Combined PC1 and PC2 loading chart
combined_df = loadings[['PC1_Loading', 'PC2_Loading']].copy()
ax = combined_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'red'], width=0.6)
plt.title("PC1 and PC2 Feature Loadings")
plt.ylabel("Loading Value")
plt.xlabel("Feature")
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.legend(['PC1', 'PC2'], fontsize=10)
plt.tight_layout()
plt.savefig("1_pca_combined_pc1_pc2_loadings.png")

# Save results
merged_df.to_csv("1_pca_borough_scores.csv", index=False)
loadings_sorted_pc1.to_csv("1_pca_pc1_loadings_sorted.csv")
loadings.to_csv("1_pca_full_loadings.csv")
