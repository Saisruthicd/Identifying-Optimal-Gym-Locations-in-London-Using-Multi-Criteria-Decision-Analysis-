import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load PCA, clustering, gym count, and retail rent datasets
def load_and_merge_data():
    boroughs = pd.read_csv("2_clustered_boroughs.csv")
    gym_counts = pd.read_csv("gym_count.csv")
    retail_rents = pd.read_csv("London_Borough_Retail_Rent_Prices.csv")

    # Standardise column names for consistency
    gym_counts.rename(columns={'lad_name': 'Borough', 'count_of_gyms': 'Gym_Count'}, inplace=True)
    retail_rents.rename(columns={'Price (£/sq.ft/year)': 'Retail_Rent'}, inplace=True)

    # Merge all datasets using 'Borough' as the common key
    df = pd.merge(boroughs, gym_counts[['Borough', 'Gym_Count']], on='Borough', how='left')
    df = pd.merge(df, retail_rents[['Borough', 'Retail_Rent']], on='Borough', how='left')

    # Handle missing values by applying default substitutes
    df['Gym_Count'] = df['Gym_Count'].fillna(0)
    df['Retail_Rent'] = df['Retail_Rent'].fillna(df['Retail_Rent'].median())

    # Display any boroughs that failed to merge correctly
    print("Missing boroughs after merge:", df[df['Gym_Count'].isna() | df['Retail_Rent'].isna()]['Borough'].tolist())
    return df

# Normalize features to a common 0–1 scale and invert where lower values are more desirable
def normalize_scores(df):
    df['PC1_Score'] = (df['PC1'] - df['PC1'].min()) / (df['PC1'].max() - df['PC1'].min())
    df['PC2_Score'] = 1 - (df['PC2'] - df['PC2'].min()) / (df['PC2'].max() - df['PC2'].min())
    df['Gym_Score'] = 1 - (df['Gym_Count'] - df['Gym_Count'].min()) / (df['Gym_Count'].max() - df['Gym_Count'].min())
    df['Rent_Score'] = 1 - (df['Retail_Rent'] - df['Retail_Rent'].min()) / (df['Retail_Rent'].max() - df['Retail_Rent'].min())
    return df

# Compute AHP weights using the eigenvalue method and verify consistency
def compute_ahp_weights():
    pairwise_matrix = np.array([
        [1, 3, 5, 7],
        [1 / 3, 1, 2, 3],
        [1 / 5, 1 / 2, 1, 2],
        [1 / 7, 1 / 3, 1 / 2, 1]
    ])

    # Calculate eigenvalues and extract principal eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    weights = eigenvectors[:, np.argmax(eigenvalues)].real
    weights /= weights.sum()

    # Calculate consistency ratio to validate pairwise matrix
    n = pairwise_matrix.shape[0]
    lambda_max = np.max(eigenvalues).real
    consistency_index = (lambda_max - n) / (n - 1)
    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12}.get(n)
    consistency_ratio = consistency_index / random_index
    print(f"Consistency Ratio: {consistency_ratio:.2f} (CR < 0.1 is acceptable)")

    return weights, consistency_ratio

# Calculate AHP composite score and rank based on given weights
def calculate_ahp_scores(df, w_pc1, w_pc2, w_gym, w_rent, name):
    df[f"AHP_{name}"] = (
        df['PC1_Score'] * w_pc1 +
        df['PC2_Score'] * w_pc2 +
        df['Gym_Score'] * w_gym +
        df['Rent_Score'] * w_rent
    )
    df[f"Rank_{name}"] = df[f"AHP_{name}"].rank(ascending=False, method='dense').astype(int)

# Plot the top 10 boroughs for each weight scenario
def visualize_top10(df, score_column, name):
    palette_map = {
        'with_rent': 'magma',
        'no_rent': 'viridis',
        'high_rent_sensitive': 'rocket',
        'flat': 'plasma',
        'pairwise_matrix': 'coolwarm'
    }

    top10 = df.nlargest(10, score_column)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top10,
        y='Borough',
        x=score_column,
        hue='Borough',
        dodge=False,
        palette=palette_map.get(name, 'deep'),
        legend=False
    )
    plt.title(f"Top 10 Boroughs for Gym Location ({name.replace('_', ' ').title()} Weights)")
    plt.xlabel("AHP Score")
    plt.ylabel("Borough")
    plt.tight_layout()
    plt.savefig(f"3ahp_top10_barplot_{name}.png")
    plt.close()

# Save AHP scores and ranks to CSV for later use
def save_outputs(df, weight_sets):
    df.to_csv("3ahp_combined_results.csv", index=False)
    sensitivity_cols = ['Borough'] + [f"Rank_{name}" for name in weight_sets]
    df[sensitivity_cols].to_csv("3ahp_sensitivity_ranks.csv", index=False)

# Execute the full AHP workflow across multiple weighting scenarios
def main():
    df = load_and_merge_data()
    df = normalize_scores(df)
    weights_from_matrix, _ = compute_ahp_weights()

    # Define alternative weight configurations for scenario analysis
    weight_sets = {
        'with_rent': (0.4, 0.2, 0.2, 0.2),
        'no_rent': (0.5, 0.3, 0.2, 0.0),
        'high_rent_sensitive': (0.3, 0.2, 0.2, 0.3),
        'flat': (0.25, 0.25, 0.25, 0.25),
        'pairwise_matrix': tuple(weights_from_matrix)
    }

    print("\nTop 3 Boroughs Under Different Weight Configurations:")
    for name, weights in weight_sets.items():
        calculate_ahp_scores(df, *weights, name)
        top = df.sort_values(by=f"AHP_{name}", ascending=False).head(3)
        print(f"{name:20}: {', '.join(top['Borough'])}")
        visualize_top10(df, f"AHP_{name}", name)

    # Validate output by checking overlap with known high-performance boroughs
    actual_top_locations = ["City of London", "Westminster"]
    predicted_top = df.nlargest(5, 'AHP_pairwise_matrix')['Borough'].tolist()
    overlap = len(set(actual_top_locations) & set(predicted_top))
    print(f"Real-world validation accuracy (top 5): {overlap / len(actual_top_locations):.0%}")

    save_outputs(df, weight_sets)

if __name__ == "__main__":
    main()

