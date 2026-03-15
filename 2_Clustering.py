import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load PCA dataset containing borough-level socio-economic principal components
df = pd.read_csv("1_pca_borough_scores.csv")
pca_features = ['PC1', 'PC2']

# Elbow Method – determine the optimal number of clusters based on inertia
inertia = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[pca_features])
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(6, 4))
plt.plot(range(2, 8), inertia, marker='o')
plt.axvline(x=3, color='red', linestyle='--', label='k = 3')
plt.title("Elbow Method for K-means Clustering of Boroughs")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion (Inertia)")
plt.legend()
plt.tight_layout()
plt.savefig("2_kmeans_elbow_plot.png")

# Silhouette scores for k=2 to 7
silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df[pca_features])
    silhouette_scores.append(silhouette_score(df[pca_features], labels))

print("Silhouette Scores for K=2 to 7:", silhouette_scores)

# Final clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[pca_features])

# Human-readable labels
centers = pd.DataFrame(kmeans.cluster_centers_, columns=pca_features)
sorted_ids = centers.sort_values('PC1').index.tolist()
label_map = {
    sorted_ids[0]: "High Need (Low PC1)",
    sorted_ids[1]: "Emerging Opportunity",
    sorted_ids[2]: "Affluent & Healthy (High PC1)"
}
df['Cluster_Label'] = df['Cluster'].map(label_map)

# Save clustering results
df.to_csv("2_clustered_boroughs.csv", index=False)

# Cluster scatter plot in PCA space
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=3)

# Scatterplot with labels
sns.scatterplot(
    data=df,
    x='PC1', y='PC2',
    hue='Cluster_Label',
    palette=palette,
    s=130,
    edgecolor='black'
)

# Cluster centers (black Xs)
plt.scatter(
    centers['PC1'], centers['PC2'],
    c='black', s=180, marker='X', label='Cluster Centers'
)

# Annotate outliers (e.g., Affluent group)
affluent = df[df['Cluster_Label'] == "Affluent & Healthy (High PC1)"]
for _, row in affluent.iterrows():
    plt.text(row['PC1'] + 0.1, row['PC2'], row['Borough'], fontsize=10, color='black')

# Final plot setup
plt.title("PCA-Based Clustering of London Boroughs")
plt.xlabel("PC1 (Socioeconomic Index)")
plt.ylabel("PC2 (Secondary Trend)")
plt.legend()
plt.tight_layout()
plt.savefig("2_pca_cluster_scatterplot.png")
plt.close()

# Print summary table
print("\nCluster Summary:")
print(df[['Borough', 'PC1', 'PC2', 'Cluster_Label']].sort_values(by='PC1'))
