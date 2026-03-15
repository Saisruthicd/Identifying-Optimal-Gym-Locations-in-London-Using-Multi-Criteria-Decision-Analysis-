import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load prediction results combining AHP scores and ML predictions
df = pd.read_csv("4_ml_predictions.csv")

# Extract top 5 boroughs based on AHP scores and ML predicted scores
top5_ahp = df.sort_values(by="AHP_pairwise_matrix", ascending=False).head(5).copy()
top5_ml = df.sort_values(by="ML_Pred", ascending=False).head(5).copy()

# Combine both sets to create a unique list of top boroughs from both methods
top_boroughs = pd.unique(pd.Series(top5_ahp["Borough"].tolist() + top5_ml["Borough"].tolist()))

# Reshape dataframe for visual comparison between AHP and ML scores
compare_df = df[df["Borough"].isin(top_boroughs)][["Borough", "AHP_pairwise_matrix", "ML_Pred"]]
compare_df = pd.melt(compare_df, id_vars="Borough", var_name="Source", value_name="Score")

# Improve source labels for plot readability
compare_df["Source"] = compare_df["Source"].map({
    "AHP_pairwise_matrix": "AHP Score",
    "ML_Pred": "ML Predicted Score"
})

# Order boroughs by average score to group bars meaningfully
compare_df["Borough"] = pd.Categorical(
    compare_df["Borough"],
    categories=compare_df.groupby("Borough")["Score"].mean().sort_values(ascending=False).index,
    ordered=True
)

# Create side-by-side bar plot to compare AHP and ML scores
plt.figure(figsize=(10, 5))
sns.barplot(data=compare_df, x="Score", y="Borough", hue="Source")
plt.title("Top 5 Boroughs: AHP Scores vs ML Predicted Scores")
plt.xlabel("Score")
plt.ylabel("London Borough")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig("5_ahp_vs_ml_comparison_barplot_top5.png")
