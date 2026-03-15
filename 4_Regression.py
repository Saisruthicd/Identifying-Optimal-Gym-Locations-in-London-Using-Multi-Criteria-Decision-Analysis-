import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load engineered dataset containing AHP scores and normalized features
df = pd.read_csv("3ahp_combined_results.csv")

# Select predictors and target variable for regression
features = ['PC1_Score', 'PC2_Score', 'Gym_Score', 'Rent_Score']
X = df[features].values
y = df['AHP_pairwise_matrix'].values

# Partition the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Construct a pipeline with robust scaling and random forest regressor
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('rf',     RandomForestRegressor(random_state=42))
])

# Define a grid of hyperparameters for randomized search
param_dist = {
    'rf__n_estimators':      [50, 100],
    'rf__max_depth':         [5, 10],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf':  [2, 4]
}

# Perform 3-fold cross-validation with randomized parameter search
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=12,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

# Output best model configuration and cross-validation performance
best_model = search.best_estimator_
print("Best hyperparameters:", search.best_params_)
print(f"CV R² (3-fold): {search.best_score_:.3f}")

# Evaluate model performance on holdout test set
y_pred = best_model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R²:   {r2:.3f}")

# Generate ML-based prediction and ranking for each borough
df['ML_Pred'] = best_model.predict(X)
df['ML_Rank'] = df['ML_Pred'].rank(ascending=False, method='dense')

# Save predictions to CSV
predictions_df = df[['Borough', 'AHP_pairwise_matrix', 'Rank_pairwise_matrix', 'ML_Pred', 'ML_Rank']].copy()
predictions_df.to_csv("4_ml_predictions.csv", index=False)

# Assess agreement between ML ranking and AHP ranking using Spearman correlation
rho, _ = spearmanr(df['Rank_pairwise_matrix'], df['ML_Rank'])
print(f"Spearman’s ρ: {rho:.3f}")

# Use SHAP to explain top features for a random 50-sample subset of the test set
rf = best_model.named_steps['rf']
X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
idx_sample = np.random.choice(len(X_test_scaled), min(50, len(X_test_scaled)), replace=False)

explainer   = shap.TreeExplainer(rf)
shap_vals   = explainer.shap_values(X_test_scaled[idx_sample])
shap.summary_plot(
    shap_vals,
    X_test_scaled[idx_sample],
    feature_names=features,
    show=False
)
plt.tight_layout()
plt.savefig("4_ml_shap_summary.png")
plt.close()

# Visualise top 10 boroughs by predicted ML score
top10 = df.nlargest(10, 'ML_Pred')
plt.figure(figsize=(10, 6))
sns.barplot(
    x='ML_Pred',
    y='Borough',
    data=top10,
    color='skyblue'
)
plt.title("Top Ten Boroughs by Predicted AHP Suitability")
plt.xlabel("Predicted AHP Score")
plt.ylabel("London Borough")
plt.tight_layout()
plt.savefig("4_ml_top10_barplot.png")
plt.close()
