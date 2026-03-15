# Identifying Optimal Gym Locations in London Using Multi-Criteria Decision Analysis

## Project Overview
This project identifies the most suitable locations for opening new gyms in London using **Multi-Criteria Decision Analysis (MCDA)** and **Principal Component Analysis (PCA)**.

The analysis combines multiple socioeconomic indicators to evaluate boroughs and determine areas with high potential demand for fitness facilities.

## Objectives
- Identify boroughs with high economic activity and working-age populations
- Analyze demographic and socioeconomic indicators
- Use PCA to reduce dimensionality and identify key influencing factors
- Rank boroughs based on combined indicators

## Dataset
The project uses several datasets related to London borough demographics, including:

- Economically Active Population (%)
- Age 18–64 Population (%)
- Work From Home (%)
- Income per Head
- Disabled Population (%)

These datasets were cleaned and merged for analysis.

## Methodology

### 1. Data Cleaning
Raw datasets were cleaned and standardized to ensure consistency.

### 2. Feature Selection
Key demographic and economic indicators were selected to represent gym demand.

### 3. Principal Component Analysis (PCA)
PCA was applied to:
- Reduce dimensionality
- Identify dominant factors influencing gym demand
- Capture most variance in fewer components

Results:
- **PC1 captures economic activity and working-age population**
- **PC2 captures disability and income factors**

### 4. Interpretation
The PCA results help identify boroughs with strong indicators for potential gym demand.

## Key Results
- KMO Score: **0.71**
- Bartlett’s Test p-value: **0.0000**
- Total Variance Captured: **81.76%**

PC1 represents:
- Economically Active Population
- Age 18–64 Population
- Work From Home %

PC2 represents:
- Disabled Population
- Income per Head

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Factor Analyzer
- Matplotlib
- Seaborn



### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Identifying-Optimal-Gym-Locations-in-London-Using-Multi-Criteria-Decision-Analysis-.git
