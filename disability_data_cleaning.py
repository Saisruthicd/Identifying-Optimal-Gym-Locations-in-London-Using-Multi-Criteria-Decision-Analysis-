import pandas as pd

# Load the dataset while skipping the first 8 metadata rows
df = pd.read_csv("disability_data.csv", skiprows=8, header=None)

# Rename key columns for clarity and consistency
df.columns = ['Borough', 'Area Code', 'Total population', 'Disabled People']

# Drop completely empty rows
df = df.dropna(how='all')

# Remove the last 4 footer rows
df = df.iloc[:-4]

# Basic data quality checks: check for missing values and duplicate rows
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()

# Print the result of data integrity checks
if missing == 0 and duplicates == 0:
    print("No missing values or duplicate rows in the dataset.")
else:
    print(f"Missing values: {missing}")
    print(f"Duplicate rows: {duplicates}")

# Save the cleaned dataset
df.to_csv("disability_data_c.csv", index=False)
print("Cleaned dataset saved as 'disability_data_c.csv'")
