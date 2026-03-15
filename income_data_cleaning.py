import pandas as pd

# Load the dataset while skipping the first 8 metadata rows
df = pd.read_csv("income_data.csv", skiprows=8, header=None)

# Drop completely empty rows
df = df.dropna(how='all')

# Assign appropriate column names for clarity and consistency
df.columns = ['Borough', 'Area Code', 'GDHI (£m)', 'GDHI per Head (£)']

# Remove the last 7 rows
df = df.iloc[:-7]

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
df.to_csv("income_data_c.csv", index=False)
print("Cleaned dataset saved as 'income_data_c.csv'")
