import pandas as pd

# Load the gym dataset
df = pd.read_csv("gym_data.csv")

# Drop completely empty rows
df = df.dropna(how='all')

# Remove the last 4 rows
df = df.iloc[:-4]

# Rename key columns for clarity and consistency
df.rename(columns={
    '@lat': 'Latitude',
    '@lon': 'Longitude',
    'name': 'Gym Name'
}, inplace=True)

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
df.to_csv("gym_data_c.csv", index=False)
print("Cleaned file saved as 'gym_data_c.csv'")
