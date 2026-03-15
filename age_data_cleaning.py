import pandas as pd

# Load the dataset and skip the first 7 metadata rows
df = pd.read_csv("age_18_64_data.csv", skiprows=7)

# Drop completely empty rows
df = df.dropna(how='all')

# Remove the last 4 footer rows (which typically contain notes or totals)
df = df.iloc[:-4]

# Rename key columns for clarity and consistency
df.rename(columns={
    'local authority: county / unitary (as of April 2023)': 'Borough',
    'mnemonic': 'Area Code'
}, inplace=True)

# Define age bands that fall within the 18–64 range
age_columns = [
    'Aged 18 years',
    'Aged 19 years',
    'Aged 20 to 24 years',
    'Aged 25 to 34 years',
    'Aged 35 to 49 years',
    'Aged 50 to 64 years'
]

# Calculate total population aged 18 to 64 by summing across age bands
df['Age 18 to 64 Total'] = df[age_columns].sum(axis=1, skipna=True)

# Basic data quality checks: missing values and duplicate rows
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()

# Print data integrity results
if missing == 0 and duplicates == 0:
    print("No missing values or duplicate rows in the dataset.")
else:
    print(f"Missing values: {missing}")
    print(f"Duplicate rows: {duplicates}")

# Save the cleaned dataset
df.to_csv("age_18_64_data_c.csv", index=False)
print("Cleaned file saved as 'age_18_64_data_c.csv'")