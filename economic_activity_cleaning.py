import pandas as pd

# Load the dataset while skipping the first 7 metadata rows
df = pd.read_csv("economic_activity_data.csv", skiprows=7)

# Drop completely empty rows
df = df.dropna(how='all')

# Remove the last 4 footer rows (which often contain totals or notes)
df = df.iloc[:-4]

# Rename key columns for clarity and consistency
df.rename(columns={
    'local authority: district / unitary (as of April 2023)': 'Borough',
    'mnemonic': 'Area Code'
}, inplace=True)

# Define columns representing economically active individuals
economic_columns = [
    "Economically active (excluding full-time students)",
    "Economically active and a full-time student"
]

# Calculate total economically active population by summing selected columns
df['Total Economically Active Population'] = df[economic_columns].sum(axis=1, skipna=True)

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
df.to_csv("economic_activity_data_c.csv", index=False)
print("Cleaned dataset saved as 'economic_activity_data_c.csv'")
