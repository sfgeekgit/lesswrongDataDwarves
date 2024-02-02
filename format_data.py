import pandas as pd

# Load the CSV file
df = pd.read_csv('dwarves_raw.csv')

# Drop the 'Full Fort Description' column
df.drop('Full Fort Description', axis=1, inplace=True)
df.drop('Fort Name', axis=1, inplace=True)

# Strip "?" and "available" from column names
df.columns = df.columns.str.replace("?", "", regex=False).str.replace("available", "", regex=False)

# Remove all whitespace from column names
df.columns = df.columns.str.replace(" sent", "", regex=False)
df.columns = df.columns.str.replace(" ", "", regex=False)

# Add an ID column as the very first column
df.insert(0, 'ID', range(1, 1 + len(df)))

# Convert categorical columns to one-hot encoding
categorical_cols = ['Biome']
df = pd.get_dummies(df, columns=categorical_cols)

# Ensure the last two columns are 'Fort_Survived' and 'Fort_Value'
cols = list(df.columns)

cols.remove('FortSurvived')
cols.remove('FortValue')
df = df[cols + ['FortSurvived', 'FortValue']]


# Save the reformatted data to a new CSV file
df.to_csv('dwarves_formated.csv', index=False)
