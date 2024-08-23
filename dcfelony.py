import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('Felony_Sentences.csv')

# Filter data from 2010 to 2020
df = df[(df['SENTENCE_YEAR'] >= 2010) & (df['SENTENCE_YEAR'] <= 2020)]

# Define categories and severity indicators
categories = ['RACE', 'GENDER', 'AGE_GROUP']
severity_indicators = ['OFFENSE', 'OFFENSE_TYPE', 'HOMICIDE_TYPE', 'OFFENSE_SEVERITY_GROUP']

# Initialize results dictionary
results = {category: {'chi2': [], 'p_value': []} for category in categories}

# Calculate Chi-squared values and p-values for each category
for category in categories:
    for year in range(2010, 2021):
        contingency_table = pd.crosstab(df[df['SENTENCE_YEAR'] == year][category], df[df['SENTENCE_YEAR'] == year]['SENTENCE_TYPE'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        results[category]['chi2'].append((year, chi2))
        results[category]['p_value'].append((year, p))

# Sort results by year
for category in categories:
    results[category]['chi2'].sort(key=lambda x: x[0])
    results[category]['p_value'].sort(key=lambda x: x[0])

# Extract sorted values
for category in categories:
    results[category]['chi2'] = [x[1] for x in results[category]['chi2']]
    results[category]['p_value'] = [x[1] for x in results[category]['p_value']]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 20))  # Increased figure size

for i, category in enumerate(categories):
    ax = axes[i]
    ax.plot(range(2010, 2021), results[category]['chi2'], label='Chi-squared', linestyle='-', color='black')
    ax.set_ylabel('Chi-squared')
    ax2 = ax.twinx()
    ax2.plot(range(2010, 2021), results[category]['p_value'], label='p-value', linestyle='--', color='black')
    ax2.axhline(y=0.05, color='red', linestyle=':')
    ax2.set_ylabel('p-value')
    ax.set_title(f'Chi-squared and p-value for {category}')
    ax.set_xticks(range(2010, 2021))
    ax.set_xticklabels(range(2010, 2021), rotation=90)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

plt.tight_layout(pad=3.0)  # Increased padding
plt.savefig('chi-pvalue.png',dpi=300)
plt.show()

