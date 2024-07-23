import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data
df = pd.read_csv('Felony_Sentences.csv')

#df = df[df['RACE'].isin(['Black', 'White'])]

# Define the categories
categories = ['GENDER', 'RACE']

# Define line styles and widths
line_styles = ['-', '--', '-.', ':']
line_widths = [1, 2]

# Initialize a figure with a larger vertical size to accommodate the legend box
fig, ax = plt.subplots(figsize=(10, 8))

# Initialize dictionaries to store p-values
pvals_anova = {}
pvals_chi = {}
pvals_combined = {}

# Loop over the categories
for i, category in enumerate(categories):
    pvals_anova[category] = []
    pvals_chi[category] = []
    pvals_combined[category] = []
    for year in df['SENTENCE_YEAR'].unique():
        # Subset the data for the current year
        df_year = df[df['SENTENCE_YEAR'] == year]

        # Perform ANOVA test
        model = ols('SENTENCE_IMPOSED_MONTHS ~ C({})'.format(category), data=df_year).fit()
        pval_anova = sm.stats.anova_lm(model, typ=2)['PR(>F)'][0]
        pvals_anova[category].append(pval_anova)

        # Perform Chi-Square test
        contingency_table = pd.crosstab(df_year[category], df_year['SENTENCE_IMPOSED_MONTHS'])
        chi2, pval_chi, dof, expected = stats.chi2_contingency(contingency_table)
        pvals_chi[category].append(pval_chi)

        # Combine the p-values using Fisher's method
        combined_pval = stats.combine_pvalues([pval_anova, pval_chi], method='fisher')[1]
        pvals_combined[category].append(combined_pval)

    # Plot the p-values
    for j, pvals in enumerate([pvals_anova, pvals_chi, pvals_combined]):
        ax.plot(df['SENTENCE_YEAR'].unique(), pvals[category], 
                color='black', linestyle=line_styles[j], linewidth=line_widths[i%2], 
                label='{}: {}'.format(['ANOVA', 'Chi-Square', 'Fisher'][j], category))

# Add a horizontal line at y=0.05
ax.axhline(y=0.05, color='r', linestyle='--')

# Set the y-label
ax.set_ylabel('p-value')

# Rotate x-axis labels
plt.xticks(rotation=90)

# Move the legend box outside and under the plot
ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')

# Adjust the subplot parameters to make room for the legend box
avg_sentence_by_gender = df.groupby('GENDER')['SENTENCE_IMPOSED_MONTHS'].mean()
print(round(avg_sentence_by_gender,2))
avg_sentence_by_race = df.groupby('RACE')['SENTENCE_IMPOSED_MONTHS'].mean()
print(round(avg_sentence_by_race,2))

plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
# Show the plot
plt.savefig('result.png',dpi=300)
plt.show()

