import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

fundamentals_data = '/home/danny/Downloads/dataset.csv'
unprocessed_data = pd.read_csv(fundamentals_data)

id_columns = ['Industry','Sector','Company', 'Symbol']

features_SVM_model = ['EBITDA',
                      'Free Cash Flow', 'Total Cash',
                      'Total Revenue', 'Revenue/sh', 'Enterprise Value','Enterprise To Revenue','Enterprise To EBITDA',
                      'Sharpe Ratio', 'Div Payout Ratio','Consecutive Yrs Div Increase',
                      'Piotroski F'	,'Beneish M',	'Altman Z',
                      'S&P 500 Member',  'Dow Jones Member',]

available_features = [columns for columns in features_SVM_model if columns in unprocessed_data.columns]
print(available_features)
print(f"Available features: {len(available_features)} out of {len(features_SVM_model)}")

id_data = unprocessed_data[id_columns].copy()

def preprocess_data(unprocessed_data):
    processed_data = unprocessed_data[available_features].copy()
    for column in processed_data.columns:
        if column in id_columns:
            continue
        if processed_data[column].dtype == 'object':
            # Check if any non-null value exists in the column
            if not processed_data[column].dropna().empty:
                # Process each value individually
                def convert_value(x):
                    # Return directly if not a string
                    if not isinstance(x, str):
                        return x

                    # Handle currency values with $ and commas
                    if '$' in x or ',' in x:
                        try:
                            return float(x.replace('$', '').replace(',', '').strip())
                        except ValueError:
                            return x

                    # Handle percentage values
                    if '%' in x:
                        try:
                            return float(x.replace('%', '').strip()) / 100
                        except ValueError:
                            return x

                    # Try converting to float if it looks numeric
                    try:
                        return float(x)
                    except ValueError:
                        return x

                processed_data[column] = processed_data[column].apply(convert_value)

    return processed_data

data = preprocess_data(unprocessed_data)

for column in available_features:
    try:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    except:
        print(f"{column} could not be converted to numeric")

monetary_features = ['EBITDA', 'Free Cash Flow', 'Total Cash',
                     'Total Revenue', 'Enterprise Value']
ratio_features = ['Enterprise To Revenue', 'Enterprise To EBITDA', 'Sharpe Ratio',
                  'Div Payout Ratio', 'Revenue/sh']
score_features = ['Piotroski F', 'Beneish M', 'Altman Z']
indices_inclusion = ['Dow Jones Member','S&P 500 Member', 'Russell 3000 Member']


for column in monetary_features:
    median_values = data[column].median()
    data[column] = data[column].fillna(median_values)
    data[f'{column}_log'] = np.log1p(data[column].clip(lower=1))

for column in ratio_features:
    median_values = data[column].median()
    data[column] = data[column].fillna(median_values)

for column in score_features:
    median_values = data[column].median()
    data[column] = data[column].fillna(median_values)


def outliers(data, columns, lower_percentile=0.01, upper_percentile=0.99):
    data_copy = data.copy()
    for column in columns:
        if column in data_copy.columns:
            lower_bound = data_copy[column].quantile(lower_percentile)
            upper_bound = data_copy[column].quantile(upper_percentile)
            # Use clip instead of between to handle outliers
            data_copy[column] = data_copy[column].clip(lower=lower_bound, upper=upper_bound)
    return data_copy

for category in [monetary_features, score_features, ratio_features]:
    data = outliers(data, category)

data = outliers(data, features_SVM_model)


final_monetary_feature = available_features + [f'{column}_log' for column in monetary_features]
print(f"{final_monetary_feature}\n")
def create_risk_score(row):
    score = 0

    if pd.notna(row.get('Altman Z')):
        if row['Altman Z'] > 3:
            score += -4
        elif row['Altman Z'] <1.8:
            score += 4

    if pd.notna(row.get('Piotroski F')):
        if row['Piotroski F'] >= 7:
            score -= 4
        elif row['Piotroski F'] <= 3:
            score += 4

    if pd.notna(row.get('Div Yield')):
        if row['Div Yield'] < 1:
            score -=1
        elif row['Div Yield'] > 3:
            score -= 3
        elif row['Div Yield'] < 8:
            score -= 3
        elif row['Div Yield'] > 12:
            score += 3

        # Add dividend consistency factor if available
    if pd.notna(row.get('Dividend')) and pd.notna(row.get('Consecutive Yrs Div Increase')):
        if row['Dividend'] > 0:
            if row['Consecutive Yrs Div Increase'] > 10:
                score -= 5  # Long dividend history significantly reduces risk
            elif row['Consecutive Yrs Div Increase'] > 5:
                score -= 2  # Moderate dividend history reduces risk

    elif pd.notna(row.get('S&P 500 Member')):
        if row['S&P 500 Member'] == 1:
            score -= 4

    elif pd.notna(row.get('Dow Jones Member')):
        if row['Dow Jones Member'] == 1:
            score -= 6


    # Cash position
    if pd.notna(row.get('Total Cash')):
        if row['Total Cash'] > 50_000_000_000:  # Over $50 billion cash
            score -= 6  # Extremely strong cash position
        elif row['Total Cash'] > 10_000_000_000:  # Over $10 billion cash
            score -= 3  # Strong cash position
        elif row['Total Cash'] < 10_000_000 and pd.notna(row.get('Market Cap')) and row['Market Cap'] > 1:
            # Low cash for company size
            score += 2  # Cash crunch risk
        # Add industry-specific risk adjustments
    if pd.notna(row.get('Industry')):
        # High-risk industries
        high_risk_industries = ['Biotechnology', 'Pharmaceutical', 'Oil & Gas E&P',
                                'Semiconductor', 'Airlines', 'Cryptocurrency']

        # Medium-risk industries
        medium_risk_industries = ['Software', 'Technology', 'Entertainment',
                                  'Retail', 'Automobiles']

        # Lower-risk industries
        low_risk_industries = ['Utilities', 'Consumer Staples', 'Insurance', 'REIT']

        if any(industry in str(row['Industry']) for industry in high_risk_industries):
            score += 10  # Substantial risk increase for high-risk industries
        elif any(industry in str(row['Industry']) for industry in medium_risk_industries):
            score += 2  # Modest risk increase for medium-risk industries
        elif any(industry in str(row['Industry']) for industry in low_risk_industries):
            score -= 4  # Small risk reduction for traditionally stable industries

    if pd.notna(row.get('Total Revenue')):
        if row['Total Revenue'] > 100_000_000_000:  # Over $100 billion
            score -= 6  # Large, stable revenue
        elif row['Total Revenue'] > 1_000_000_000:  # Over $1 billion
            score -= 3  # Good revenue size
        elif row['Total Revenue'] < 100_000_000:  # Under $100 million
            score += 4  # Small revenue base
        elif row['Total Revenue'] < 10_000_000:  # Under $10 million
            score += 6  # Very small revenue base
    return score


data['Risk Score'] = data.apply(create_risk_score, axis=1)

data['Risk Category'] = pd.cut(
    data['Risk Score'],
    bins=[-float('inf'), -3,3, float('inf')],
    labels = ['Low', 'Medium', 'High'],
)

for column in id_columns:
    if column in id_data.columns:
        data[column] = id_data[column].values

valid_columns = [col for col in final_monetary_feature + id_columns + ['Risk Score','Risk Category']]
processed_data_prepared = data[valid_columns]
processed_data_prepared.to_csv('/home/danny/Desktop/processed_data23.csv', index=False)
print(f"Processed data saved with{len(processed_data_prepared)} rows and {len(final_monetary_feature)} columns")

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mor(data_path):

    fundamentals_data = pd.read_csv(data_path)
    group_companies = fundamentals_data.groupby(['Sector', 'Industry']).size().reset_index(name='countofCompanies')

    group_companies = group_companies.sort_values(by=['countofCompanies'], ignore_index=True)

    for row in group_companies.itertuples():
        print(f"Sector: {row.Sector}, Industry {row.Industry}, Number of Companies: {row.countofCompanies}")

    industry_counts = (
        fundamentals_data
        .groupby("Sector")
        .size()
        .reset_index(name="countOfCompanies")
        .sort_values(by="countOfCompanies", ascending=False, ignore_index=True)
    )

    plt.figure(figsize = (10,10))
    plt.pie(
        industry_counts["countOfCompanies"],
        labels = industry_counts["Sector"],
        autopct = '%5.2f%%',
        textprops={'fontsize': 8},
        startangle = 140
    )

    plt.title("Companies Distribution by Industry")
    plt.tight_layout()
    plt.show()
    return group_companies

if __name__ == "__main__":
    data_path = '/home/danny/Downloads/dataset.csv'
    result = mor(data_path)

"""

"""

missing_values = fundamentals_data.isnull().sum()
print("\nMissing values:")
print(missing_values[missing_values > 0])

missingvalue_percentage = (missing_values / len(fundamentals_data)) * 100
print("\nMissing values percentage:")
print(missingvalue_percentage[missing_values > 0])

features_SVM_model = ['EBITDA','Gross Profits','Free Cash Flow', 'Total Cash',	'Total Debt',	'Total Revenue',
                      'Revenue/sh',
                      'Enterprise To Revenue',	'Enterprise To EBITDA'	,'Enterprise Value', 'Sharpe Ratio',
                      'Div Payout Ratio',
                      'Piotroski F'	,'Beneish M',	'Altman Z', 'Company', 'Symbol']


fundamentals_data_prepared = fundamentals_data[features_SVM_model]
fundamentals_data_prepared = fundamentals_data_prepared.dropna()
print(f'{fundamentals_data_prepared}')
"""
