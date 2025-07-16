import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf


input_data = '/home/danny/Desktop/001254746_FYP_Code/data/raw_dataset.csv'
output_data = '/home/danny/Desktop/processed_datafinal510.csv'

id_columns = ['Industry', 'Sector', 'Company', 'Symbol']

features_classifier_model = [
    'EBITDA', 'Free Cash Flow', 'Total Cash', 'Total Revenue',
    'Revenue/sh', 'Enterprise Value', 'Enterprise To Revenue', 'Enterprise To EBITDA',
    'Sharpe Ratio', 'Div Payout Ratio', 'Consecutive Yrs Div Increase','ROE','ROA',
     'Beneish M', 'Altman Z',
    'S&P 500 Member'
]

monetary_features = ['EBITDA', 'Free Cash Flow', 'Total Cash', 'Total Revenue', 'Enterprise Value']
ratio_features = ['Enterprise To Revenue', 'Enterprise To EBITDA', 'Sharpe Ratio', 'Div Payout Ratio', 'Revenue/sh','ROE','ROA']
score_features = ['Piotroski F', 'Beneish M', 'Altman Z', 'S&P 500 Member']



def load_data(path):
    return pd.read_csv(path)

def remove_delisted_companies(data, symbol_column='Symbol'):

    symbols = data[symbol_column].dropna().astype(str).str.strip().str.upper().unique().tolist()

    # Download last daily data for all companies at once
    try:
        history_data = yf.download(symbols, period="1d", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Download failed: {e}")
        return data  # Return unchanged

    # Active ticker are the ones which do not have empty daily data
    active_symbols = []
    for symbol in symbols:
        try:
            if not history_data[symbol].dropna().empty:
                active_symbols.append(symbol)
        except KeyError:
            pass  # No data for this symbol, consider it delisted

    print(f"Active companies found: {len(active_symbols)} / {len(symbols)}")

    return data[data[symbol_column].str.strip().str.upper().isin(active_symbols)].reset_index(drop=True)

def preprocess_features(data, available_features):
    processed = data[available_features].copy()

    # Handle object types
    for column in processed.columns:
        if processed[column].dtype == 'object':
            processed[column] = processed[column].apply(convert_value)

    # Ensure numeric
    for column in available_features:
        processed[column] = pd.to_numeric(processed[column], errors='coerce')

    return processed


def convert_value(x):
    if not isinstance(x, str):
        return x
    x = x.strip()

    # Handle negative numbers shown with parentheses
    if x.startswith('(') and x.endswith(')'):
        x = '-' + x[1:-1]  # Remove parentheses and prepend minus sign

    if '$' in x or ',' in x:
        try:
            return float(x.replace('$', '').replace(',', '').strip())
        except ValueError:
            return np.nan
    if '%' in x:
        try:
            return float(x.replace('%', '').strip()) / 100
        except ValueError:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def fill_missing_values(data):
    for column in monetary_features + ratio_features + score_features:
        if column in data.columns:
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)
    return data


def clip_outliers(data, columns, lower=0.01, upper=0.99):
    for column in columns:
        if column in data.columns:
            lower_bound = data[column].quantile(lower)
            upper_bound = data[column].quantile(upper)
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data


def create_risk_score(row):
    score = 0

    if pd.notna(row.get('Altman Z')):
        if row['Altman Z'] > 3:
            score -= 3
        elif row['Altman Z'] < 1.8:
            score += 3

    if pd.notna(row.get('Consecutive Yrs Div Increase')):
        if row['Consecutive Yrs Div Increase'] > 50:
            score -= 6
        if row['Consecutive Yrs Div Increase'] > 25:
            score -= 4
        if row['Consecutive Yrs Div Increase'] > 10:
            score -= 2
        elif row['Consecutive Yrs Div Increase'] > 5:
            score -= 2

    if pd.notna(row.get('S&P 500 Member')) and row['S&P 500 Member'] == 1:
        score -= 4


    if pd.notna(row.get('Total Cash')):
        if row['Total Cash'] > 50_000_000_000:
            score -= 6
        elif row['Total Cash'] > 10_000_000_000:
            score -= 3

    if pd.notna(row.get('Total Revenue')):
        if row['Total Revenue'] > 100_000_000_000:
            score -= 6
        elif row['Total Revenue'] > 20_000_000_000:
            score -= 3
        elif row['Total Revenue'] < 100_000_000:
            score += 4
        elif row['Total Revenue'] < 10_000_000:
            score += 6

    return score

def assign_risk_category(data):
    data['Risk Score'] = data.apply(create_risk_score, axis=1)
    data['Risk Category'] = pd.cut(
        data['Risk Score'],
        bins=[-float('inf'), -3, 3, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    return data



def save_data(data, path):
    data = data.loc[data.isnull().mean(axis=1) <= 0.5]
    data.to_csv(path, index=False)
    print(f"Processed data saved to {path} with {len(data)} rows and {len(data.columns)} columns.")


def plot_sector_distribution(data):
    sector_counts = (
        data.groupby("Sector")
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    plt.figure(figsize=(10, 10))
    plt.pie(
        sector_counts['count'],
        labels=sector_counts['Sector'],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 8}
    )
    plt.title("Companies Distribution by Sector")
    plt.tight_layout()
    plt.show()



def main():
    raw_data = load_data(input_data)
    raw_data = remove_delisted_companies(raw_data)
    id_data = raw_data[id_columns].copy()

    available_features = [f for f in features_classifier_model if f in raw_data.columns]
    data = preprocess_features(raw_data, available_features)
    data = fill_missing_values(data)
    data = clip_outliers(data, available_features)

    data = assign_risk_category(data)

    for col in id_columns:
        data[col] = id_data[col].values

    valid_columns = available_features + id_columns + ['Risk Score', 'Risk Category']
    final_data = data[valid_columns]

    save_data(final_data, output_data)
    plot_sector_distribution(raw_data)


if __name__ == "__main__":
    main()


