import numpy as np
import pandas as pd
import sklearn


print(sklearn.__version__)


def fetch_financial_data(company='AMZN'):
    import pandas_datareader.data as web
    return web.DataReader(name=company, data_source='stooq')

# załadowanie danych finansowych
print('\n\nZaładowanie danych finansowych...\n\n')
df_raw = fetch_financial_data()
print(df_raw.head())

# utworzenie kopii danych
print('\n\nUtworzenie kopii danych...\n\n')
df = df_raw.copy()
df = df[:5]
print(df.info())

# generowanie nowych zmiennych
print('\n\ngenerowanie nowych zmiennych...\n\n')
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
print(df)

# Dyskretyzacja zmiennej ciągłej
print('\n\nDyskretyzacja zmiennej ciągłej...\n\n')
df = pd.DataFrame(
    data={
        'height': [175., 178.5, 185., 191., 184.5, 183., 168.]
    }
)
print(df)
df['height_cat'] = pd.cut(x=df.height, bins=3)
print(df)
df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195))
print(df)
df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195), labels=['small', 'medium', 'height'])
print(df)
print(pd.get_dummies(df, drop_first=True, prefix='height'))

# Ekstrakcja cech
print('\n\nEkstrakcja cech...\n\n')
df = pd.DataFrame(
    data={
        'lang': [['PL', 'ENG'], ['GER', 'ENG', 'PL', 'FRA'], ['RUS']]
    }
)
print(df)
df['lang_number'] = df['lang'].apply(len)
print(df)
df['PL_flag'] = df['lang'].apply(lambda x: 1 if 'PL' in x else 0)
print(df)
df = pd.DataFrame(
    data={
        'website': ['wp.pl', 'onet.pl', 'google.com']
    }
)
print(df)
new = df.website.str.split('.', expand=True)
df['portal'] = new[0]
df['extension'] = new[1]
print(df)

