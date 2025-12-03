import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
register_matplotlib_converters()
sns.set()


data = {
    'price': [108, 109, 110, 110, 109, np.nan, np.nan, 112, 111, 111]
}
data_range = pd.date_range(start='01-01-2020 09:00', end='01-01-2020 18:00', periods=10)

df = pd.DataFrame(data=data, index=data_range)

plt.figure(figsize=(10, 4))
plt.title('Braki danych')
_ = plt.plot(df.price)
plt.show()
df_plotly = df.reset_index()
df_plotly = df_plotly.dropna()
fig = px.line(df_plotly, 'index', 'price', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()
df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(0)
fig = px.line(df_plotly, 'index', 'price_fill', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(df_plotly['price'].mean())
fig = px.line(df_plotly, 'index', 'price_fill', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(df_plotly['price'].interpolate())
fig = px.line(df_plotly, 'index', 'price_fill', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(method='bfill')
fig = px.line(df_plotly, 'index', 'price_fill', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(method='ffill')
fig = px.line(df_plotly, 'index', 'price_fill', width=600, height=400, title='Szeregi czasowe - braki danych')
fig.show()

