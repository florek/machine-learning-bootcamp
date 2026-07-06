import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


data = {
    "size":   ["XL", "L", "M", np.nan, "M", "M"],
    "color":  ["red", "green", "blue", "green", "red", "green"],
    "gender": ["female", "male", np.nan, "female", "female", "male"],
    "price":  [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    "weight": [500.0, 450.0, 300.0, np.nan, 410.0, np.nan],
    "bought": ["yes", "no", "yes", "no", "yes", "no"]
}

df_raw = pd.DataFrame(data=data)
df = df_raw.copy()

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(df[["weight"]])
df['weight'] = imputer.transform(df[['weight']])

imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=99.0)
imputer.fit(df[['price']])
df['price'] = imputer.transform(df[['price']])

imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value='L')
df['size'] = imputer.fit_transform(df[['size']]).ravel()

print(df_raw[pd.isnull(df_raw['weight'])])
print(df_raw[~pd.isnull(df_raw['weight'])])
