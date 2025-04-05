import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

data = {
    'house_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'area_sqft': [1200, 1500, None, 1800, 2000, 2200, 2400, 2600],
    'bedrooms': [2, 3, 3, 4, 4, 3, 5, 'four'],
    'bathrooms': [1, 2, 1.5, 2, 3, 2, 3, 2],
    'age_years': [10, 5, 15, 20, 3, None, 1, 8],
    'location': ['Urban', 'Suburb', 'Urban', 'Rural', 'Suburb', 'Urban', None, 'Rural'],
    'price': [250000, 350000, 275000, 400000, 450000, 375000, 500000, 420000],
    'last_renovated': ['2015', '2018', 'Never', '2005', '2020', None, '2021', '2017']
}
df = pd.DataFrame(data)
#print("Original Data: ")
#print(df)
#print(df.info())
#print(df.describe(include='all'))
#print(df.isnull().sum())
#print(df.duplicated().sum())
#Data cleaning and Preproceesing
pd.set_option('future.no_silent_downcasting',True)
df['area_sqft']= df['area_sqft'].fillna(df['area_sqft'].median())
df['age_years'] = df['age_years'].fillna(df['age_years'].median())
df['location']=df['location'].fillna(df['location'].mode()[0])
df['is_renovated']=df['last_renovated'].apply(lambda x: 0 if x=='Never' or pd.isna(x) else 1)
df.drop('last_renovated',axis = 1, inplace = True)
df['bedrooms']=pd.to_numeric(df['bedrooms'].replace('four',4))
df.drop('house_id',axis=1,inplace=True)
#print(df)
#Outlier Detection
numeric_cols = ['area_sqft','bedrooms','bathrooms','age_years','price']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = df[(df[col]<lower_bound) | (df[col]>upper_bound)]
    print(f"{col}:{len(outliers)} Outliers detected")

# Feature engineering
df['price_per_sqft']=df['price']/df['area_sqft']
df['total_rooms']=df['bedrooms']+df['bathrooms']
df['age_group']=pd.cut(df['age_years'],bins=[0,5,10,20,50],labels=['new','recent','old','very_old'])
print("\nData after cleaning and feature engineering:")
#print(df.head())
#Data transformation
df = pd.get_dummies(df,columns=['location'],prefix='loc') #one-hot encoding
age_group_mapping = {'new':0,'recent':1,'old':2,'very_old':3}
df['age_group']=df['age_group'].map(age_group_mapping)
scaler = MinMaxScaler()
scaled_cols = ['area_sqft','bedrooms','bathrooms','age_years']
df[scaled_cols]=scaler.fit_transform(df[scaled_cols])
print("\nData after encoding and scaling: ")
print(df.head())
# 5.1 Check for remaining missing values
print("\nRemaining Missing Values:")
print(df.isnull().sum())

# 5.2 Verify data types
print("\nFinal Data Types:")
print(df.dtypes)

# 5.3 Check the final shape
print("\nFinal Data Shape:", df.shape)

# 5.4 Save cleaned data
df.to_csv('cleaned_house_data.csv', index=False)