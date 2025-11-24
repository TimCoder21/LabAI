import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


df = pd.read_csv("spaceship-titanic/train.csv")
df2 = pd.read_csv("apple_stock.csv")

df.info()
df2.info()
df.dtypes
print(df.head(100))
cols = df.columns
cols2 = df2.columns
print(cols2)
nan_matrix = df.isnull()
print(nan_matrix.head(100))
print(nan_matrix.sum())


age_mean = df['Age'].mean()
vip_mode = df['VIP'].mode()[0]
planet_mode = df['HomePlanet'].mode()[0]



df.fillna({'HomePlanet': planet_mode}, inplace=True)

df.fillna({'Age':age_mean}, inplace=True)

df.fillna({'VIP':vip_mode}, inplace=True)


nan_matrix = df.isnull()
print(nan_matrix.sum())

df2['Date'] = pd.to_datetime(df2['Date'])
df2['Date'] = df2['Date'].astype(int)/1000000000

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

#scaler = MinMaxScaler()
#df2['Adj Close'] = scaler.fit_transform(df2[['Adj Close']])
#scaler = MinMaxScaler()
#df2['Date'] = scaler.fit_transform(df2[['Date']])

df = pd.get_dummies(df, columns=['HomePlanet'])


df.drop('VRDeck', axis='columns', inplace= True)
df.drop('ShoppingMall', axis='columns', inplace= True)
df.drop('FoodCourt', axis='columns', inplace= True)
df.drop('CryoSleep', axis='columns', inplace= True)
df.drop('Cabin', axis='columns', inplace= True)
df.drop('RoomService', axis='columns', inplace= True)
df.drop('Spa', axis='columns', inplace= True)
df.drop('Destination', axis='columns', inplace= True)
df.drop('Name', axis='columns', inplace= True)


df2.drop('Close', axis='columns', inplace= True)
df2.drop('High', axis='columns', inplace= True)
df2.drop('Low', axis='columns', inplace= True)
df2.drop('Open', axis='columns', inplace= True)
df2.drop('Volume', axis='columns', inplace= True)

print(df.head(10))
print(df2.head(10))

df.to_csv("processed_titanic.csv", index=False)
df2.to_csv("processed_apple.csv", index=False)


