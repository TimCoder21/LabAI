import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


df = pd.read_csv("spaceship-titanic/train.csv")
df2 = pd.read_csv("apple_stock.csv")
df3 = pd.read_csv("Fish.csv")
df.head(5) # head(n) returns first n records only. Can also use sample(n) for random n record
df2.info()
df3.info()
cols2 = df2.columns
print(cols2)
cols3 = df3.columns
print(cols3)


age_mean = df['Age'].mean()
vip_mode = df['VIP'].mode()[0]
planet_mode = df['HomePlanet'].mode()[0]
df.fillna({'HomePlanet': planet_mode}, inplace=True)
df.fillna({'Age':age_mean}, inplace=True)
df.fillna({'VIP':vip_mode}, inplace=True)

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

df3.drop('Species', axis='columns', inplace= True)
print(df.corr('pearson'))
print(df.corr('spearman'))
print(df.corr('kendall'))
#print(df.head(10))
#print(df2.head(10))



df.to_csv("processed_titanic.csv", index=False)
df2.to_csv("processed_apple.csv", index=False)
df3.to_csv("processed_Fish.csv", index=False)