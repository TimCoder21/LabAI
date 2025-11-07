import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("spaceship-titanic/train.csv")
df.info()
df.dtypes
print(df.head(100))
cols = df.columns

nan_matrix = df.isnull()
print(nan_matrix.head(100))
print(nan_matrix.sum())

rood_median = df['RoomService'].median()
age_median = df['Age'].median()
food_median = df['FoodCourt'].median()
shopping_median = df['ShoppingMall'].median()
vip_mode = df['VIP'].mode()[0]
spa_median = df['Spa'].median()
vrdeck_median = df['VRDeck'].median()
cabin_mode = df['Cabin'].mode()[0]
planet_mode = df['HomePlanet'].mode()[0]
sleep_mean = df['CryoSleep'].mean
destination_mode = df['Destination'].mode()[0]
name_mode = df['Name'].mode()[0]

df.fillna({'Cabin': cabin_mode}, inplace=True)
df.fillna({'HomePlanet': planet_mode}, inplace=True)
df.fillna({'CryoSleep': sleep_mean}, inplace=True)
df.fillna({'Destination': destination_mode}, inplace=True)
df.fillna({'Name': name_mode}, inplace=True)
df.fillna({'RoomService': rood_median}, inplace=True)
df.fillna({'Age':age_median}, inplace=True)
df.fillna({'ShoppingMall': shopping_median}, inplace=True)
df.fillna({'FoodCourt':food_median}, inplace=True)
df.fillna({'VIP':vip_mode}, inplace=True)
df.fillna({'Spa':spa_median}, inplace=True)
df.fillna({'VRDeck':vrdeck_median}, inplace=True)

nan_matrix = df.isnull()
print(nan_matrix.sum())

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
scaler = MinMaxScaler()
df['RoomService'] = scaler.fit_transform(df[['RoomService']])
scaler = MinMaxScaler()
df['FoodCourt'] = scaler.fit_transform(df[['FoodCourt']])
scaler = MinMaxScaler()
df['ShoppingMall'] = scaler.fit_transform(df[['ShoppingMall']])
scaler = MinMaxScaler()
df['Spa'] = scaler.fit_transform(df[['Spa']])
scaler = MinMaxScaler()
df['VRDeck'] = scaler.fit_transform(df[['VRDeck']])

#df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)
#df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)
#df = pd.get_dummies(df, columns=['Destination'], drop_first=True)
#df = pd.get_dummies(df, columns=['Name'], drop_first=True)

df.drop('VRDeck', axis='columns', inplace= True)
df.drop('ShoppingMall', axis='columns', inplace= True)
df.drop('FoodCourt', axis='columns', inplace= True)
df.drop('CryoSleep', axis='columns', inplace= True)
df.drop('Cabin', axis='columns', inplace= True)
df.drop('RoomService', axis='columns', inplace= True)
df.drop('Spa', axis='columns', inplace= True)
df.drop('Destination', axis='columns', inplace= True)
df.drop('HomePlanet', axis='columns', inplace= True)
df.drop('Name', axis='columns', inplace= True)

df.to_csv("processed_titanic.csv", index=False)
