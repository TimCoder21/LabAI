import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


df = pd.read_csv("spaceship-titanic/train.csv")
df2 = pd.read_csv("apple_stock.csv")
df3 = pd.read_csv("Fish.csv")
df4 = pd.read_csv("spambase.csv")
df5 = pd.read_csv("diabetes.csv")

df4.head()
print(df4.head(100))
df4.info()

print(df4.isnull().sum())
for col in df4.columns:
    scaler = MinMaxScaler()
    df4[col] = scaler.fit_transform(df4[[col]])


age_mean = df['Age'].median()
vip_mode = df['VIP'].mode()[0]
sleep_mode = df['CryoSleep'].mode()[0]

planet_mode = df['HomePlanet'].mode()[0]
df.fillna({'HomePlanet': planet_mode}, inplace=True)
#df.fillna({'Age':age_mean}, inplace=True)
#df.fillna({'VIP':vip_mode}, inplace=True)
#df.fillna({'CryoSleep':sleep_mode}, inplace=True)


df2['Date'] = pd.to_datetime(df2['Date'])
df2['Date'] = df2['Date'].astype(int)/1000000000

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

scaler = MinMaxScaler()
df3['Length1'] = scaler.fit_transform(df3[['Length1']])
scaler = MinMaxScaler()
df3['Length2'] = scaler.fit_transform(df3[['Length2']])
scaler = MinMaxScaler()
df3['Length3'] = scaler.fit_transform(df3[['Length3']])
scaler = MinMaxScaler()
df3['Height'] = scaler.fit_transform(df3[['Height']])
scaler = MinMaxScaler()
df3['Weight'] = scaler.fit_transform(df3[['Weight']])
scaler = MinMaxScaler()
df3['Width'] = scaler.fit_transform(df3[['Width']])


scaler = MinMaxScaler()
df5['Pregnancies'] = scaler.fit_transform(df5[['Pregnancies']])
scaler = MinMaxScaler()
df5['Glucose'] = scaler.fit_transform(df5[['Glucose']])
scaler = MinMaxScaler()
df5['BloodPressure'] = scaler.fit_transform(df5[['BloodPressure']])
scaler = MinMaxScaler()
df5['SkinThickness'] = scaler.fit_transform(df5[['SkinThickness']])
scaler = MinMaxScaler()
df5['Insulin'] = scaler.fit_transform(df5[['Insulin']])
scaler = MinMaxScaler()
df5['BMI'] = scaler.fit_transform(df5[['BMI']])
scaler = MinMaxScaler()
df5['DiabetesPedigreeFunction'] = scaler.fit_transform(df5[['DiabetesPedigreeFunction']])
scaler = MinMaxScaler()
df5['Age'] = scaler.fit_transform(df5[['Age']])


#scaler = MinMaxScaler()
#df2['Adj Close'] = scaler.fit_transform(df2[['Adj Close']])
#scaler = MinMaxScaler()
#df2['Date'] = scaler.fit_transform(df2[['Date']])

df = pd.get_dummies(df, columns=['HomePlanet'])


df.drop('VRDeck', axis='columns', inplace= True)
df.drop('ShoppingMall', axis='columns', inplace= True)
df.drop('FoodCourt', axis='columns', inplace= True)
df.drop('Cabin', axis='columns', inplace= True)
df.drop('RoomService', axis='columns', inplace= True)
df.drop('Spa', axis='columns', inplace= True)
df.drop('Destination', axis='columns', inplace= True)
df.drop('Name', axis='columns', inplace= True)
df.drop('VIP', axis='columns', inplace= True)

df2.drop('Close', axis='columns', inplace= True)
df2.drop('High', axis='columns', inplace= True)
df2.drop('Low', axis='columns', inplace= True)
df2.drop('Open', axis='columns', inplace= True)

df.drop('PassengerId', axis=1, inplace=True)
df.drop('HomePlanet_Earth', axis=1, inplace=True)
df.drop('HomePlanet_Europa', axis=1, inplace=True)
df.drop('HomePlanet_Mars', axis=1, inplace=True)

df3.drop('Species', axis='columns', inplace= True)
print(df.corr('pearson'))
print(df.corr('spearman'))
print(df.corr('kendall'))
#print(df.head(10))
#print(df2.head(10))



df4.to_csv("processed_spambase.csv", index=False)
df.to_csv("processed_titanic.csv", index=False)
df2.to_csv("processed_apple.csv", index=False)
df3.to_csv("processed_Fish.csv", index=False)
df5.to_csv("processed_diadetes.csv", index=False)