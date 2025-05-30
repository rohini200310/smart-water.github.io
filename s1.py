import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

water_usage = pd.read_csv("smart_meter_data.csv") 
weather_data = pd.read_csv("weather_data.csv")  

# Convert date columns
water_usage['date'] = pd.to_datetime(water_usage['date'])
weather_data['date'] = pd.to_datetime(weather_data['date'])

# Merge on date
merged_data = pd.merge(water_usage, weather_data, on="date", how="left")
print(merged_data.head())
print(merged_data.columns)



# Time series plot
daily_usage = water_usage.groupby("date")["consumption_liters"].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_usage, x="date", y="consumption_liters")
plt.title("Daily Water Consumption")
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np

# Detect outliers (z-score)
from scipy.stats import zscore
water_usage['zscore'] = water_usage.groupby("household_id")['consumption_liters'].transform(zscore)
outliers = water_usage[water_usage['zscore'] > 3]
print(outliers.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature engineering
merged_data['dayofweek'] = merged_data['date'].dt.dayofweek
features = merged_data[['pressure_level', 'temperature', 'rainfall', 'dayofweek']]
labels = merged_data['consumption_liters']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

import mysql.connector
from sqlalchemy import create_engine

# MySQL connection using SQLAlchemy
#engine = create_engine("mysql+mysqlconnector://username:password@localhost/water_monitoring")

from sqlalchemy import create_engine

# Replace with your actual credentials
username = 'root'
password = 'rohini'
host = 'localhost'        # or an IP address
port = '3306'             # default MySQL port
database = 'smart_water_monitoring'

# Create the engine
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")


# Push DataFrames to MySQL
water_usage.to_sql("household_consumption", con=engine, if_exists='replace', index=False)
weather_data.to_sql("weather_data", con=engine, if_exists='replace', index=False)
print(water_usage.head())




