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
