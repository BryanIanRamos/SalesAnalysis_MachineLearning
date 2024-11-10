import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Create the database connection string
connection_string = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
engine = create_engine(connection_string)

# SQL Query to fetch the required data
query = """
SELECT
    s.salesid,
    s.customerid,
    s.stockcode,
    s.quantity,
    s.unitprice,
    s.totalprice,
    t.day,
    t.month,
    t.year,
    c.country
FROM
    sales s
JOIN
    time t ON s.timeid = t.timeid
JOIN
    customer c ON s.customerid = c.customerid
"""

# Load the data from the database
df = pd.read_sql(query, engine)

# Check for missing values
print("Missing values in the data:")
print(df.isnull().sum())

# Create a new column for the date
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Feature Engineering
df['day_of_week'] = df['Date'].dt.dayofweek
df['month_of_year'] = df['Date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['sales_lag_1'] = df['totalprice'].shift(1)

# Aggregating sales by date
sales_data = df.groupby('Date').agg({
    'totalprice': 'sum',
    'day_of_week': 'mean',
    'month_of_year': 'mean',
    'is_weekend': 'mean',
    'sales_lag_1': 'mean'
}).reset_index()

# Convert dates into numeric format for regression
sales_data['DateNumeric'] = (sales_data['Date'] - pd.Timestamp("1970-01-01")).dt.days

# Load the pre-trained model
model = joblib.load('sales_forecasting_rf_model.pkl')

# Forecast future sales (next 12 months)
future_dates = pd.date_range(
    start=sales_data['Date'].max() + pd.DateOffset(months=1),  # Start from the next month
    periods=12,  # Forecast for 12 months
    freq='MS'  # MS means "month start", so it gives the first day of each month
)

# Create a DataFrame for future dates
future_df = pd.DataFrame({'Date': future_dates})

# Add additional features for future dates
future_df['DateNumeric'] = (future_df['Date'] - pd.Timestamp("1970-01-01")).dt.days
future_df['day_of_week'] = future_df['Date'].dt.dayofweek
future_df['month_of_year'] = future_df['Date'].dt.month
future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

# For 'sales_lag_1', use the last known totalprice value from the existing data
future_df['sales_lag_1'] = sales_data['totalprice'].iloc[-1]

# Predict future sales
future_sales = model.predict(future_df[['DateNumeric', 'day_of_week', 'month_of_year', 'is_weekend', 'sales_lag_1']])

# Prepare the data for JSON output
forecast_data = []
for date, sales in zip(future_dates, future_sales):
    forecast_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'sales': sales
    })

# Convert the forecast data to JSON format
json_data = json.dumps(forecast_data, indent=4)
print(json_data)  # For debugging, print the JSON data

# Plot future sales (Optional: For visualization)
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_sales, label='Forecasted Sales', color='green')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast for Next Year (Random Forest)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
