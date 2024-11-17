import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib
from dotenv import load_dotenv
import os
import json
import time

def create_figure_directories():
    """Create standardized directory structure for all predictions"""
    base_dir = 'figures'
    directories = {
        'customer_segmentation': ['charts', 'data'],
        'product_forecast': {
            'daily_forecast': ['charts', 'data'],
            'weekly_forecast': ['charts', 'data'],
            'monthly_forecast': ['charts', 'data']
        },
        'sales_forecast': ['charts', 'data']
    }
    
    for main_dir, subdirs in directories.items():
        if isinstance(subdirs, list):
            for subdir in subdirs:
                os.makedirs(os.path.join(base_dir, main_dir, subdir), exist_ok=True)
        else:
            for forecast_type, forecast_subdirs in subdirs.items():
                for subdir in forecast_subdirs:
                    os.makedirs(os.path.join(base_dir, main_dir, forecast_type, subdir), exist_ok=True)
    
    return base_dir

def fetch_data():
    # Load environment variables
    load_dotenv()
    
    # Set up database connection
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(connection_string)
    
    # SQL Query
    query = """
    SELECT 
        t.day,
        t.month,
        t.year,
        SUM(s.totalprice) as daily_sales
    FROM 
        sales s
    JOIN 
        time t ON s.timeid = t.timeid
    GROUP BY 
        t.day, t.month, t.year
    ORDER BY 
        t.year, t.month, t.day
    """
    
    return pd.read_sql(query, engine)

def prepare_data(df):
    # Create date column
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Create features
    df['DateNumeric'] = (df['Date'] - pd.Timestamp("1970-01-01")).dt.days
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month_of_year'] = df['Date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['sales_lag_1'] = df['daily_sales'].shift(1)
    
    return df.dropna()

def forecast_future_sales(model, sales_data, days):
    # Create future dates
    future_dates = pd.date_range(start=sales_data['Date'].max() + pd.Timedelta(days=1), 
                               periods=days)
    
    # Create features for prediction
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['DateNumeric'] = (future_df['Date'] - pd.Timestamp("1970-01-01")).dt.days
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['month_of_year'] = future_df['Date'].dt.month
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['sales_lag_1'] = sales_data['daily_sales'].iloc[-1]
    
    # Make predictions
    X_future = future_df[['DateNumeric', 'day_of_week', 'month_of_year', 'is_weekend', 'sales_lag_1']]
    future_sales = model.predict(X_future)
    
    return future_dates, future_sales

def create_and_save_visualizations(future_dates, future_sales, days, base_dir):
    charts_dir = os.path.join(base_dir, 'sales_forecast', 'charts')
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_sales, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Customize plot
    plt.title(f'Sales Forecast for Next {days} Days\n({future_dates[0].strftime("%Y-%m-%d")} to {future_dates[-1].strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, (date, sale) in enumerate(zip(future_dates, future_sales)):
        plt.annotate(f'${sale:,.0f}', 
                    (date, sale),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f'forecast_{days}days.png'))
    plt.close()
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(future_dates, future_sales, color='skyblue')
    plt.title(f'Daily Sales Forecast\n({future_dates[0].strftime("%Y-%m-%d")} to {future_dates[-1].strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales ($)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, sale in enumerate(future_sales):
        plt.text(i, sale, f'${sale:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f'forecast_{days}days_bar.png'))
    plt.close()

def save_forecast_data(future_dates, future_sales, days, base_dir):
    data_dir = os.path.join(base_dir, 'sales_forecast', 'data')
    
    # Prepare data for JSON
    forecast_data = {
        'metadata': {
            'forecast_period': f'{days} days',
            'start_date': future_dates[0].strftime('%Y-%m-%d'),
            'end_date': future_dates[-1].strftime('%Y-%m-%d'),
            'total_predicted_sales': float(future_sales.sum()),
            'average_daily_sales': float(future_sales.mean())
        },
        'daily_predictions': [
            {
                'date': date.strftime('%Y-%m-%d'),
                'predicted_sales': float(sales),
                'is_weekend': date.dayofweek in [5, 6]
            }
            for date, sales in zip(future_dates, future_sales)
        ]
    }
    
    # Save to JSON
    json_file = os.path.join(data_dir, f'forecast_{days}days.json')
    with open(json_file, 'w') as f:
        json.dump(forecast_data, f, indent=4)
    
    # Save to CSV
    csv_data = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': future_sales,
        'is_weekend': [date.dayofweek in [5, 6] for date in future_dates]
    })
    csv_file = os.path.join(data_dir, f'forecast_{days}days.csv')
    csv_data.to_csv(csv_file, index=False)

def main():
    # Create directory structure
    base_dir = create_figure_directories()
    
    # Load data
    print("Fetching data...")
    df = fetch_data()
    
    print("Preparing data...")
    sales_data = prepare_data(df)
    
    # Load model
    print("Loading model...")
    try:
        model = joblib.load('sales_forecasting_rf_model.pkl')
    except FileNotFoundError:
        print("Error: Model file 'sales_forecasting_rf_model.pkl' not found!")
        return
    
    # Generate forecasts for different periods
    forecast_periods = [7, 30]  # 7 days and 30 days forecasts
    
    for days in forecast_periods:
        start_time = time.time()
        print(f"\nGenerating {days}-day forecast...")
        
        # Make predictions
        future_dates, future_sales = forecast_future_sales(model, sales_data, days)
        
        # Create and save visualizations
        create_and_save_visualizations(future_dates, future_sales, days, base_dir)
        
        # Save prediction data
        save_forecast_data(future_dates, future_sales, days, base_dir)
        
        # Print summary
        print(f"\n{days}-Day Forecast Summary:")
        print(f"Period: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Total predicted sales: ${future_sales.sum():,.2f}")
        print(f"Average daily sales: ${future_sales.mean():,.2f}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
    print("\nAll forecasts completed successfully!")
    print(f"\nResults saved in {base_dir}/sales_forecast/")
    print("├── charts/")
    print("│   ├── forecast_7days.png")
    print("│   ├── forecast_7days_bar.png")
    print("│   ├── forecast_30days.png")
    print("│   └── forecast_30days_bar.png")
    print("└── data/")
    print("    ├── forecast_7days.json")
    print("    ├── forecast_7days.csv")
    print("    ├── forecast_30days.json")
    print("    └── forecast_30days.csv")

if __name__ == "__main__":
    main() 