import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm

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

def get_database_connection():
    load_dotenv()
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(connection_string)

def fetch_product_data(engine):
    query = """
    SELECT 
        s.stockcode,
        p.description,
        t.day,
        t.month,
        t.year,
        COUNT(s.salesid) as daily_transactions,
        SUM(s.quantity) as daily_quantity,
        SUM(s.totalprice) as daily_revenue
    FROM 
        sales s
    JOIN 
        product p ON s.stockcode = p.stockcode
    JOIN 
        time t ON s.timeid = t.timeid
    GROUP BY 
        s.stockcode, p.description, t.day, t.month, t.year
    """
    return pd.read_sql(query, engine)

def prepare_features(df):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    return df

def make_predictions(df, model_data, forecast_period='daily', periods=7):
    """Make predictions using loaded models"""
    clf = model_data['classifier']
    reg = model_data['regressor']
    feature_columns = model_data['feature_columns']
    
    # Filter for top products
    recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
    top_products = (recent_data.groupby('stockcode')['daily_quantity']
                   .sum()
                   .sort_values(ascending=False)
                   .head(len(df['stockcode'].unique()) // 4)
                   .index)
    
    # Set up forecast dates
    start_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    if forecast_period == 'daily':
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        period_label = f"Next {periods} Days"
    elif forecast_period == 'weekly':
        future_dates = pd.date_range(start=start_date, periods=periods*7, freq='D')
        period_label = f"Next {periods} Weeks"
    else:  # monthly
        future_dates = pd.date_range(start=start_date, periods=periods*30, freq='D')
        period_label = f"Next {periods} Months"
    
    predictions = []
    print(f"\nMaking predictions for {len(top_products)} products...")
    for stockcode in tqdm(top_products, desc="Processing products"):
        product_data = df[df['stockcode'] == stockcode].sort_values('date')
        
        for future_date in future_dates:
            # Calculate rolling averages
            features = {
                'day_of_week': future_date.dayofweek,
                'month_of_year': future_date.month,
                'is_weekend': int(future_date.dayofweek in [5, 6]),
                'qty_last_7d_avg': product_data['daily_quantity'].tail(7).mean(),
                'qty_last_14d_avg': product_data['daily_quantity'].tail(14).mean(),
                'qty_last_30d_avg': product_data['daily_quantity'].tail(30).mean(),
                'revenue_last_7d_avg': product_data['daily_revenue'].tail(7).mean(),
                'revenue_last_14d_avg': product_data['daily_revenue'].tail(14).mean(),
                'revenue_last_30d_avg': product_data['daily_revenue'].tail(30).mean(),
                'transactions_last_7d_avg': product_data['daily_transactions'].tail(7).mean(),
                'transactions_last_14d_avg': product_data['daily_transactions'].tail(14).mean(),
                'transactions_last_30d_avg': product_data['daily_transactions'].tail(30).mean(),
                'days_since_last_sale': (future_date - product_data['date'].max()).days
            }
            
            # Create DataFrame with features in correct order
            X_pred = pd.DataFrame([features])
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in X_pred.columns:
                    print(f"Warning: Missing feature '{col}' in prediction data")
                    print(f"Available features: {list(X_pred.columns)}")
                    return None, None
            
            # Select only the features used by the model
            X_pred = X_pred[feature_columns]
            
            will_sell = clf.predict(X_pred)[0]
            quantity = reg.predict(X_pred)[0] if will_sell else 0
            
            predictions.append({
                'date': future_date,
                'stockcode': stockcode,
                'description': product_data['description'].iloc[0],
                'will_sell': will_sell,
                'predicted_quantity': round(quantity) if quantity > 0 else 0
            })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Aggregate predictions based on period
    if forecast_period != 'daily':
        if forecast_period == 'weekly':
            predictions_df['period'] = predictions_df['date'].dt.isocalendar().week
        else:  # monthly
            predictions_df['period'] = predictions_df['date'].dt.month
            
        predictions_df = predictions_df.groupby(['stockcode', 'description', 'period']).agg({
            'predicted_quantity': 'sum',
            'will_sell': 'max',
            'date': 'min'
        }).reset_index()
    
    return predictions_df, period_label

def create_and_save_visualizations(predictions_df, period_label, config, base_dir):
    charts_dir = os.path.join(base_dir, 'product_forecast', config['subdir'], 'charts')
    
    # Plot 1: Products predicted to sell
    plt.figure(figsize=(12, 6))
    daily_products = predictions_df[predictions_df['will_sell'] == 1].groupby('date').size()
    plt.bar(daily_products.index, daily_products.values)
    plt.title(f'Number of Products Predicted to Sell\n{period_label}')
    plt.xlabel('Date')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'daily_products_forecast.png'))
    plt.close()
    
    # Plot 2: Top 10 products
    plt.figure(figsize=(12, 6))
    top_products = (predictions_df.groupby(['stockcode', 'description'])['predicted_quantity']
                   .sum()
                   .nlargest(10)
                   .reset_index())
    plt.barh(top_products['description'], top_products['predicted_quantity'])
    plt.title(f'Top 10 Products by Predicted Quantity\n{period_label}')
    plt.xlabel('Predicted Quantity')
    plt.ylabel('Product Description')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'top_products_forecast.png'))
    plt.close()
    
    # Plot 3: Daily forecast for top 5 products
    plt.figure(figsize=(14, 7))
    top_5_products = predictions_df.groupby('stockcode')['predicted_quantity'].sum().nlargest(5).index
    for product in top_5_products:
        product_data = predictions_df[predictions_df['stockcode'] == product]
        product_desc = product_data['description'].iloc[0]
        plt.plot(product_data['date'], product_data['predicted_quantity'], 
                marker='o', label=f"{product} - {product_desc[:30]}...")
    plt.title(f'Daily Quantity Forecast - Top 5 Products\n{period_label}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Quantity')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'top_5_products_forecast.png'))
    plt.close()

def save_prediction_data(predictions_df, config, base_dir):
    data_dir = os.path.join(base_dir, 'product_forecast', config['subdir'], 'data')
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_save = predictions_df.copy()
    
    # Convert datetime columns to string format
    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
    
    # Convert to records format and handle any remaining datetime objects
    prediction_data = []
    for record in df_to_save.to_dict(orient='records'):
        # Convert any remaining datetime objects to strings
        processed_record = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                processed_record[key] = value.strftime('%Y-%m-%d')
            else:
                processed_record[key] = value
        prediction_data.append(processed_record)
    
    # Save as JSON
    json_file = os.path.join(data_dir, f"{config['period']}_predictions.json")
    with open(json_file, 'w') as f:
        json.dump(prediction_data, f, indent=4)
    
    # Save as CSV (original DataFrame with datetime format)
    csv_file = os.path.join(data_dir, f"{config['period']}_predictions.csv")
    predictions_df.to_csv(csv_file, index=False)

def main():
    # Create directory structure
    base_dir = create_figure_directories()
    
    # Configuration flags
    RUN_DAILY = True
    RUN_WEEKLY = True
    RUN_MONTHLY = False
    
    # Load models and prepare data
    print("Loading models and data...")
    model_data = joblib.load('product_prediction_models.pkl')
    print("\nModel expects these features:", model_data['feature_columns'])
    engine = get_database_connection()
    df = fetch_product_data(engine)
    df = prepare_features(df)
    
    # Define forecast configurations
    forecast_configs = [
        {'period': 'daily', 'periods': 7, 'subdir': 'daily_forecast', 'enabled': RUN_DAILY},
        {'period': 'weekly', 'periods': 4, 'subdir': 'weekly_forecast', 'enabled': RUN_WEEKLY},
        {'period': 'monthly', 'periods': 3, 'subdir': 'monthly_forecast', 'enabled': RUN_MONTHLY}
    ]
    
    # Filter to only enabled forecasts
    active_forecasts = [config for config in forecast_configs if config['enabled']]
    
    for config in active_forecasts:
        print(f"\nGenerating {config['period']} forecast...")
        
        # Make predictions
        predictions_df, period_label = make_predictions(
            df, 
            model_data, 
            forecast_period=config['period'],
            periods=config['periods']
        )
        
        # Save visualizations and data
        create_and_save_visualizations(predictions_df, period_label, config, base_dir)
        save_prediction_data(predictions_df, config, base_dir)
        
        print(f"\n{config['period'].title()} Forecast Summary:")
        print(f"Period: {predictions_df['date'].min().strftime('%Y-%m-%d')} to {predictions_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total products predicted to sell: {predictions_df['will_sell'].sum()}")
        print(f"Results saved in figures/product_forecast/{config['subdir']}")

if __name__ == "__main__":
    main() 