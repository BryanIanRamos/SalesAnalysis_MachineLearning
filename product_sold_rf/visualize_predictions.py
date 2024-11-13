import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def get_database_connection():
    """Create database connection using environment variables"""
    try:
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        return create_engine(db_url)
    except Exception as e:
        print(f"Error creating database connection: {e}")
        return None

def fetch_product_data(engine):
    """Fetch product sales data from database"""
    query = """
    WITH daily_product_sales AS (
        SELECT 
            s.stockcode,
            p.description,
            t.day,
            t.month,
            t.year,
            COUNT(DISTINCT s.salesid) as daily_transactions,
            SUM(s.quantity) as daily_quantity,
            SUM(s.totalprice) as daily_revenue,
            COUNT(DISTINCT s.customerid) as unique_customers
        FROM sales s
        JOIN time t ON s.timeid = t.timeid
        JOIN product p ON s.stockcode = p.stockcode
        WHERE s.quantity > 0 AND s.totalprice > 0
        GROUP BY s.stockcode, p.description, t.day, t.month, t.year
    )
    SELECT *
    FROM daily_product_sales
    ORDER BY year, month, day
    """
    
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_features(df):
    """Prepare features for product prediction"""
    # Create date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month_of_year'] = df['date'].dt.month
    
    # Calculate rolling averages for the last 7, 14, and 30 days
    features = []
    for stockcode in df['stockcode'].unique():
        product_data = df[df['stockcode'] == stockcode].sort_values('date')
        
        # Calculate rolling metrics
        for window in [7, 14, 30]:
            product_data[f'qty_last_{window}d_avg'] = product_data['daily_quantity'].rolling(window).mean()
            product_data[f'revenue_last_{window}d_avg'] = product_data['daily_revenue'].rolling(window).mean()
            product_data[f'transactions_last_{window}d_avg'] = product_data['daily_transactions'].rolling(window).mean()
        
        # Calculate days since last sale
        product_data['days_since_last_sale'] = product_data['date'].diff().dt.days
        
        features.append(product_data)
    
    return pd.concat(features).dropna()

def make_predictions(df, model_data):
    """Make predictions using loaded models"""
    clf = model_data['classifier']
    reg = model_data['regressor']
    feature_columns = model_data['feature_columns']
    
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    predictions = []
    
    for stockcode in df['stockcode'].unique():
        product_data = df[df['stockcode'] == stockcode].sort_values('date').copy()
        
        for future_date in future_dates:
            features = {
                'day_of_week': future_date.dayofweek,
                'is_weekend': int(future_date.dayofweek in [5, 6]),
                'month_of_year': future_date.month,
                'qty_last_7d_avg': product_data['daily_quantity'].tail(7).mean(),
                'qty_last_14d_avg': product_data['daily_quantity'].tail(14).mean(),
                'qty_last_30d_avg': product_data['daily_quantity'].tail(30).mean(),
                'revenue_last_7d_avg': product_data['daily_revenue'].tail(7).mean(),
                'revenue_last_14d_avg': product_data['daily_revenue'].tail(14).mean(),
                'revenue_last_30d_avg': product_data['daily_revenue'].tail(30).mean(),
                'transactions_last_7d_avg': product_data['daily_transactions'].tail(7).mean(),
                'transactions_last_14d_avg': product_data['daily_transactions'].tail(14).mean(),
                'transactions_last_30d_avg': product_data['daily_transactions'].tail(30).mean(),
                'days_since_last_sale': (future_date - product_data['date'].iloc[-1]).days
            }
            
            X_pred = pd.DataFrame([features])[feature_columns]
            will_sell = clf.predict(X_pred)[0]
            quantity = reg.predict(X_pred)[0] if will_sell else 0
            
            predictions.append({
                'date': future_date,
                'stockcode': stockcode,
                'description': product_data['description'].iloc[0],
                'will_sell': will_sell,
                'predicted_quantity': round(quantity) if quantity > 0 else 0
            })
    
    return pd.DataFrame(predictions)

def create_visualizations(predictions_df, output_dir='outputs'):
    """Create and save visualizations"""
    # Plot 1: Products predicted to sell each day
    plt.figure(figsize=(12, 6))
    daily_products = predictions_df[predictions_df['will_sell'] == 1].groupby('date').size()
    plt.bar(daily_products.index, daily_products.values)
    plt.title('Number of Products Predicted to Sell Each Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_products_forecast.png')
    plt.close()
    
    # Plot 2: Top 10 products by predicted quantity
    plt.figure(figsize=(12, 6))
    top_products = predictions_df.groupby('stockcode')['predicted_quantity'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_products.index, y=top_products.values)
    plt.title('Top 10 Products by Predicted Weekly Quantity')
    plt.xlabel('Stock Code')
    plt.ylabel('Predicted Quantity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_products_forecast.png')
    plt.close()

    # Plot 3: Line chart for top 5 products daily forecast
    plt.figure(figsize=(14, 7))
    top_5_products = predictions_df.groupby('stockcode')['predicted_quantity'].sum().nlargest(5).index
    
    for product in top_5_products:
        product_data = predictions_df[predictions_df['stockcode'] == product]
        plt.plot(product_data['date'], 
                product_data['predicted_quantity'], 
                marker='o', 
                label=f"{product} - {product_data['description'].iloc[0][:30]}")
    
    plt.title('Daily Quantity Forecast - Top 5 Products')
    plt.xlabel('Date')
    plt.ylabel('Predicted Quantity')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_product_line_forecast.png', bbox_inches='tight')
    plt.close()

    # New Plot 4: Pie chart of top 10 products by predicted quantity
    plt.figure(figsize=(12, 8))
    top_10_products = predictions_df.groupby(['stockcode', 'description'])['predicted_quantity'].sum().nlargest(10)
    
    # Create labels with both stockcode and truncated description
    labels = [f"{code}\n({desc[:20]}...)" for (code, desc), _ in top_10_products.items()]
    
    # Calculate percentages
    total_quantity = top_10_products.sum()
    sizes = [(qty/total_quantity)*100 for qty in top_10_products]
    
    # Create pie chart with percentage labels
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Predicted Sales Volume (Top 10 Products)')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/top_products_distribution_pie.png', bbox_inches='tight')
    plt.close()

    # Optional: Another pie chart for weekday vs weekend predictions
    plt.figure(figsize=(10, 8))
    weekend_mask = predictions_df['date'].dt.dayofweek.isin([5, 6])
    weekend_total = predictions_df[weekend_mask]['predicted_quantity'].sum()
    weekday_total = predictions_df[~weekend_mask]['predicted_quantity'].sum()
    
    labels = ['Weekday Sales', 'Weekend Sales']
    sizes = [weekday_total, weekend_total]
    colors = ['#ff9999', '#66b3ff']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Predicted Sales: Weekday vs Weekend')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/weekday_weekend_distribution_pie.png', bbox_inches='tight')
    plt.close()

def main():
    """Main function to load model and create visualizations"""
    # Create output directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the saved models
    print("Loading models...")
    try:
        model_data = joblib.load('product_forecast/product_prediction_models.pkl')
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure the model has been trained and saved.")
        return
    
    # Get database connection and fetch data
    print("Loading and preparing data...")
    try:
        engine = get_database_connection()
        if engine is None:
            raise Exception("Failed to create database connection")
        
        df = fetch_product_data(engine)
        if df is None:
            raise Exception("Failed to fetch data from database")
            
        print(f"Retrieved {len(df)} records from database")
        df = prepare_features(df)
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return
    
    # Make predictions
    print("Making predictions...")
    predictions_df = make_predictions(df, model_data)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(predictions_df, output_dir)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total products predicted to sell: {predictions_df['will_sell'].sum()}")
    print("\nTop 5 products by predicted weekly quantity:")
    top_5 = predictions_df.groupby(['stockcode', 'description'])['predicted_quantity'].sum().sort_values(ascending=False).head()
    print(top_5)
    
    print("\nVisualizations have been saved to the 'outputs' directory.")

if __name__ == "__main__":
    main() 