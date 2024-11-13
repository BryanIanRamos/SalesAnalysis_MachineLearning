import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

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

def fetch_sales_data(engine):
    """Fetch and prepare sales data from database"""
    query = """
    SELECT 
        s.salesid,
        s.stockcode,
        p.description,
        s.quantity,
        s.unitprice,
        s.totalprice,
        t.day,
        t.month,
        t.year,
        t.hour,
        c.country
    FROM sales s
    JOIN time t ON s.timeid = t.timeid
    JOIN customer c ON s.customerid = c.customerid
    JOIN product p ON s.stockcode = p.stockcode
    WHERE s.quantity > 0
      AND s.totalprice > 0
    ORDER BY t.year, t.month, t.day
    """
    
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_features(df):
    """Prepare features for the model"""
    # Create date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Aggregate daily sales
    daily_sales = df.groupby(['date', 'stockcode', 'description']).agg({
        'quantity': 'sum',
        'totalprice': 'sum',
        'country': 'count'  # Number of transactions
    }).reset_index()
    
    # Create time-based features
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
    
    # Create lag features
    for i in range(1, 4):
        daily_sales[f'lag_{i}'] = daily_sales.groupby('stockcode')['quantity'].shift(i)
    
    return daily_sales.dropna()

def train_and_predict(daily_sales, n_products=10):
    """Train models and make predictions for top products"""
    feature_columns = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_2', 'lag_3']
    performance_metrics = {}
    predictions = {}
    
    # Get top products by sales volume
    top_products = daily_sales.groupby('stockcode')['quantity'].sum().sort_values(ascending=False).head(n_products)
    
    for stockcode in top_products.index:
        print(f"\nTraining model for product {stockcode}")
        product_data = daily_sales[daily_sales['stockcode'] == stockcode].copy()
        
        # Prepare train/test data
        X = product_data[feature_columns]
        y = product_data['quantity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, f'models/model_{stockcode}.joblib')
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        performance_metrics[stockcode] = metrics
        
        # Generate future predictions
        last_date = product_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        
        future_data = pd.DataFrame({
            'date': future_dates,
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
            'lag_1': [product_data['quantity'].iloc[-1]] * 7,
            'lag_2': [product_data['quantity'].iloc[-2]] * 7,
            'lag_3': [product_data['quantity'].iloc[-3]] * 7
        })
        
        predictions[stockcode] = model.predict(future_data[feature_columns])
    
    return predictions, performance_metrics, future_dates

def plot_predictions(predictions, future_dates):
    """Plot the predictions for all products"""
    plt.figure(figsize=(12, 6))
    for product, pred in predictions.items():
        plt.plot(future_dates, pred, marker='o', label=f'Product {product}')
    
    plt.title('7-Day Sales Forecast for Top Products')
    plt.xlabel('Date')
    plt.ylabel('Predicted Quantity')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/sales_forecast.png', bbox_inches='tight')
    plt.close()

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Get database connection
    engine = get_database_connection()
    if engine is None:
        return
    
    # Fetch and prepare data
    print("Fetching data from database...")
    df = fetch_sales_data(engine)
    if df is None:
        return
    
    print("Preparing features...")
    daily_sales = prepare_features(df)
    
    # Train models and make predictions
    print("Training models and making predictions...")
    predictions, metrics, future_dates = train_and_predict(daily_sales)
    
    # Save results
    results_df = pd.DataFrame(predictions, index=future_dates)
    results_df.to_csv('outputs/weekly_sales_forecast.csv')
    pd.DataFrame(metrics).T.to_csv('outputs/model_performance_metrics.csv')
    
    # Plot results
    plot_predictions(predictions, future_dates)
    
    print("\nForecasting completed! Check the 'outputs' directory for results.")
    print("\nModel performance summary:")
    for product, metric in metrics.items():
        print(f"\nProduct {product}:")
        for name, value in metric.items():
            print(f"{name}: {value:.2f}")

if __name__ == "__main__":
    main()