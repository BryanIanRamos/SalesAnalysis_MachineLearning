import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
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

def train_models(df):
    """Train classification and regression models for product prediction"""
    feature_columns = [
        'day_of_week', 'is_weekend', 'month_of_year',
        'qty_last_7d_avg', 'qty_last_14d_avg', 'qty_last_30d_avg',
        'revenue_last_7d_avg', 'revenue_last_14d_avg', 'revenue_last_30d_avg',
        'transactions_last_7d_avg', 'transactions_last_14d_avg', 'transactions_last_30d_avg',
        'days_since_last_sale'
    ]
    
    # Prepare classification data (will product be sold or not)
    df['will_sell'] = (df['daily_quantity'] > 0).astype(int)
    
    # Split data
    X = df[feature_columns]
    y_class = df['will_sell']
    y_reg = df['daily_quantity']
    
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # Train classification model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_class_train)
    
    # Train regression model
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)
    
    # Print model performance
    print("\nClassification Model Performance:")
    y_class_pred = clf.predict(X_test)
    print(classification_report(y_class_test, y_class_pred))
    
    print("\nRegression Model Performance:")
    y_reg_pred = reg.predict(X_test)
    print(f"Mean Absolute Error: {mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
    
    return clf, reg, feature_columns

def predict_next_week(df, clf, reg, feature_columns):
    """Predict products and quantities for the next 7 days"""
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    predictions = []
    
    for stockcode in df['stockcode'].unique():
        product_data = df[df['stockcode'] == stockcode].sort_values('date').copy()
        
        for future_date in future_dates:
            # Prepare features for prediction
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
            
            # Create feature vector
            X_pred = pd.DataFrame([features])[feature_columns]
            
            # Make predictions
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

def plot_predictions(predictions_df):
    """Create visualizations for the predictions"""
    # Plot 1: Products predicted to sell each day
    plt.figure(figsize=(12, 6))
    daily_products = predictions_df[predictions_df['will_sell'] == 1].groupby('date').size()
    plt.bar(daily_products.index, daily_products.values)
    plt.title('Number of Products Predicted to Sell Each Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('product_forecast/daily_products_forecast.png')
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
    plt.savefig('product_forecast/top_products_forecast.png')
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
    plt.savefig('product_forecast/daily_product_line_forecast.png', bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    os.makedirs('product_forecast', exist_ok=True)
    
    # Get database connection
    print("Connecting to database...")
    engine = get_database_connection()
    if engine is None:
        return
    
    # Fetch and prepare data
    print("Fetching and preparing data...")
    df = fetch_product_data(engine)
    if df is None:
        return
    
    df = prepare_features(df)
    
    # Train models
    print("Training models...")
    clf, reg, feature_columns = train_models(df)
    
    # Make predictions
    print("Making predictions for next week...")
    predictions_df = predict_next_week(df, clf, reg, feature_columns)
    
    # Save predictions
    predictions_df.to_csv('product_forecast/product_predictions.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_predictions(predictions_df)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total products predicted to sell: {predictions_df['will_sell'].sum()}")
    print("\nTop 5 products by predicted weekly quantity:")
    top_5 = predictions_df.groupby(['stockcode', 'description'])['predicted_quantity'].sum().sort_values(ascending=False).head()
    print(top_5)
    
    print("\nPredictions completed! Check the 'product_forecast' directory for detailed results.")
    
    # After training models, add:
    print("Saving models...")
    model_data = {
        'classifier': clf,
        'regressor': reg,
        'feature_columns': feature_columns,
        'top_products': predictions_df.groupby('stockcode')['predicted_quantity'].sum().nlargest(5).index.tolist()
    }
    joblib.dump(model_data, 'product_forecast/product_prediction_models.pkl')

if __name__ == "__main__":
    main() 