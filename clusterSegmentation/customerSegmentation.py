import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up the database connection using environment variables
db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(db_url)

# SQL Query to fetch the required data (same query as before)
query = """
SELECT
    c.customerid,
    SUM(s.totalprice) AS total_sales,
    COUNT(s.salesid) AS order_frequency
FROM
    customer c
JOIN
    sales s ON c.customerid = s.customerid
GROUP BY
    c.customerid
"""

# Load data from the database
df = pd.read_sql(query, engine)

# Inspect the first few rows of the data
df.head()

# Load the saved K-Means model and scaler
kmeans = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

# Preprocess the data (normalize the features using the saved scaler)
df_scaled = scaler.transform(df[['total_sales', 'order_frequency']])

# Apply the K-Means model to predict clusters for the new data
df['Cluster'] = kmeans.predict(df_scaled)

# Inspect the cluster assignments
print("Cluster assignments for new data:")
print(df[['customerid', 'total_sales', 'order_frequency', 'Cluster']].head())

# Assign meaningful cluster labels based on cluster centers
# Here, the mapping will be the same as before, but you can adjust it based on your cluster centers if needed.
cluster_centers = kmeans.cluster_centers_

if cluster_centers[0][0] > cluster_centers[1][0]:
    cluster_labels = {
        0: 'High-Value Customers',
        1: 'Low-Value Customers',
        2: 'Moderate-Value Customers'
    }
else:
    cluster_labels = {
        0: 'Low-Value Customers',
        1: 'High-Value Customers',
        2: 'Moderate-Value Customers'
    }

# Map cluster labels to the dataframe
df['Cluster Label'] = df['Cluster'].map(cluster_labels)

# Display the updated dataframe with cluster labels
print("\nData with Cluster Labels:")
print(df.head())

# Visualize the clustering result with color labels
scatter = plt.scatter(df['total_sales'], df['order_frequency'], 
                      c=df['Cluster'], cmap='viridis')

# Create a custom legend to show which color corresponds to which cluster
handles, _ = scatter.legend_elements()
legend_labels = [cluster_labels[i] for i in range(3)]
plt.legend(handles, legend_labels, title="Customer Segments")

# Label the axes and title
plt.xlabel('Total Sales')
plt.ylabel('Order Frequency')
plt.title('Customer Segmentation by K-Means Clustering')
plt.show()
