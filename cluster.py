import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the saved K-Means model
kmeans = joblib.load('customer_segmentation_model.pkl')

# Load the scaler used for normalization (assuming you saved it too)
scaler = joblib.load('scaler.pkl')  # If you saved the scaler separately during training

# Function to preprocess new customer data
def preprocess_customer_data(new_data):
    # Preprocess the new customer data (same steps as during training)
    # new_data should be a DataFrame with 'Sales' and 'Order ID' columns
    customer_data = new_data.groupby('Customer ID').agg({
        'Sales': 'sum',   # Total sales for each customer
        'Order ID': 'count'  # Order frequency (number of orders)
    }).rename(columns={'Sales': 'Total Sales', 'Order ID': 'Order Frequency'})

    # Normalize the new customer data
    customer_data_scaled = scaler.transform(customer_data)
    
    return customer_data_scaled

# Example of new customer data (replace with real data from your application)
new_customer_data = pd.DataFrame({
    'Customer ID': [1, 2, 3],
    'Sales': [500, 150, 200],
    'Order ID': [5, 2, 3]
})

# Preprocess the new data
new_data_scaled = preprocess_customer_data(new_customer_data)

# Use the model to predict the cluster for the new customer data
new_customer_data['Cluster'] = kmeans.predict(new_data_scaled)

# Map the clusters to human-readable labels (same labels as during training)
cluster_labels = {
    0: 'High-Value Customers',     
    1: 'Low-Value Customers',      
    2: 'Moderate-Value Customers'
}

# Add the cluster label to the new data
new_customer_data['Cluster Label'] = new_customer_data['Cluster'].map(cluster_labels)

# Display the customer data with the predicted cluster label
print(new_customer_data)

# Visualization of the new data points
plt.scatter(new_customer_data['Sales'], new_customer_data['Order ID'], 
            c=new_customer_data['Cluster'], cmap='viridis', label='New Customers')

# Plot the cluster centers (centroids)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Cluster Centers')

# Add labels and title to the plot
plt.xlabel('Total Sales')
plt.ylabel('Order Frequency')
plt.title('Customer Segmentation for New Data')
plt.legend()
plt.show()
