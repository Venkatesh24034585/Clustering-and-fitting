import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
df = pd.read_csv('/sales_data_sample.csv', encoding='latin1')

# Preview the dataset
print("Dataset Head:")
print(df.head())

# Data Preprocessing
df = df.dropna()

# Select features for clustering
X = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Plot for Order Data Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means clustering with optimal clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Scatter plot for Clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['QUANTITYORDERED'], df['SALES'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Order Data Clustering')
plt.xlabel('Quantity Ordered')
plt.ylabel('Sales')
plt.colorbar(label='Cluster')
plt.show()

# Linear regression to predict sales based on price each and quantity ordered
X_reg = df[['PRICEEACH', 'QUANTITYORDERED']]
y = df['SALES']

# Linear regression model
regressor = LinearRegression()
regressor.fit(X_reg, y)

# Add predictions to the dataset
df['FittedSales'] = regressor.predict(X_reg)

# Plot actual vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(df['SALES'], df['FittedSales'], color='blue', alpha=0.5, label='Actual vs Predicted')
plt.title('Linear Fitting: Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.legend()
plt.show()

# Correlation heatmap
correlation_matrix = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Order Data')
plt.show()
