import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the CSV file
df = pd.read_csv('ultrasonic_data.csv')

# Convert 'Time' column to datetime object with explicit format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# Extract hour and minute from 'Time' column
df['Hour'] = df['Time'].dt.hour
df['Minute'] = df['Time'].dt.minute

# Prepare data for clustering
X = df[['Hour', 'Minute', 'Distance']].values

# Perform k-means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 2], c=y_kmeans, s=50, cmap='viridis')
plt.xlabel('Hour')
plt.ylabel('Distance')
plt.title('Clustering of Ultrasonic Sensor Data')
plt.colorbar(label='Cluster')
plt.show()
