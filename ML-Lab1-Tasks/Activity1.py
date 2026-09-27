# Activity 1: Pakistani Provinces Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
data = pd.DataFrame({
    'Province': ['Punjab', 'Sindh', 'KPK', 'Balochistan'],
    'Population (millions)': [127.6, 55.6, 40.8, 14.8],
    'Literacy Rate (%)': [64, 63, None, 55], # Added missing value
    'Region': ['East', 'South', 'North', 'West']
})

# Handle missing values: Fill KPK literacy rate with median
median_literacy = data['Literacy Rate (%)'].median()
data['Literacy Rate (%)'] = data['Literacy Rate (%)'].fillna(median_literacy)

# Region Encoding: Label Encoding and One-Hot Encoding
data['Region_Label'] = data['Region'].astype('category').cat.codes
data_encoded = pd.get_dummies(data, columns=['Region'], prefix='Region')
print("Encoded DataFrame:\n", data_encoded)

# Visualize Population vs Literacy Rate
plt.figure(figsize=(8, 5))
plt.scatter(data['Population (millions)'], data['Literacy Rate (%)'], color='purple')
for i, txt in enumerate(data['Province']):
    plt.annotate(txt, (data['Population (millions)'][i], data['Literacy Rate (%)'][i]), fontsize=12)
plt.title('Population vs Literacy Rate of Pakistani Provinces')
plt.xlabel('Population (millions)')
plt.ylabel('Literacy Rate (%)')
plt.grid(True)
plt.show()

# Outlier Detection via IQR
Q1 = data['Literacy Rate (%)'].quantile(0.25)
Q3 = data['Literacy Rate (%)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['Literacy Rate (%)'] < lower_bound) | (data['Literacy Rate (%)'] > upper_bound)]
print("Outliers in Literacy Rate:\n", outliers)