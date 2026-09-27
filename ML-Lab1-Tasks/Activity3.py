import pandas as pd
import matplotlib.pyplot as plt

# Load online dataset
url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df = pd.read_csv(url)

# Extract Pakistan data
pak_data = df[df['Country/Region'] == 'Pakistan'].copy()
pak_data['Date'] = pd.to_datetime(pak_data['Date'])
pak_data = pak_data.sort_values('Date')

# Handle missing values
pak_data['Confirmed'] = pak_data['Confirmed'].ffill().fillna(0)

# Plot line chart
plt.figure(figsize=(10, 5))
plt.plot(pak_data['Date'], pak_data['Confirmed'], color='red')
plt.title('COVID-19 Confirmed Cases in Pakistan Over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.grid(True)
plt.show()

# Day with highest confirmed cases
max_row = pak_data.loc[pak_data['Confirmed'].idxmax()]
print(f"Day with highest confirmed cases: {max_row['Date'].date()} with {max_row['Confirmed']} cases.")

# Outlier detection using boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(pak_data['Confirmed'], vert=False)
plt.title('Boxplot of Confirmed Cases (Outlier Detection)')
plt.show()