# 1. Manual Data Creation
import pandas as pd
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['Lahore', 'Faisalabad', 'Karachi', 'Multan'],
    'Salary': [50000, 60000, 70000, 55000]
})
print(data)


# 2. Data Loading Task by CSV File
import pandas as pd
# Local CSV
df_csv = pd.read_csv('Data sets/Iris.csv')
print("CSV Data:\n", df_csv.head())
# Online CSV
import pandas as pd
url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df_online = pd.read_csv(url)
print("Online CSV Data:\n", df_online.head())


# 3. Loading from a JSON File
import pandas as pd
# Load data from a JSON file
df_json = pd.read_json('Data sets/attendance.json')
print("JSON File Data:\n", df_json.head())


# 4. Loading from an Excel File
import pandas as pd
# Load data from an Excel file 
df_excel = pd.read_excel('Data sets/extra.xlsx', sheet_name='Sheet1')
print("Excel File Data:\n", df_excel.head())


# 5. Data Loading Task by Text File
import pandas as pd
df_txt = pd.read_csv('Data sets/data.txt', delimiter='\t')
print("Text File Data:\n", df_txt)


# 6. Data Generation Task with NumPy
import numpy as np
import pandas as pd
np.random.seed(42)
data_np = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Score': [88, 78, 64, 92, 57],
    'Height': [151.2, 156.3, 176.4, 160.9, 174.8],
    'Weight': [64.9, 73.7, 61.4, 70.6, 75.0]
})
print("NumPy Generated Data:\n", data_np)


# 7. Data Loading Task using Sklearn (Iris)
from sklearn.datasets import load_iris
iris = load_iris()
print("Keys:", iris.keys())
print("Target Names:", iris["target_names"])
print("Features (first 5):\n", iris.data[:5])
print("Target values (first 5):", iris.target[:5])


# 8. Handling Missing Values
import pandas as pd
import numpy as np
# Original DataFrame with missing values (NaN)
data = pd.DataFrame({
    'A': [1.0, 2.0, np.nan, 4.0, 5.0],
    'B': [6.0, np.nan, 8.0, 9.0, 10.0],
    'C': [11.0, 12.0, 13.0, np.nan, 15.0]
})
print("Original DataFrame:\n", data)
# Option A: Remove rows containing missing values
df_dropped_rows = data.dropna()
print("\nDataFrame after dropping rows with missing values:\n", df_dropped_rows)
# Option B: Remove columns containing missing values
df_dropped_cols = data.dropna(axis=1)
print("\nDataFrame after dropping columns with missing values:\n", df_dropped_cols)
# Option C: Fill missing values with the median
df_filled = data.fillna(data.median())
print("\nDataFrame after filling missing values with the median:\n", df_filled)


### 9. Handling Categorical Values (Ordinal & One-Hot Encoding)
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
# Sample data containing categorical text columns
data = pd.DataFrame({
    'Education': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors']
})
# Manually defining the order for Ordinal Encoding
order = [['High School', 'Bachelors', 'Masters', 'PhD']]
oe = OrdinalEncoder(categories=order)
data['Education_Encoded'] = oe.fit_transform(data[['Education']])
print("Ordinal Encoded Data:\n", data)
# On-Hot Encoding using pandas get_dummies
df_categorical = pd.DataFrame({'Method': ['Method2', 'Method1', 'Method3']})
df_onehot = pd.get_dummies(df_categorical, prefix='Method')
print("\nOne-Hot Encoded DataFrame:\n", df_onehot)


# 10.Box Plot 
import matplotlib.pyplot as plt
import numpy as np
# Sample data with an outlier
data = [10, 12, 11, 13, 12, 14, 13, 100]  # 100 is an outlier[cite: 3]
# Plot boxplot
plt.boxplot(data)
plt.title("Outlier Visualization")
plt.show()

# 11. Data Preprocessing Task for Handling Outliers (IQR)
import pandas as pd
data = pd.DataFrame({'Income': [25000, 27000, 26000, 28000, 500000, 24000]})
Q1 = data['Income'].quantile(0.25)
Q3 = data['Income'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
data_cleaned = data[(data['Income'] >= lower_limit) & (data['Income'] <= upper_limit)]
print(data_cleaned)

# 12. Data Preprocessing Task for Log Transformation
import numpy as np
import pandas as pd
data = pd.DataFrame({'Income': [25000, 27000, 26000, 28000, 500000, 24000]})
data['Income_Log'] = np.log(data['Income'])
print(data)

# 13. Data Preprocessing Task for Feature Scaling
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
data = pd.DataFrame({'Age': [22, 25, 47, 35, 60], 'Income': [25000, 32000, 95000, 60000, 150000]})
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)

# 14. Data Preprocessing Task for Feature Engineering and PCA
import pandas as pd
from sklearn.decomposition import PCA
data = pd.DataFrame({
    'Height_cm': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90],
    'Date': ['2026-01-05', '2026-02-10', '2026-03-15', '2026-04-01', '2026-05-20']
})
# Feature Engineering: BMI
data['BMI'] = data['Weight'] / ((data['Height_cm'] / 100) ** 2)
# PCA Dimensionality Reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data[['Height_cm', 'Weight', 'BMI']])
print("Reduced Data:\n", reduced_data)

# 15. Data Preprocessing Task for Handling Imbalanced Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
X = pd.DataFrame({'Feature1': np.random.rand(1000), 'Feature2': np.random.rand(1000)})
y = pd.Series([0]*900 + [1]*100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# 16 .Line and Scatter Plots (Example 2.1)
import matplotlib.pyplot as plt
year = [1994, 1995, 1998, 2000]  # data points
population = [2.59, 3.69, 5.33, 6.77]  # data points
plt.plot(year, population)  # plots the data on specified coordinates in background
plt.show()  # visualises the graph
plt.scatter(year, population)
plt.show()


# 17. Legends, Titles & Labels (Example 2.2)
import matplotlib.pyplot as plt
year = [1994, 1995, 1998, 2000]  # data points
population = [2.59, 3.69, 5.33, 6.77]  # data points
plt.plot(year, population, label='population 1')  # plots the data on specified coordinates in background
pop2 = [4.44, 3.22, 5.55, 6.88]
plt.plot(year, pop2, label='population 2')  # plots the data and assign label to it
plt.xlabel('Independent var')  # specifies xlabel
plt.ylabel('Dependent var')  # specifies ylabel
plt.title('Interesting Graph')  # specifies title
plt.legend()  # visualises labels specified to the data
plt.show()


# 18. Grid, Line Styling & Fill Between (Example 2.3)
import matplotlib.pyplot as plt
year = [1994, 1995, 1998, 2000]  # data points
population = [2.59, 3.69, 5.33, 6.77]  # data points
plt.plot(year, population, 'r', label='Population 1', linewidth=5)  # plots the data on specified coordinates in background
pop2 = [4.44, 3.22, 5.55, 6.88]
plt.plot(year, pop2, 'c', label='Population 2', linewidth=5)  # plots the data and assign label to it
plt.xlabel('Independent var')  # specifies xlabel
plt.ylabel('Dependent var')  # specifies ylabel
plt.title('Interesting Graph')  # specifies title
plt.legend()  # visualises labels specified to the data
plt.grid(True, color='k')
plt.fill_between(year, population, 0, color='green')  # fills the specified color below the data points
plt.show()


# 19. Histograms (Example 2.4)
import matplotlib.pyplot as plt
# help(plt.hist)
# list with 12 values
values = [1.2, 1.3, 2.2, 3.3, 2.4, 6.5, 6.6, 7.7, 8.8, 9.9, 4.2, 5.3]
plt.hist(values, bins=3)
plt.show()


# 20. Olivetti Faces Dataset Exploration & Shape Inspection
from sklearn.datasets import fetch_olivetti_faces
# fetch the faces data
faces = fetch_olivetti_faces()
faces.keys()
n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)


# 21. Visualizing a Single Olivetti Face Image
X, y = faces.data, faces.target
import matplotlib.pyplot as plt
# Show one image (5th index)
plt.imshow(faces.images[5], cmap="gray")
plt.title(f"Face ID: {faces.target[5]}")
plt.axis("off")  # hide axis
plt.show()


# 22. Visualizing a Grid of First 16 Olivetti Face Images
# Show first 16 images
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces.images[i], cmap="gray")
    plt.axis("off")  # hide axes
    plt.title(faces.target[i])
plt.show()


# 23. Detailed Iris Dataset Exploration
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.keys())
print(iris["target_names"])
n_samples, n_features = iris.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
print('Feature Name:', iris.feature_names)
print('Target Name:', iris.target_names)
print('Dimension of Input', iris.data.shape)
print('Dimension of Output', iris.target.shape)
print('First 5 rows of Input:', iris.data[1:5])
print('Target Data:', iris.target)
