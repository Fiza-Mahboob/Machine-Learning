import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Encode categorical Department column using One-Hot Encoding
df = pd.get_dummies(df, columns=['Department'])

# Detect outliers in Salary using Boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df['Salary'], vert=False)
plt.title('Salary Outlier Detection')
plt.xlabel('Salary')
plt.show()

print("""
Comment on Zara's Salary: 
Zara's salary (2,000,000) is drastically higher than the rest of the dataset, 
appearing as a clear outlier via boxplot analysis. Depending on business context, 
it could be a data entry error (typo) requiring correction, or a genuine high executive salary. 
If it is an error, it should be cleaned/imputed; if true, models sensitive to outliers (like linear models) 
might require robust scaling or log transformation.
""")