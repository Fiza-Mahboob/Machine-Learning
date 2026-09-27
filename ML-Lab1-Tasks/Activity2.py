import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
data = pd.DataFrame({
    'ID': range(1, n + 1),
    'Math': np.random.randint(40, 100, size=n),
    'Science': np.random.randint(40, 100, size=n),
    'English': np.random.randint(40, 100, size=n),
    'Grade': np.random.choice(['A', 'B', 'C'], size=n)
})

# Introduce and handle missing marks using mean
data.loc[10:15, 'Math'] = np.nan
data['Math'] = data['Math'].fillna(data['Math'].mean())

# Encode Grade category manually or via mapping
grade_mapping = {'C': 0, 'B': 1, 'A': 2}
data['Grade_Encoded'] = data['Grade'].map(grade_mapping)

# Histogram of Math scores with 10 bins
plt.figure(figsize=(7, 4))
plt.hist(data['Math'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Math Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Total Score Boxplot for Outliers
data['Total_Score'] = data['Math'] + data['Science'] + data['English']
plt.figure(figsize=(6, 4))
plt.boxplot(data['Total_Score'], vert=False)
plt.title('Boxplot of Total Scores for Outlier Detection')
plt.xlabel('Total Score')
plt.show()

print("Summary: Grade group with highest average scores can be found via groupby:")
print(data.groupby('Grade')['Total_Score'].mean())