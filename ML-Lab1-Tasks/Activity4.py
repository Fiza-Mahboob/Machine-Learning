import pandas as pd
import matplotlib.pyplot as plt

# Load files
df_csv = pd.read_csv('Data sets/students.csv')
df_json = pd.read_json('Data sets/attendance.json')
df_excel = pd.read_excel('Data sets/extra.xlsx')

# Merge DataFrames
merged_df = df_csv.merge(df_json, on='Name').merge(df_excel, on='Name')
print("Merged DataFrame:\n", merged_df)

# Scatter plot highlighting students with attendance < 70%
plt.figure(figsize=(8, 5))
colors = ['red' if att < 70 else 'blue' for att in merged_df['Attendance']]
plt.scatter(merged_df['Marks'], merged_df['Attendance'], c=colors, s=100)
plt.axvline(x=70, color='gray', linestyle='--')
plt.title('Marks vs Attendance')
plt.xlabel('Marks')
plt.ylabel('Attendance (%)')
plt.grid(True)
plt.show()

# One-hot encoding on Name
encoded_names = pd.get_dummies(merged_df['Name'], prefix='Name')
final_df = pd.concat([merged_df, encoded_names], axis=1)
print("One-Hot Encoded DataFrame:\n", final_df)