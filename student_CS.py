'''
steps - 
Load and explore the dataset.
Preprocess the data (clean, handle missing data).
Split the data into training and testing sets.
Train ML models (e.g., Linear Regression, Random Forest).
Evaluate models.
Make predictions on new or test data.
(Optional) Visualize results.
'''

# importing the required files 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
df = pd.read_csv('student_big.csv')

# 2. Preprocess Data - drop rows with missing values for simplicity
df = df.dropna()

# 3. Define features and target
X = df[['Math', 'Science', 'English']]
# y = df['Total']  
# use 'Average' for predicting average score
y = df['Average']

# 4. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train models
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate models
for model in [lr, rf]:
    y_pred = model.predict(X_test)
    print(f'Model: {model.__class__.__name__}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}')
    print(f'R2 Score: {r2_score(y_test, y_pred):.2f}\n')

# 7. Make prediction example
new_student1 = [[85, 90, 80]]  # example scores
# predicted_total = rf.predict(new_student1)
predicted_avg = rf.predict(new_student1)
# print(f'Predicted Total Score for new student1 - [85, 90, 80]: {predicted_total[0]:.2f}')
print(f'Predicted Average Score for new student1 - [85, 90, 80]: {predicted_avg[0]:.2f}')

new_student2 = [[50, 30, 60]]  # example scores
#predicted_total = rf.predict(new_student2)
predicted_avg = rf.predict(new_student2)
# print(f'Predicted Total Score for new student2 - [50, 30, 60]: {predicted_total[0]:.2f}')
print(f'Predicted Average Score for new student2 - [50, 30, 60]: {predicted_avg[0]:.2f}')