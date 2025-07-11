# 🩺 Diabetes Prediction using Linear Regression

This project demonstrates how to build a simple linear regression model to predict diabetes progression using the popular `load_diabetes` dataset from `scikit-learn`.

## 📊 Dataset

from sklearn.datasets import load_diabetes
It includes 10 baseline variables (age, sex, BMI, blood pressure, and 6 blood serum measurements) and a quantitative measure of disease progression one year after baseline.

📦 Libraries Used
pandas

matplotlib

scikit-learn

🚀 How It Works
Load Dataset
The diabetes dataset is loaded and converted into a DataFrame.

Train-Test Split
The data is split into training and testing sets using train_test_split.

Model Training
A LinearRegression model is trained on the training data.

Prediction & Evaluation
Predictions are made on the test data and evaluated using:

Mean Squared Error (MSE)

R² Score

Visualization
A scatter plot compares actual vs predicted values.

🧪 Output Example

Mean squared error (MSE) : 2900.193628493475
R^2 score : 0.451
