from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd

data=load_diabetes()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target;
print(df.head())
X=df.drop("target",axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean squared error (MSE) : {mse}")
print(f"R^2 score : {r2}")
plt.figure(figsize=(8,2))
plt.scatter(y_test,y_pred,color='blue',edgecolors='k')
plt.xlabel("Actual diabetes data")
plt.ylabel("Predicted diabetes data")
plt.title("Actual vs Predicted diabetes data")
plt.grid(True)
plt.show()