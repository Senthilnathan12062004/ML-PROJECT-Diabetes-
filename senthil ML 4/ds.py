import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
df=pd.read_csv("daibb.csv")
print(df.head())

df.info()
df.describe().T
cat_col=[col for col in df.columns 
         if df[col].dtype=='object']
print("Categorical columns:",cat_col)
df[cat_col].nunique().T

print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

sns.histplot(df['Feature1'], kde=True)
plt.title("Diabetes")
plt.show()

sns.countplot(x='Feature1', data=df)
plt.show()

sns.histplot(df['Feature2'], kde=True)
plt.title("Diabetes")
plt.show()

sns.countplot(x='Feature2', data=df)
plt.show()

sns.scatterplot(x='Feature1', y='Feature2', data=df)
plt.show()

sns.scatterplot(x='Feature2', y='Feature1', data=df)
plt.show()

sns.scatterplot(x='Feature1', y='Outcome', data=df)
plt.show()

plt.figure(figsize=(9,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

sns.histplot(df['Feature1'], kde=True, bins=30)
plt.title("Diabetes Result")
plt.xlabel("Feature1")
plt.ylabel("Outcome")
plt.show()

sns.histplot(df['Feature2'], kde=True, bins=30)
plt.title("Diabetes Result")
plt.xlabel("Feature2")
plt.ylabel("Outcome")
plt.show()

sns.scatterplot(x='Feature1', y='Outcome', data=df)
plt.title("Feature1 vs Outcome")
plt.show()

sns.scatterplot(x='Feature2', y='Outcome', data=df)
plt.title("Feature2 vs Outcome")
plt.show()
sns.scatterplot(x='Feature1', y='Feature2', data=df)
plt.title("Feature1 vs Feature2")
plt.show()

sns.scatterplot(x='Feature2', y='Feature1', data=df)
plt.title("Feature2 vs Feature1")
plt.show()

sns.boxplot(x='Feature1', y='Outcome', data=df)
plt.title("Feature1 vs Outcome")
plt.show()

sns.boxplot(x='Feature2', y='Outcome', data=df)
plt.title("Feature2 vs Outcome")
plt.show()

sns.boxplot(x='Feature1', y='Feature2', data=df)
plt.title("Feature1 vs Feature2")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[[ 'Feature1','Feature2','Outcome' ]])
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("daibb.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("daibb.csv")

# Define Features (X) and Target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Accuracy:",accuracy)
print("\nConfusion Matrix:\n",cm)
print("\nClassification Report:\n", report)


# Confusion Matrix Plot
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
print(report)

print("Accuracy:", accuracy_score(y_test, y_pred))

#Classification Report (Table Format)
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
print("\nClassification Report Table\n")
print(report)