import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 

df = pd.read_csv(
    r"C:\Users\B01846982\OneDrive - University of the West of Scotland\Desktop\Medical_Insurance\Medical_insurance.csv"
)

print(df.isnull().sum())

# Convert categorical variables to numbers
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Encode region and SAVE the result back into df
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Rounding charges to 2 decimal places
df["charges"] = np.round(df["charges"], 2)

print(df.head())

# Correlation matrix
print(df.corr(numeric_only=True))

# Regression plot
sns.regplot(x="bmi", y="charges", data=df)

plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("BMI vs Charges")

plt.show()

# Box plot
sns.boxplot(x="smoker", y="charges", data=df)

plt.show()

# R² using smoker only
X = df[["smoker"]]
Y = df["charges"]

lm = LinearRegression()
lm.fit(X, Y)

print("R² using smoker:", lm.score(X, Y))

# R² using multiple variables
Z = df[
    [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest"
    ]
]

lm.fit(Z, Y)

print("R² using all variables:", lm.score(Z, Y))

# Creating a Pipeline
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
])
pipeline.fit(Z,Y)
ypipe=pipeline.predict(Z)
print("r2_score using all", r2_score(Y,ypipe))

#Training and Testing 
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)

#Initializing Ridge Regression
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))


#Polynomial Transformation
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))

