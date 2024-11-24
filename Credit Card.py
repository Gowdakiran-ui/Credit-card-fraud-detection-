# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df=pd.read_csv(r"C:\Users\Kiran gowda.A\Downloads\creditcard.csv\creditcard.csv")
df.head()

df.isnull().sum()

df.shape

df.info()

corellation=df.corr()

plt.figure(figsize=(20,10))
sns.heatmap(corellation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('corellation')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df,color='skyblue')
plt.title('TO DETECT OUTLIERS')
plt.xlabel('columns',fontsize=12)
plt.ylabel('values',fontsize=12)
plt.show()

from sklearn.preprocessing import StandardScaler

scalar=StandardScaler()
df_standardized=scalar.fit_transform(df)
df_standardized=pd.DataFrame(df_standardized,columns=df.columns)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


X = df.drop("Class", axis=1)
Y = df["Class"]  


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


# +
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input data from form
        user_input = request.form["input_data"]
        # Replace with your model prediction logic
        prediction = f"Prediction for {user_input}"
        return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

