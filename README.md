# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.

 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohanapriya U
RegisterNumber:212220040091  
*/
```
import pandas as pd data = pd.read_csv("Placement_Data.csv") data.head()

data1 = data.copy() data1 = data1.drop(["sl_no","salary"],axis = 1) data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder le = LabelEncoder() data1["gender"] = le.fit_transform(data1["gender"]) data1["ssc_b"] = le.fit_transform(data1["ssc_b"]) data1["hsc_b"] = le.fit_transform(data1["hsc_b"]) data1["hsc_s"] = le.fit_transform(data1["hsc_s"]) data1["degree_t"] = le.fit_transform(data1["degree_t"]) data1["workex"] = le.fit_transform(data1["workex"]) data1["specialisation"] = le.fit_transform(data1["specialisation"]) data1["status"] = le.fit_transform(data1["status"]) data1
x = data1.iloc[:,:-1] x

y = data1["status"] y

from sklearn.model_selection import train_test_split x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression lr = LogisticRegression(solver = "liblinear") lr.fit(x_train,y_train)

y_pred = lr.predict(x_test) y_pred

from sklearn.metrics import accuracy_score accuracy = accuracy_score(y_test,y_pred) accuracy
from sklearn.metrics import classification_report classification_report1 = classification_report(y_test,y_pred) classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])





## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
1.Placement data
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/1ec74bae-d0e4-4767-84e7-ea9b49f56e0c)
2.Salary data
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/d4214fab-c6ce-4b7e-a898-2a4d217fec77)
3.Checking the null() function
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/9d53e324-8426-433f-8884-d8ddf61faca5)
4.Data Duplicate
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/f3fbbb5d-b1df-433d-9199-9602efc28a50)
5.Print data
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/c06f6071-e4c0-49e7-b9f3-621291b7bf42)
6.Data status
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/e4759a41-446b-476a-bdde-621f63ba05cf)
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/a7b04dc5-d225-4be9-8b94-0734a6316df6)
7.y_prediction array
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/3b0ba225-c5fe-4470-bcae-74720bda755f)
8.Accuracy value
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/262c9328-aabf-4ef4-b497-6477fa12c3e4)
9.Confusion array
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/d6f0eecc-9a99-4758-b29a-939907f3323f)
10.Classification report
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/16a0dd18-2bfa-4192-b1ca-1ef98c370a43)
11.Prediction of LR
![image](https://github.com/MohanapriyaU76/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133958624/6eac5daf-93ed-4361-918e-ba46ae10acd1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
