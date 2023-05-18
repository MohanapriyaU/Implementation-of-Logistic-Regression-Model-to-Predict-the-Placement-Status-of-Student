# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohanapriya.U
RegisterNumber:  212220040091
*/
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

from sklearn.metrics import confusion_matrix confusion = confusion_matrix(y_test,y_pred) confusion
from sklearn.metrics import classification_report classification_report1 = classification_report(y_test,y_pred) classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])





## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
1.Placement data
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/9a920a78-6d75-4521-8629-b9c62f08ea28)
2.Salary data
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/db22ba7e-a5d2-4e64-8229-3cc577333096)
3.Checking the null() function
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/fa160757-8d20-4fc2-9207-a61d76a30b3d)
4.Data Duplicate
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/8bd85038-bf4f-4f84-9a19-29d05ac768de)
5.Print Data
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/f4b417d6-8cef-4e05-a4d1-7ea9b16297c0)
6.Data-Status
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/725998e5-0256-4a82-8e3d-aa7863123ec6)
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/df5e77f2-1795-4c69-940c-f5332ba91f08)
7.y_prediction array
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/33cc00a1-0d83-4227-b230-4eb8209df6a1)
8.Accuracy value
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/62320961-fff2-48dd-bc9b-91ccafdd0e46)
9.Confusion array
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/38bb64b0-3129-467d-b38e-5dadc29f7860)
10.Classification report
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/5f79146f-43c5-46b2-a4ea-7d0d22fbaf46)
11.Predictionn of LR
![image](https://github.com/MohanapriyaU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/116153626/d7dbe3da-5815-48d8-9b23-e63a28b49e44)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
