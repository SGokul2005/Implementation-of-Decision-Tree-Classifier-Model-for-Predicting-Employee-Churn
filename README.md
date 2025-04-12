# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values. 5.From sklearn.model_selection import train_test_split. 6.Assign the train dataset and test dataset. 7.From sklearn.tree import 5. 5. DecisionTreeClassifier. 8.Use criteria as entropy. 9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHINESH R
RegisterNumber:  212223220019
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv("Employee.csv")
import pandas as pd

print(data.head())
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
print(dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/c5cf7f02-0bb3-4aa4-9026-9aa72337886c)
![image](https://github.com/user-attachments/assets/33d878ae-bd41-491e-a40c-3c241d10af9f)
![image](https://github.com/user-attachments/assets/5b15c883-34c4-4794-babd-fae148c1e3f9)
![image](https://github.com/user-attachments/assets/b89ff5b3-9d8c-42cf-b3d5-8a0d599796df)
![image](https://github.com/user-attachments/assets/92401da6-b992-42c1-8b29-19f650805f36)
![image](https://github.com/user-attachments/assets/531d83e3-b337-45fe-86f5-2b40be83854c)
![image](https://github.com/user-attachments/assets/10b43da0-7974-4631-9f6f-589431485723)
![image](https://github.com/user-attachments/assets/86267cc6-e573-4762-b3a9-13177e59d9f1)
![image](https://github.com/user-attachments/assets/703c80e0-8414-4b7c-808b-09d289a100ab)
![image](https://github.com/user-attachments/assets/aaa5099c-634d-4d63-b5f9-2a718d4b79ec)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
