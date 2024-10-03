# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and perform any necessary preprocessing, such as handling missing values and encoding categorical variables.
2. Initialize the logistic regression model and train it using the training data.
3. Use the trained model to predict the placement status for the test set.
4. Evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MadhanKumar J
RegisterNumber: 2305001016 
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()

data1=data.copy()
data1.head()


data1=data1.drop(['sl_no','salary'],axis=1)
data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:, : -1]
x

y=data1.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred,x_test

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",cr)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:
![image](https://github.com/user-attachments/assets/87d075c5-f06c-4e1d-b792-30e6a6b76108)

![image](https://github.com/user-attachments/assets/d6791d3e-e50c-410f-a728-f33cec2b2572)

![image](https://github.com/user-attachments/assets/63a4bc20-0b1a-45c5-a7bd-4ddb995e9e9c)

![image](https://github.com/user-attachments/assets/e43ac8ba-bbf4-4fa5-bc96-de3a681f8a20)

![image](https://github.com/user-attachments/assets/99343339-9e38-49af-8c26-d2295c57607e)

![image](https://github.com/user-attachments/assets/65ae34df-c49e-4f08-ad2c-83727616f55a)

![image](https://github.com/user-attachments/assets/afa108a0-86e2-4a33-9304-9f3522fda0e3)

![image](https://github.com/user-attachments/assets/92c017c0-1f6c-4cbe-9a5b-97b273895bce)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
