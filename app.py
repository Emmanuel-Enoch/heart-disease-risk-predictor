import numpy as np
from pandas import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
heart_data=read_csv("D:\Theme Based Project\heart-disease-risk-predictor\heart1.csv")
heart_data.head()
# Create individual models
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
voting_model = VotingClassifier(
    estimators=[('lr', model1), ('rf', model2), ('gb', model3)],
    voting='hard' )
heart_data.tail()
#no of rows and columns in dataset
heart_data.shape
#getting some info about data
heart_data.info()
#checking for missing values
heart_data.isnull().sum()#statistical measure about the data
heart_data.describe()
#checking the distribution of target variable
heart_data['target'].value_counts()
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)
print(Y)
#splitting data into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
voting_model.fit(X_train,Y_train)
#tranning the logisticregression model
model=LogisticRegression()
model.fit(X_train,Y_train)#tranning the logisticregression model
#accuracy on testing data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('accuracy on test data:',test_data_accuracy)
# Accuracy on training data
train_preds = voting_model.predict(X_train)
train_accuracy = accuracy_score(train_preds,Y_train)# Accuracy on testing data
print("Training Accuracy:", train_accuracy)
accuracy_voting= train_accuracy
accuracy1 = accuracy_voting* 100
print("Accuracy1: {:.2f}%".format(accuracy1))
test_preds = voting_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_preds)
print("Testing Accuracy:", test_accuracy)
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('accuracy on training data:',training_data_accuracy)
accuracy_logistic=training_data_accuracy
accuracy2 = accuracy_logistic * 100
print("Accuracy: {:.2f}%".format(accuracy2))
#building a predictive system

input_data=input("enter age 	sex 	cp 	trestbps 	chol 	fbs 	restecg 	thalach 	exang 	oldpeak 	slope 	ca 	thal")
input_data = [float(x) for x in input_data.split(',')]
#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy arraya as we r predicting for only one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print('the person does not have heart disease')
else:
    print('the person has heart disease')
features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
# Create bar graph
plt.figure(figsize=(10,6))
plt.bar(features, input_data, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel("Feature Names")
plt.ylabel("Input Values")
plt.title("Heart Disease Prediction - Input Feature Values")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)



plt.figure(figsize=(8,5))
models = ['voting classifier','Logistic Regression']
accuracies = [99.51,86.31]

bars=plt.bar(models, accuracies, color=['lightblue', 'lightblue'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', ha='center', va='bottom')
plt.xlabel('ML Models')
plt.ylabel('Accuracy')
plt.title('   \n Comparison of ML Model Accuracies')
plt.ylim(0,150)
plt.show()