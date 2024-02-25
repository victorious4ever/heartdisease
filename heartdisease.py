# Import necessary libraries



import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
heart_disease2 = pd.read_csv(r'heart-disease.csv')
st.write("""
        ## Way to calculate risk of heart disease without going to the doctor. This data is based on the Heart disease cleveland UCI data set from kaggle https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
          Note: these results are not equivalent to a medical diagnosis!  
         """)

heart_disease1 = heart_disease2.drop('oldpeak', axis = 'columns')
heart_disease = heart_disease1.drop('slope', axis = 'columns')


age=st.number_input("Whats your age")



sex=st.sidebar.selectbox("Select your gender", ("Female", 
                             "Male"))
cp =st.sidebar.selectbox("How bad is your chest pain", ("Typical Angina", 
                             "Atypical Angina", "Non-Aginal Pain", "Asymptomatic"))

trestbps =st.number_input("What is your resting blood pressure")
chol = st.number_input("What is your cholestrol(mg/dl)")
fbs = st.sidebar.selectbox("Is your fasting blood pressure greater than 120(mg/ml)", ("No", 
                             "Yes" ))
restecg = st.sidebar.selectbox("How is your resting electrocardiographic measurement", ("Normal", 
                             "ST-T Wave Abnormality", "Left Ventricular Hypertrophy" ))
thalach = st.number_input("What is your maximum heart rate achieved")
exang = st.sidebar.selectbox("Have you had any chest pain from exercise",( "Yes", "No"))
ca = st.sidebar.selectbox("How many major blood vessels do you have", ( 0, 1,2,3))
thal = st.sidebar.selectbox("What level of thalassemia do you have", ("Normal", "Fixed defect", "Reversable defect"))


dataToPredic = pd.DataFrame({
   "age": [age],
   "sex": [sex],
   "cp": [cp],
   "trestbps": [trestbps],
   "chol": [chol],
   "fbs": [fbs],
   "restecg": [restecg],
   "thalach": [thalach],
   "exang": [exang],
   "ca": [ca],
   "thal": [thal],
 })


# Mapping the data as explained in the script above
dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)

dataToPredic.replace("Typical Angina", 0, inplace = True)
dataToPredic.replace("Atypical Angina", 1, inplace = True)
dataToPredic.replace("Non-anginal pain", 2, inplace = True)
dataToPredic.replace("Asymptomatic", 3, inplace = True)
dataToPredic.replace("No", 0, inplace = True)
dataToPredic.replace("Yes", 1, inplace = True)
dataToPredic.replace("Normal", 0, inplace = True)
dataToPredic.replace("ST-T Wave Abnormality", 1, inplace = True)
dataToPredic.replace("Left Ventricular Hypertrophy", 2, inplace = True)
dataToPredic.replace("Normal", 3, inplace = True)
dataToPredic.replace("Fixed defect", 6, inplace = True)
dataToPredic.replace("Reversable defect", 7, inplace = True)


X = heart_disease.drop("target", axis=1)


y = heart_disease["target"]



X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.25)
clf = RandomForestClassifier()
clf.fit(X = X_train, y = y_train)
y_preds = clf.predict(X=X_test)

train_acc = clf.score(X=X_train, y=y_train)

print(f"The model's accuracy on the training dataset is: {train_acc*100}%")
test_acc = clf.score(X=X_test, y=y_test)
print(f"The model's accuracy on the testing dataset is: {test_acc*100:.2f}%")






st.sidebar.title('Please, fill your informations to predict your heart condition')









# Load the previously saved machine learning model
Prob1 = clf.predict_proba(dataToPredic)
Prob2=round(Prob1[0][1] * 100, 2)

if(Prob2 < 0):
  Prob2 = 2

if st.button('PREDICT'):
 # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
  if(Prob2 < 50):
    st.write('You have a low chance of getting heart disease' )
  else:
    st.write('You have a high chance of getting heart disease')
