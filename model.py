import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

data = pd.read_csv('./Symptoms and COVID Presence.csv')
data = data.dropna()

predictors = data[[
    'Breathing_Problem','Fever','Dry_Cough','Sore_throat','Running_Nose','Asthma','Chronic_Lung_Disease','Headache',
    'Heart_Disease','Diabetes','Hyper_Tension','Fatigue','Gastrointestinal','Abroad_travel','Contact_with_COVID_Patient',
    'Attended_Large_Gathering','Visited_Public_Exposed_Places','Family_working_in_Public_Exposed_Places','Wearing_Masks',
    'Sanitization_from_Market'
]]

targets = data.COVID_19

predictors_labels = ['Breathing_Problem','Fever','Dry_Cough','Sore_throat','Running_Nose','Asthma','Chronic_Lung_Disease','Headache',
    'Heart_Disease','Diabetes','Hyper_Tension','Fatigue','Gastrointestinal','Abroad_travel','Contact_with_COVID_Patient',
    'Attended_Large_Gathering','Visited_Public_Exposed_Places','Family_working_in_Public_Exposed_Places','Wearing_Masks',
    'Sanitization_from_Market']

target_labels = ['True','False']

X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(predictors,targets)

arbol = DecisionTreeClassifier()
arbol.fit(X_entrenamiento, y_entrenamiento)

score = arbol.score(X_test,y_test)
print(score)

arbol.score(X_entrenamiento,y_entrenamiento)

test = arbol.predict([[False,False,False,True,True,True,True,True,True,True,True,True,True,False,True,False,True,True,True,True,]])
print(test)


