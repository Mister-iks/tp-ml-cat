import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
iris_data = pd.read_csv('iris.csv')

# Affichage des données
st.title("Analyse de classification Iris")
st.write("Ce projet utilise la régression logistique pour classifier les espèces d'iris.")

st.subheader("Données")
st.write(iris_data.head())

# Statistiques descriptives
st.subheader("Statistiques descriptives")
st.write(iris_data.describe())

# Visualisation des données
st.subheader("Visualisation des données")
fig, ax = plt.subplots()
sns.pairplot(iris_data, hue='Species')
plt.show()

# Préparation des données
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions et évaluation
st.subheader("Évaluation du modèle")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

st.write(f"Exactitude du modèle : {accuracy:.2f}")
st.write("Rapport de classification :")
st.write(report)
