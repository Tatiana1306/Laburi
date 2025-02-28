import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# Încărcarea dataseturilor
data1 = pd.read_csv("dataset1.csv")
data2 = pd.read_csv("dataset2.csv")

# Afișarea numelor coloanelor pentru verificare
print("Coloanele dataset1:", data1.columns)
print("Coloanele dataset2:", data2.columns)

# Combinarea dataseturilor pe baza coloanei comune ('instant')
data = pd.merge(data1, data2, on='instant', how='inner')
# Vizualizarea informațiilor despre date
print(data.info())
print(data.describe())

# Vizualizarea statisticilor descriptive
sns.pairplot(data)
plt.show()

# Tratarea valorilor lipsă
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# Detectarea și eliminarea outlierilor
z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
data = data[(np.abs(z_scores) < 3).all(axis=1)]
# Definirea variabilelor independente și dependente
X = data.drop(columns=['cnt'])  # 'cnt' este variabila dependentă
y = data['cnt']

# Împărțirea setului de date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelul de regresie liniară
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)

# Modelul k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_mse = mean_squared_error(y_test, y_pred_knn)
model_performance = {
    "Linear Regression": lin_mse,
    "KNN": knn_mse
}

st.title("Compararea modelelor de regresie")
st.write("### MSE pentru fiecare model:")
for model_name, model_mse in model_performance.items():
    st.write(f'**{model_name}**: {model_mse}')

# Selectarea unui model și realizarea predicțiilor
option = st.selectbox("Selectați modelul", ["Linear Regression", "KNN"])
prediction = lin_reg.predict(X_test) if option == "Linear Regression" else knn.predict(X_test)
st.write("### Predicțiile modelului ales:")
st.write(prediction)