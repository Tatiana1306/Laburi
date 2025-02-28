import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats

# Setare variabilă pentru a evita erori legate de procesare paralelă
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Funcție pentru dezarhivare sigură
def extract_zip(file_path, extract_to):
    if os.path.exists(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Eroare: Fișierul {file_path} nu există!")

# Setare căi corecte pentru Windows
base_path = os.path.dirname(os.path.abspath(__file__))
zip1 = os.path.join(base_path, 'bank.zip')
zip2 = os.path.join(base_path, 'bank-additional.zip')
extract_path = os.path.join(base_path, 'data')

# Creare director de extracție dacă nu există
os.makedirs(extract_path, exist_ok=True)

extract_zip(zip1, extract_path)
extract_zip(zip2, extract_path)

# Încărcarea datelor
csv_path = os.path.join(extract_path, 'bank-additional', 'bank-additional-full.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Fișierul {csv_path} nu a fost găsit!")

df = pd.read_csv(csv_path, sep=';')

# Verificare date
if df.empty:
    raise ValueError("Setul de date este gol!")

print(df.head())
print(df.info())

# Vizualizare statistici descriptive
print(df.describe())

# Tratarea valorilor lipsă
df = df.dropna()

# Detectarea și eliminarea outlierilor dacă există coloane numerice
if not df.select_dtypes(include=[np.number]).empty:
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]

# Standardizare date
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Reducerea dimensionalității folosind PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Aplicarea clustering-ului
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_pca)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_pca)

# Calcularea scorului Silhouette
if len(set(kmeans_labels)) > 1:
    kmeans_silhouette = silhouette_score(df_pca, kmeans_labels)
else:
    kmeans_silhouette = -1

if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(df_pca, dbscan_labels)
else:
    dbscan_silhouette = -1

print(f'Scor Silhouette K-Means: {kmeans_silhouette}')
print(f'Scor Silhouette DBSCAN: {dbscan_silhouette}')

# Vizualizarea clusterelor
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='viridis')
axes[0].set_title('K-Means Clustering')
axes[1].scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan_labels, cmap='plasma')
axes[1].set_title('DBSCAN Clustering')
plt.show()

