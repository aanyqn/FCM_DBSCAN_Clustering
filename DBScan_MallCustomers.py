# DBSCAN dengan scikit-learn - Mall Customers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('Mall_Customers.csv')
print(df.head()); print(df.isna().sum())

fitur = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[fitur].values
Xs = StandardScaler().fit_transform(X)

MIN_SAMPLES = 6
jarak, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs).kneighbors(Xs)
k_dist = np.sort(jarak[:, -1])
plt.figure(figsize=(6, 4))
plt.plot(k_dist); plt.axhline(0.38, color='r', ls='--', label='eps = 0.38')
plt.xlabel('Titik (terurut)'); plt.ylabel(f'Jarak ke tetangga ke-{MIN_SAMPLES}')
plt.title('k-distance plot'); plt.legend(); plt.tight_layout()
plt.savefig('Mall_Customers/DBScan_Kdistance_lib.png', dpi=150)

baris = []
for ms in (4, 5, 6, 8):
    for eps in np.arange(0.25, 0.50, 0.05):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
        m = lab != -1
        n_c = len(set(lab[m]))
        sil = silhouette_score(Xs[m], lab[m]) if n_c > 1 else np.nan
        baris.append([ms, round(eps, 2), n_c, (~m).sum(), sil])
print(pd.DataFrame(baris, columns=['min_samples', 'eps', 'n_cluster', 'noise', 'silhouette']).round(3))

EPS = 0.38
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(Xs)
labels = db.labels_
is_core = np.zeros(len(Xs), dtype=bool)
is_core[db.core_sample_indices_] = True

df['Cluster'] = labels
df['Tipe'] = np.where(labels == -1, 'noise', np.where(is_core, 'core', 'border'))
print(df['Tipe'].value_counts())

m = labels != -1
print('Silhouette:', round(silhouette_score(Xs[m], labels[m]), 3))
print('Davies-Bouldin:', round(davies_bouldin_score(Xs[m], labels[m]), 3))

print(df.groupby('Cluster')[['Age', fitur[0], fitur[1]]].mean().round(1))
print(df['Cluster'].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(7, 5.5))
for k in sorted(set(labels)):
    p = labels == k
    if k == -1:
        ax.scatter(X[p, 0], X[p, 1], c='k', marker='x', s=45, label='Noise')
    else:
        ax.scatter(X[p, 0], X[p, 1], s=40, label=f'Cluster {k}')
ax.set(title=f'DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})', xlabel=fitur[0], ylabel=fitur[1])
ax.legend(); plt.tight_layout(); plt.savefig('Mall_Customers/DBScan_lib.png', dpi=150)
df.to_csv('Mall_Customers/DBScan_lib.csv', index=False)