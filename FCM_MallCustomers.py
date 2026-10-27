import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('Mall_Customers.csv')
print(df.head())

fitur = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[fitur].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

M = 2.0
hasil = []
for c in range(2, 9):
    cntr, u, u0, d, jm, n_iter, fpc = fuzz.cluster.cmeans(
        Xs.T, c=c, m=M, error=1e-5, maxiter=150, seed=42)
    lab = u.argmax(axis=0)
    pe = -np.sum(u * np.log(np.fmax(u, 1e-12))) / u.shape[1]
    hasil.append([c, n_iter, fpc, pe,
                  silhouette_score(Xs, lab), davies_bouldin_score(Xs, lab)])
tabel = pd.DataFrame(hasil, columns=['c', 'iter', 'FPC', 'PE', 'Silhouette', 'DBI'])
print(tabel.round(3))

C = 5
cntr, u, u0, d, jm, n_iter, fpc = fuzz.cluster.cmeans(
    Xs.T, c=C, m=M, error=1e-5, maxiter=150, seed=42)
print(f'Iterasi: {n_iter} | FPC: {fpc:.3f} | J akhir: {jm[-1]:.3f}')
 
df['Cluster'] = u.argmax(axis=0)
df['MaxMembership'] = u.max(axis=0)
for k in range(C):
    df[f'U_C{k}'] = u[k]
    
print(df.groupby('Cluster')[['Age', fitur[0], fitur[1]]].mean().round(1))
print(df['Cluster'].value_counts().sort_index())
print('Data ambigu (keanggotaan maks < 0.6):', (df['MaxMembership'] < 0.6).sum())

baru = np.array([[70, 80], [30, 20]])
u_baru, *_ = fuzz.cluster.cmeans_predict(
    scaler.transform(baru).T, cntr, m=M, error=1e-5, maxiter=150)
print(pd.DataFrame(u_baru.T, columns=[f'C{k}' for k in range(C)]).round(3))

pusat = scaler.inverse_transform(cntr)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(X[:, 0], X[:, 1], c=df['Cluster'], cmap='tab10', s=40)
ax[0].scatter(pusat[:, 0], pusat[:, 1], c='k', marker='X', s=200, label='Centroid')
ax[0].set(title='Hasil FCM scikit-fuzzy (c=5)', xlabel=fitur[0], ylabel=fitur[1])
ax[0].legend()
ax[1].plot(range(1, len(jm) + 1), jm, marker='o')
ax[1].set(title='Konvergensi fungsi objektif', xlabel='Iterasi', ylabel='J')
plt.tight_layout(); plt.savefig('Mall_Customers/FCM_skfuzzy.png', dpi=150)
 
df.to_csv('Mall_Customers/FCM_skfuzzy.csv', index=False)