import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('marketing_campaign.csv', sep='\t')
 
df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'])

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
tgl_acuan = df['Dt_Customer'].max()
mnt = [c for c in df.columns if c.startswith('Mnt')]
df['Total_Spending'] = df[mnt].sum(axis=1)
df['Age'] = tgl_acuan.year - df['Year_Birth']
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
df['Tenure_Days'] = (tgl_acuan - df['Dt_Customer']).dt.days

df = df.dropna(subset=['Income'])
df = df[(df['Income'] < 200000) & (df['Age'] < 100)]
df = df.reset_index(drop=True)
 
fitur = ['Income', 'Total_Spending']
X = df[fitur].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print(df.shape, Xs.shape)

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

C = 3
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

baru = np.array([
    df[fitur].quantile(0.90).values,
    df[fitur].quantile(0.10).values,
])

u_baru, *_ = fuzz.cluster.cmeans_predict(
    scaler.transform(baru).T, cntr, m=M, error=1e-5, maxiter=150)
print(pd.DataFrame(u_baru.T, columns=[f'C{k}' for k in range(C)]).round(3))

pusat = scaler.inverse_transform(cntr)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.get_cmap('tab10', C)
for k in range(C):
    anggota = df['Cluster'] == k
    ax[0].scatter(X[anggota, 0], X[anggota, 1], color=cmap(k), s=40,
                  label=f'Cluster {k}')
ax[0].scatter(pusat[:, 0], pusat[:, 1], c='k', marker='X', s=200, label='Centroid')
ax[0].set(title='Hasil FCM scikit-fuzzy (c=3)', xlabel=fitur[0], ylabel=fitur[1])
ax[0].legend()
ax[1].plot(range(1, len(jm) + 1), jm, marker='o')
ax[1].set(title='Konvergensi fungsi objektif', xlabel='Iterasi', ylabel='J')
plt.tight_layout(); plt.savefig('Marketing_Campaign/FCM_skfuzzy.png', dpi=150)
# plt.tight_layout(); plt.show()

df.to_csv('Marketing_Campaign/FCM_skfuzzy.csv', index=False)