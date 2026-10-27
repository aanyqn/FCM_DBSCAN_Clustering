# DBSCAN dengan scikit-learn - Marketing Campaign
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_PATH = 'marketing_campaign.csv'
OUT_DIR = 'Marketing_Campaign'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep='\t')
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
Xs = StandardScaler().fit_transform(X)
N = len(Xs)
print('Data:', df.shape, '| Fitur ternormalisasi:', Xs.shape)

MIN_SAMPLES = 6
jarak, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs).kneighbors(Xs)
k_dist = np.sort(jarak[:, -1])
persen = {p: np.percentile(k_dist, p) for p in (50, 90, 95, 99)}
print('k-distance per persentil:', {p: round(v, 3) for p, v in persen.items()})

MIN_CLUSTER_SHARE = 0.05      # cluster ke-2 minimal 5% data
MAX_NOISE_SHARE = 0.20        # noise maksimal 20% data

baris = []
for ms in (4, 5, 6, 8, 10):
    for eps in np.arange(0.05, 0.31, 0.01):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
        m = lab != -1
        ukuran = pd.Series(lab[m]).value_counts().tolist()
        n_c = len(ukuran)
        sil = silhouette_score(Xs[m], lab[m]) if n_c > 1 else np.nan
        baris.append([ms, round(eps, 2), n_c, int((~m).sum()),
                      ukuran[0] if n_c > 0 else 0,
                      ukuran[1] if n_c > 1 else 0, sil])
grid = pd.DataFrame(baris, columns=['min_samples', 'eps', 'n_cluster', 'noise',
                                    'cluster_1', 'cluster_2', 'silhouette'])
print(grid[grid['n_cluster'] >= 2].round(3).to_string(index=False))

layak = grid[(grid['n_cluster'] >= 2) &
             (grid['cluster_2'] >= MIN_CLUSTER_SHARE * N) &
             (grid['noise'] <= MAX_NOISE_SHARE * N)]

if len(layak) > 0:
    MODE = 'segmentasi'
    terbaik = layak.sort_values('silhouette', ascending=False).iloc[0]
    MIN_SAMPLES, EPS = int(terbaik['min_samples']), float(terbaik['eps'])
    print(f'\n[Segmentasi] eps={EPS}, min_samples={MIN_SAMPLES}')
else:
    MODE = 'anomali'
    EPS = round(float(persen[95]), 3)             # eps = persentil 95 k-distance
    print('\n[PERINGATAN] Tidak ada konfigurasi dengan >= 2 cluster bermakna '
          f'(cluster ke-2 >= {MIN_CLUSTER_SHARE:.0%} data, noise <= {MAX_NOISE_SHARE:.0%}).')
    print('DBSCAN dipakai untuk deteksi pelanggan tidak biasa (noise), '
          f'bukan segmentasi. eps={EPS}, min_samples={MIN_SAMPLES}')

plt.figure(figsize=(6, 4))
plt.plot(k_dist)
plt.axhline(EPS, color='r', ls='--', label=f'eps = {EPS}')
plt.xlabel('Titik (terurut)'); plt.ylabel(f'Jarak ke tetangga ke-{MIN_SAMPLES}')
plt.title('k-distance plot'); plt.legend(); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/DBScan_Kdistance_lib.png', dpi=150); plt.close()

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(Xs)
labels = db.labels_
is_core = np.zeros(N, dtype=bool)
is_core[db.core_sample_indices_] = True

df['Cluster'] = labels
df['Tipe'] = np.where(labels == -1, 'noise', np.where(is_core, 'core', 'border'))
print(df['Tipe'].value_counts())

m = labels != -1
n_c = len(set(labels[m]))
if n_c > 1:
    print('Silhouette:', round(silhouette_score(Xs[m], labels[m]), 3))
    print('Davies-Bouldin:', round(davies_bouldin_score(Xs[m], labels[m]), 3))
else:
    print(f'Hanya {n_c} cluster, Silhouette dan Davies-Bouldin tidak dapat dihitung.')

ringkas = df.groupby('Cluster')[['Age', fitur[0], fitur[1]]].mean().round(1)
ringkas['Jumlah'] = df['Cluster'].value_counts().sort_index()
print(ringkas)

if MODE == 'anomali':
    df['Noise'] = np.where(labels == -1, 'noise', 'normal')
    print(df.groupby('Noise')[['Age', fitur[0], fitur[1]]].agg(['mean', 'max']).round(0))

fig, ax = plt.subplots(figsize=(7, 5.5))
for k in sorted(set(labels)):
    p = labels == k
    if k == -1:
        ax.scatter(X[p, 0], X[p, 1], c='k', marker='x', s=30, label=f'Noise ({p.sum()})')
    else:
        ax.scatter(X[p, 0], X[p, 1], s=15, alpha=0.6, label=f'Cluster {k} ({p.sum()})')
ax.set(title=f'DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}, mode={MODE})',
       xlabel=fitur[0], ylabel=fitur[1])
ax.legend(); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/DBScan_lib.png', dpi=150); plt.close()

df.to_csv(f'{OUT_DIR}/DBScan_lib.csv', index=False)