# Dokumentasi Clustering: Fuzzy C-Means dan DBSCAN

Dokumen ini menjelaskan proses pengerjaan, rumus yang dipakai, dan arti setiap variabel pada kode untuk dua algoritma clustering: **Fuzzy C-Means (FCM)** dan **DBSCAN**. Dua dataset dipakai:

- `Mall_Customers.csv` (200 pelanggan) sebagai contoh utama pada Bagian 1 sampai 6.
- `marketing_campaign.csv` (2.240 baris) sebagai studi kasus pergantian dataset pada Bagian 7 sampai 9.

**Pembaruan pada versi ini** (dibanding versi yang hanya membahas Mall Customers):

| Bagian | Isi baru |
|---|---|
| 3.6.1 | Cara membaca loop pemilihan `c`: arti `pe` baris demi baris, isi `hasil.append(...)`, dan catatan batas iterasi |
| 4.7.1 | Penjelasan k-distance: definisi, cara membaca kurva, dan arti tiap baris kodenya |
| 7 | Checklist penyesuaian saat mengganti dataset |
| 8 | Studi kasus `marketing_campaign.csv`: persiapan data, hasil FCM, hasil DBSCAN, dan mode anomali |
| 9 | Kesalahan umum yang ditemukan saat kode ditinjau, beserta perbaikannya |
| 10 dan 11 | Daftar file dan catatan keterbatasan diperbarui |

Versi diperbarui: 27 Oktober 2026.

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Dataset dan Persiapan Data](#2-dataset-dan-persiapan-data)
3. [Fuzzy C-Means (FCM)](#3-fuzzy-c-means-fcm)
4. [DBSCAN](#4-dbscan)
5. [Metrik Evaluasi](#5-metrik-evaluasi)
6. [Perbandingan FCM dan DBSCAN](#6-perbandingan-fcm-dan-dbscan)
7. [Mengganti Dataset: Checklist Penyesuaian](#7-mengganti-dataset-checklist-penyesuaian)
8. [Studi Kasus: Marketing Campaign](#8-studi-kasus-marketing-campaign)
9. [Kesalahan Umum dan Perbaikannya](#9-kesalahan-umum-dan-perbaikannya)
10. [Daftar File dan Cara Menjalankan](#10-daftar-file-dan-cara-menjalankan)
11. [Catatan dan Keterbatasan](#11-catatan-dan-keterbatasan)

---

## 1. Gambaran Umum

Tujuannya adalah mengelompokkan pelanggan berdasarkan **pendapatan tahunan** dan **belanja**, lalu menafsirkan tiap kelompok sebagai segmen pelanggan. Dataset Mall Customers (200 pelanggan, skor belanja 1 sampai 100) menjadi contoh utama. Dataset Marketing Campaign (2.212 pelanggan setelah dibersihkan, belanja dalam satuan uang) dipakai untuk menunjukkan apa yang harus disesuaikan ketika dataset diganti.

| Algoritma | Jenis | Jumlah cluster | Titik ambigu |
|---|---|---|---|
| Fuzzy C-Means | *Soft clustering* berbasis jarak ke centroid | Ditentukan (c) | Diberi keanggotaan parsial di beberapa cluster |
| DBSCAN | *Hard clustering* berbasis kepadatan | Ditemukan otomatis | Dilabeli noise (-1) |

Alur kerja yang sama dipakai pada keduanya:

```
Load data -> Pilih fitur -> Normalisasi -> Pilih parameter -> Jalankan algoritma
          -> Evaluasi -> Interpretasi cluster -> Visualisasi -> Simpan hasil
```

---

## 2. Dataset dan Persiapan Data

### 2.1 Struktur dataset

Dataset berisi 200 baris tanpa *missing value*.

| Kolom | Tipe | Dipakai? | Alasan |
|---|---|---|---|
| `CustomerID` | integer | Tidak | Hanya penanda, tidak membawa informasi pola |
| `Gender` | kategorikal | Tidak | Jarak Euclidean tidak cocok untuk data kategorikal |
| `Age` | numerik | Tidak (hanya interpretasi) | Dipakai untuk menggambarkan profil cluster setelah clustering |
| `Annual Income (k$)` | numerik | **Ya** | Fitur clustering |
| `Spending Score (1-100)` | numerik | **Ya** | Fitur clustering |

Memakai dua fitur membuat hasil mudah divisualisasikan dalam grafik 2D.

### 2.2 Normalisasi (Standardisasi Z-score)

Kedua algoritma berbasis jarak, sehingga fitur harus berskala sama. Setiap nilai diubah menjadi:

$$z = \frac{x - \mu}{\sigma}$$

dengan $\mu$ rata-rata fitur dan $\sigma$ simpangan baku populasi fitur tersebut (`StandardScaler` memakai `ddof=0`). Setelah transformasi, tiap fitur memiliki rata-rata 0 dan simpangan baku 1.

```python
fitur = ['Annual Income (k$)', 'Spending Score (1-100)']
X  = df[fitur].values                       # data asli, bentuk (200, 2)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)                # data ternormalisasi, bentuk (200, 2)
```

| Variabel | Arti |
|---|---|
| `df` | DataFrame hasil pembacaan CSV |
| `fitur` | Daftar nama kolom yang dipakai sebagai fitur |
| `X` | Matriks data asli, ukuran $n \times d$ ($n=200$ data, $d=2$ fitur) |
| `scaler` | Objek `StandardScaler`, disimpan agar bisa dipakai balik (`inverse_transform`) dan untuk transformasi data baru |
| `Xs` | Matriks data yang sudah distandardisasi. Semua algoritma bekerja pada matriks ini |

### 2.3 Jarak Euclidean

Semua perhitungan jarak memakai jarak Euclidean antara dua titik $x_i$ dan $x_j$ pada $d$ dimensi:

$$d(x_i, x_j) = \lVert x_i - x_j \rVert = \sqrt{\sum_{l=1}^{d} (x_{il} - x_{jl})^2}$$

---

## 3. Fuzzy C-Means (FCM)

### 3.1 Konsep

Pada K-Means, tiap data hanya masuk satu cluster. Pada FCM, tiap data memiliki **derajat keanggotaan** $u_{ik} \in [0,1]$ terhadap **setiap** cluster $k$. Contoh: pelanggan A memiliki keanggotaan 0,70 pada cluster 1, 0,25 pada cluster 3, dan 0,05 pada cluster 2.

Batasan matriks keanggotaan $U$ (ukuran $n \times c$):

$$\sum_{k=1}^{c} u_{ik} = 1 \quad \text{untuk setiap data } i, \qquad 0 \le u_{ik} \le 1$$

### 3.2 Parameter

| Parameter | Simbol | Arti | Nilai yang dipakai |
|---|---|---|---|
| Jumlah cluster | $c$ | Banyaknya cluster yang dicari | 5 (dipilih lewat validasi) |
| Fuzzifier | $m$ | Tingkat "kekaburan". $m \to 1$ mendekati K-Means, $m$ besar membuat keanggotaan makin merata | 2,0 |
| Toleransi | `tol` / `error` | Batas perubahan $U$ untuk berhenti | $10^{-5}$ |
| Iterasi maksimum | `max_iter` / `maxiter` | Batas iterasi | 150 |
| Seed | `seed` | Pengunci angka acak agar hasil bisa direproduksi | 42 |

### 3.3 Rumus FCM

**Fungsi objektif** yang diminimalkan:

$$J_m(U,V) = \sum_{i=1}^{n}\sum_{k=1}^{c} u_{ik}^{\,m}\,\lVert x_i - v_k \rVert^2$$

dengan $x_i$ data ke-$i$ dan $v_k$ pusat (centroid) cluster ke-$k$.

**Update pusat cluster:**

$$v_k = \frac{\sum_{i=1}^{n} u_{ik}^{\,m}\, x_i}{\sum_{i=1}^{n} u_{ik}^{\,m}}$$

Centroid adalah rata-rata data yang dibobot dengan $u_{ik}^m$, sehingga data yang lebih dekat berpengaruh lebih besar.

**Update keanggotaan:**

$$u_{ik} = \frac{1}{\displaystyle\sum_{j=1}^{c}\left(\frac{d_{ik}}{d_{ij}}\right)^{\frac{2}{m-1}}} \;=\; \frac{d_{ik}^{-\frac{2}{m-1}}}{\displaystyle\sum_{j=1}^{c} d_{ij}^{-\frac{2}{m-1}}}$$

dengan $d_{ik} = \lVert x_i - v_k \rVert$. Bentuk kedua inilah yang dipakai pada kode (`inv / inv.sum(...)`).

**Kriteria berhenti:** iterasi berhenti ketika

$$\lVert U^{(t+1)} - U^{(t)} \rVert < \text{tol}$$

(norma Frobenius dari selisih matriks keanggotaan), atau ketika mencapai `max_iter`.

**Defuzzifikasi** (mengubah keanggotaan menjadi label tegas):

$$\text{label}_i = \arg\max_{k}\; u_{ik}$$

### 3.4 Algoritma FCM

1. Inisialisasi $U$ secara acak, lalu normalisasi tiap baris agar berjumlah 1.
2. Hitung centroid $v_k$ dari $U$ saat ini.
3. Hitung jarak $d_{ik}$ dan perbarui $U$.
4. Hitung $J_m$ untuk memantau konvergensi.
5. Ulangi langkah 2 sampai 4 hingga kriteria berhenti terpenuhi.
6. Ambil label dengan `argmax` dan analisis cluster.

### 3.5 Implementasi dari nol: penjelasan variabel

```python
def fcm(X, c, m=2.0, max_iter=150, tol=1e-5, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U /= U.sum(axis=1, keepdims=True)
    J_hist = []
    for it in range(max_iter):
        Um = U ** m
        V = (Um.T @ X) / Um.sum(axis=0)[:, None]
        D = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
        D = np.fmax(D, 1e-10)
        J_hist.append(np.sum(Um * D ** 2))
        inv = D ** (-2 / (m - 1))
        U_new = inv / inv.sum(axis=1, keepdims=True)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    return V, U, J_hist, it + 1
```

| Variabel | Bentuk | Arti | Rumus terkait |
|---|---|---|---|
| `X` | $(n, d)$ | Data masukan (di sini `Xs`) | $x_i$ |
| `c` | skalar | Jumlah cluster | $c$ |
| `m` | skalar | Fuzzifier | $m$ |
| `max_iter` | skalar | Batas iterasi | |
| `tol` | skalar | Toleransi konvergensi | |
| `seed` | skalar | Seed generator acak | |
| `rng` | objek | Generator angka acak dengan seed tetap | |
| `n` | skalar | Jumlah data (200) | $n$ |
| `U` | $(n, c)$ | Matriks keanggotaan. `U[i, k]` adalah $u_{ik}$ | $u_{ik}$ |
| `Um` | $(n, c)$ | `U ** m`, yaitu keanggotaan dipangkatkan $m$ | $u_{ik}^m$ |
| `V` | $(c, d)$ | Matriks centroid. Baris ke-$k$ adalah $v_k$ | $v_k$ |
| `D` | $(n, c)$ | Jarak Euclidean tiap data ke tiap centroid | $d_{ik}$ |
| `1e-10` pada `np.fmax` | skalar | Batas bawah jarak agar tidak terjadi pembagian nol saat data tepat di centroid | |
| `J_hist` | list | Nilai fungsi objektif tiap iterasi (untuk grafik konvergensi) | $J_m$ |
| `inv` | $(n, c)$ | $d_{ik}^{-2/(m-1)}$, pembilang rumus keanggotaan | |
| `U_new` | $(n, c)$ | Keanggotaan hasil update | $u_{ik}^{(t+1)}$ |
| `it` | skalar | Indeks iterasi. Nilai yang dikembalikan adalah `it + 1` (jumlah iterasi) | |

Perhitungan `D` memakai *broadcasting*: `X[:, None, :]` berbentuk $(n,1,d)$ dikurangi `V[None, :, :]` berbentuk $(1,c,d)$ menghasilkan selisih $(n,c,d)$. Operasi `norm(..., axis=2)` lalu menghitung jarak pada sumbu fitur.

### 3.6 Memilih jumlah cluster

FCM dijalankan untuk $c = 2$ sampai $8$, lalu dibandingkan dengan beberapa metrik (rumus di [Bagian 5](#5-metrik-evaluasi)).

| c | Iterasi | PC | PE | Xie-Beni | Silhouette | DBI |
|---|---|---|---|---|---|---|
| 2 | 75 | 0,697 | 0,463 | 0,430 | 0,363 | 1,115 |
| 3 | 28 | 0,680 | 0,578 | 0,171 | 0,464 | 0,723 |
| 4 | 49 | 0,663 | 0,669 | 0,168 | 0,486 | 0,725 |
| **5** | 37 | 0,671 | 0,704 | **0,087** | **0,555** | **0,572** |
| 6 | 49 | 0,617 | 0,816 | 0,471 | 0,456 | 0,673 |
| 7 | 136 | 0,565 | 0,951 | 0,613 | 0,408 | 0,679 |
| 8 | 107 | 0,561 | 0,986 | 0,529 | 0,416 | 0,717 |

PC dan PE secara alami cenderung memilih $c$ kecil, jadi keduanya tidak dijadikan satu-satunya dasar. $c = 5$ unggul pada Xie-Beni, Silhouette, dan DBI, dan sesuai dengan lima kelompok yang tampak pada sebaran data.

#### 3.6.1 Membaca loop pemilihan `c` (versi scikit-fuzzy)

```python
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
```

**Arti `pe` (Partition Entropy)** mengikuti rumus pada [Bagian 5.1](#51-metrik-khusus-fcm). Baris kodenya dibaca dari dalam ke luar:

| Bagian kode | Arti |
|---|---|
| `u` | Matriks keanggotaan berbentuk $(c, n)$ |
| `np.fmax(u, 1e-12)` | Mengganti nilai di bawah $10^{-12}$ dengan $10^{-12}$ agar $\ln 0$ tidak menghasilkan `-inf` atau `nan` |
| `np.log(...)` | Logaritma natural tiap keanggotaan, $\ln u_{ik}$ |
| `u * np.log(...)` | Perkalian elemen demi elemen, $u_{ik}\ln u_{ik}$ |
| `np.sum(...)` | Jumlah seluruh elemen (semua cluster dan semua data) |
| tanda `-` di depan | Membuat hasil positif, karena $\ln u \le 0$ untuk $u \in (0, 1]$ |
| `/ u.shape[1]` | Dibagi $n$ (jumlah data). `u.shape` adalah $(c, n)$, jadi indeks `[1]` adalah $n$ |

Pada versi dari nol, matriks `U` berbentuk $(n, c)$ (transpose dari `u`), sehingga pembaginya `U.shape[0]`. Rumusnya sama.

PE mengukur ketidakpastian keanggotaan: **makin kecil, makin tegas**. Contoh satu data dengan 3 cluster:

| Keanggotaan | Entropi |
|---|---|
| [1, 0, 0] (sangat tegas) | 0 |
| [0,70, 0,25, 0,05] | $-(0{,}7\ln 0{,}7 + 0{,}25\ln 0{,}25 + 0{,}05\ln 0{,}05) \approx 0{,}746$ |
| [1/3, 1/3, 1/3] (paling kabur) | $\ln 3 \approx 1{,}099$ |

Jadi PE berkisar dari 0 sampai $\ln c$. PE cenderung naik seiring bertambahnya $c$ karena keanggotaan memang makin terbagi (pada Mall: 0,485 untuk $c=2$ sampai 0,986 untuk $c=8$). Itu sebabnya PE dipakai sebagai pelengkap, bukan penentu tunggal.

**Isi `hasil.append([...])`.** Tiap putaran loop menambahkan satu baris berisi enam nilai untuk nilai $c$ yang sedang diuji, dengan urutan yang sama seperti `columns` pada DataFrame:

| Posisi | Nilai | Arti | Arah terbaik |
|---|---|---|---|
| 1 | `c` | Jumlah cluster yang diuji | |
| 2 | `n_iter` | Iterasi sampai konvergen | |
| 3 | `fpc` | Fuzzy Partition Coefficient (sama dengan PC) | Besar |
| 4 | `pe` | Partition Entropy | Kecil |
| 5 | `silhouette_score(Xs, lab)` | Silhouette pada label `argmax` | Besar |
| 6 | `davies_bouldin_score(Xs, lab)` | Davies-Bouldin pada label `argmax` | Kecil |

Contoh satu baris (Mall, $c=5$): `[5, 40, 0.671, 0.704, 0.555, 0.572]`. Setelah loop selesai, `hasil` berisi tujuh baris ($c$ = 2 sampai 8) yang diubah menjadi tabel oleh `pd.DataFrame`.

Tiga catatan:

1. `hasil = []` harus dibuat **sebelum** loop.
2. `fpc` dan `pe` dihitung dari keanggotaan **fuzzy** `u`, sedangkan Silhouette dan DBI dari label **tegas** `lab`. Kedua kelompok metrik menilai hal yang berbeda: ketegasan keanggotaan versus kualitas pemisahan cluster.
3. Bila `n_iter` sama dengan `maxiter` (150), algoritma berhenti karena batas iterasi, bukan karena mencapai `error`. Hasil pada baris itu belum tentu konvergen penuh. Hal ini muncul pada dataset Marketing Campaign untuk $c=6$ dan $c=7$ ([Bagian 8.3](#83-fcm-pada-marketing-campaign)).

### 3.7 Hasil FCM (c = 5, versi dari nol)

Konvergen pada iterasi ke-37. Sebanyak 26 pelanggan memiliki keanggotaan maksimum di bawah 0,6 (titik ambigu di perbatasan cluster).

| Cluster | Jumlah | Usia (rata-rata) | Income (k$) | Spending | Interpretasi |
|---|---|---|---|---|---|
| 0 | 22 | 25,3 | 25,7 | 79,4 | Muda, income rendah, belanja tinggi |
| 1 | 81 | 42,7 | 55,3 | 49,5 | Pelanggan menengah |
| 2 | 23 | 45,2 | 26,3 | 20,9 | Income rendah, belanja rendah |
| 3 | 35 | 41,1 | 88,2 | 17,1 | Income tinggi, belanja rendah (hemat) |
| 4 | 39 | 32,7 | 86,5 | 82,1 | Income tinggi, belanja tinggi (target utama) |

Kolom keluaran yang ditambahkan ke DataFrame:

| Kolom | Arti |
|---|---|
| `Cluster` | Label hasil defuzzifikasi (`argmax` keanggotaan) |
| `MaxMembership` | Keanggotaan tertinggi data tersebut. Nilai kecil berarti posisi ambigu |
| `U_C0` sampai `U_C4` | Derajat keanggotaan data terhadap tiap cluster |

### 3.8 Versi library (scikit-fuzzy)

```python
import skfuzzy as fuzz

cntr, u, u0, d, jm, n_iter, fpc = fuzz.cluster.cmeans(
    Xs.T, c=5, m=2.0, error=1e-5, maxiter=150, seed=42)
```

Library meminta input berbentuk **(fitur, data)**, sehingga `Xs` harus di-transpose.

| Variabel | Bentuk | Arti | Padanan pada versi dari nol |
|---|---|---|---|
| `Xs.T` | $(d, n)$ | Data masukan yang di-transpose | `X` |
| `cntr` | $(c, d)$ | Pusat cluster (skala ternormalisasi) | `V` |
| `u` | $(c, n)$ | Matriks keanggotaan akhir | `U` (transpose) |
| `u0` | $(c, n)$ | Matriks keanggotaan awal (acak) | |
| `d` | $(c, n)$ | Jarak data ke centroid | `D` (transpose) |
| `jm` | $(t,)$ | Riwayat fungsi objektif per iterasi | `J_hist` |
| `n_iter` | skalar | Jumlah iterasi sampai konvergen | `it + 1` |
| `fpc` | skalar | *Fuzzy Partition Coefficient* | `pc(U)` |
| `M` | skalar | Konstanta fuzzifier pada skrip (`M = 2.0`) | `m` |
| `labels` / `u.argmax(axis=0)` | $(n,)$ | Label cluster. Perhatikan `axis=0` karena `u` berbentuk $(c,n)$ | `U.argmax(axis=1)` |

Prediksi data baru memakai `cmeans_predict`:

```python
baru = np.array([[70, 80], [30, 20]])                    # [income, spending]
u_baru, *_ = fuzz.cluster.cmeans_predict(
    scaler.transform(baru).T, cntr, m=2.0, error=1e-5, maxiter=150)
```

Data baru harus ditransformasi dengan `scaler` yang **sama** (bukan `fit` ulang), lalu dihitung keanggotaannya terhadap centroid yang sudah dilatih. Nilainya juga harus berskala sama dengan data asli: contoh `[70, 80]` hanya valid untuk Mall Customers (income dalam k$, skor 1 sampai 100). Pada Marketing Campaign, income berskala puluhan ribu dan belanja berskala ratusan sampai ribuan ([Bagian 8.3](#83-fcm-pada-marketing-campaign)). Contoh hasil: pelanggan income 70 dan spending 80 memiliki keanggotaan 0,720 pada cluster "income tinggi, belanja tinggi", sedangkan income 30 dan spending 20 memiliki keanggotaan 0,985 pada cluster "income rendah, belanja rendah".

Hasil versi library: konvergen dalam 40 iterasi, FPC = 0,671, J akhir = 43,464, dengan 26 titik ambigu. Ukuran cluster identik dengan versi dari nol (35, 23, 81, 39, 22), hanya **nomor cluster berbeda** karena inisialisasi acak berbeda. Label cluster memang tidak memiliki urutan baku.

Untuk menampilkan centroid pada skala asli (k$ dan skor), centroid dikembalikan dengan:

```python
pusat = scaler.inverse_transform(cntr)      # versi library
pusat = StandardScaler().fit(X).inverse_transform(V)   # versi dari nol
```

---

## 4. DBSCAN

### 4.1 Konsep

DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) membentuk cluster dari area yang padat dan menandai titik di area jarang sebagai noise. Jumlah cluster tidak ditentukan di awal dan bentuk cluster boleh tidak bulat.

### 4.2 Parameter

| Parameter | Simbol | Arti | Nilai akhir |
|---|---|---|---|
| `eps` / `EPS` | $\varepsilon$ | Radius lingkungan sebuah titik | 0,38 |
| `min_samples` / `MIN_SAMPLES` | $\text{MinPts}$ | Jumlah minimal titik dalam radius $\varepsilon$ (termasuk titik itu sendiri) agar area dianggap padat | 6 |

### 4.3 Definisi dan rumus

**Lingkungan-ε** dari titik $p$:

$$N_\varepsilon(p) = \{\, q \in X \mid d(p,q) \le \varepsilon \,\}$$

Karena $d(p,p)=0$, titik $p$ ikut terhitung dalam $N_\varepsilon(p)$.

**Tipe titik:**

| Tipe | Syarat |
|---|---|
| **Core** | $\lvert N_\varepsilon(p) \rvert \ge \text{MinPts}$ |
| **Border** | Bukan core, tetapi berada di $N_\varepsilon(q)$ milik suatu core $q$ |
| **Noise** | Bukan core dan bukan border |

**Keterhubungan:**
- Titik $q$ *directly density-reachable* dari $p$ jika $q \in N_\varepsilon(p)$ dan $p$ adalah core.
- Titik $q$ *density-reachable* dari $p$ jika ada rantai core $p = p_1, p_2, \dots, p_r = q$ dengan tiap $p_{s+1}$ *directly density-reachable* dari $p_s$.
- Satu cluster adalah himpunan maksimal titik yang saling *density-connected*.

**k-distance** untuk memilih $\varepsilon$: jarak titik $p$ ke tetangga ke-$k$ terdekatnya, dengan $k = \text{MinPts}$:

$$\text{kdist}(p) = d\big(p,\; \text{tetangga ke-}k\text{ terdekat}\big)$$

Nilai `kdist` seluruh titik diurutkan naik lalu digambar. "Tekukan" kurva menjadi kandidat $\varepsilon$.

### 4.4 Algoritma DBSCAN

1. Hitung $N_\varepsilon(p)$ untuk semua titik, lalu tandai titik core.
2. Untuk tiap core yang belum punya cluster, buat cluster baru.
3. Perluas cluster: masukkan semua tetangganya ke antrian. Tetangga yang juga core menambahkan tetangganya sendiri. Border ikut masuk cluster tetapi tidak memperluasnya.
4. Titik yang tidak pernah terjangkau tetap berlabel -1 (noise).

### 4.5 Implementasi dari nol: penjelasan variabel

```python
def dbscan(X, eps, min_samples):
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    tetangga = [np.where(D[i] <= eps)[0] for i in range(n)]
    is_core  = np.array([len(t) >= min_samples for t in tetangga])

    labels = np.full(n, -1)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cid
        antrian = deque(tetangga[i])
        while antrian:
            j = antrian.popleft()
            if labels[j] == -1:
                labels[j] = cid
                if is_core[j]:
                    antrian.extend(tetangga[j])
        cid += 1
    return labels, is_core
```

| Variabel | Bentuk | Arti |
|---|---|---|
| `X` | $(n, d)$ | Data masukan (di sini `Xs`) |
| `eps` | skalar | Radius $\varepsilon$ |
| `min_samples` | skalar | $\text{MinPts}$ |
| `n` | skalar | Jumlah data |
| `D` | $(n, n)$ | Matriks jarak antar semua pasangan titik. `D[i, j]` adalah $d(x_i, x_j)$. Cukup untuk 200 data, tetapi memori tumbuh kuadratik untuk data besar |
| `tetangga` | list of array | `tetangga[i]` berisi indeks semua titik dalam radius $\varepsilon$ dari titik $i$ (termasuk dirinya), yaitu $N_\varepsilon(x_i)$ |
| `is_core` | $(n,)$ boolean | `True` bila titik $i$ adalah core |
| `labels` | $(n,)$ | Label cluster. Awalnya semua -1 (noise) |
| `cid` | skalar | Nomor cluster yang sedang dibentuk |
| `i` | skalar | Indeks titik yang sedang diperiksa pada loop utama |
| `antrian` | `deque` | Antrian titik yang perlu diproses saat memperluas cluster (penelusuran *breadth-first*) |
| `j` | skalar | Indeks titik yang sedang diambil dari antrian |

Titik yang semula berlabel -1 dan kemudian dijangkau dari core akan berubah menjadi anggota cluster. Inilah cara border ditambahkan. Titik yang tetap -1 sampai akhir adalah noise sejati.

### 4.6 Versi library (scikit-learn)

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.38, min_samples=6).fit(Xs)
labels  = db.labels_
is_core = np.zeros(len(Xs), dtype=bool)
is_core[db.core_sample_indices_] = True
```

| Variabel | Arti |
|---|---|
| `db` | Objek DBSCAN yang sudah dilatih |
| `db.labels_` | Label cluster tiap titik (-1 untuk noise) |
| `db.core_sample_indices_` | Indeks titik yang berstatus core |
| `is_core` | Penanda boolean core, dibangun dari indeks di atas |
| `df['Tipe']` | Kolom `core`, `border`, atau `noise`. Border adalah titik ber-cluster yang bukan core |

Versi dari nol diverifikasi dengan `adjusted_rand_score(labels, db.labels_)` dan menghasilkan **1,0** (label identik), dengan jumlah core sama-sama 150.

### 4.7 Memilih parameter

#### 4.7.1 Memahami k-distance

**k-distance** adalah jarak sebuah titik ke tetangga terdekat ke-$k$ miliknya, dengan $k$ disamakan dengan `min_samples`:

$$\text{kdist}(p) = d\big(p,\ \text{tetangga ke-}k\text{ terdekat dari }p\big)$$

Artinya: bila `kdist(p) = 0,3` dengan $k = 6$, maka dalam radius 0,3 dari $p$ terdapat 6 titik (termasuk $p$ sendiri). Dengan $\varepsilon \ge 0{,}3$, titik $p$ memenuhi syarat sebagai core. Jadi k-distance menjawab pertanyaan: *seberapa besar radius yang dibutuhkan tiap titik agar punya cukup tetangga?*

Cara membacanya:

- Titik di area **padat** punya k-distance kecil. Titik di area **jarang** (calon noise) punya k-distance besar.
- Bila semua nilai diurutkan naik lalu digambar, kurva biasanya datar di awal lalu **menekuk naik** di akhir. Titik tekukan (*knee*) menjadi kandidat $\varepsilon$.
- $\varepsilon$ di bawah tekukan: terlalu banyak titik menjadi noise. $\varepsilon$ di atas tekukan: cluster yang seharusnya terpisah mulai menyatu.

```python
MIN_SAMPLES = 6
jarak, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs).kneighbors(Xs)
k_dist = np.sort(jarak[:, -1])
plt.plot(k_dist)
```

| Baris | Arti |
|---|---|
| `NearestNeighbors(n_neighbors=MIN_SAMPLES)` | Mencari `MIN_SAMPLES` tetangga terdekat untuk tiap titik |
| `.kneighbors(Xs)` | Mengembalikan `jarak` berbentuk $(n, \text{MIN\_SAMPLES})$, terurut dari yang terdekat. Kolom pertama selalu 0 karena itu titik itu sendiri |
| `jarak[:, -1]` | Kolom terakhir, yaitu jarak ke tetangga ke-`MIN_SAMPLES`. Inilah k-distance |
| `np.sort(...)` | Mengurutkan naik agar terbentuk kurva |

Karena titik itu sendiri dihitung sebagai tetangga pertama, definisi ini cocok dengan `min_samples` pada DBSCAN yang juga menghitung titik itu sendiri.

Kurva k-distance tidak selalu punya tekukan tegas (pada Mall tidak setajam contoh buku teks, pada Marketing Campaign kurva naik perlahan sampai persentil ke-95 lalu melonjak di ujung sehingga lokasi tekukan sulit ditentukan tepat). Karena itu k-distance dipakai sebagai **titik awal** rentang $\varepsilon$, lalu dipersempit dengan grid search dan dicek lewat jumlah cluster, jumlah noise, dan profil cluster. Salah satu cara menyatakannya secara terukur adalah memakai persentil k-distance (misalnya persentil 95, sehingga sekitar 5% titik berada di atas $\varepsilon$), seperti pada [Bagian 8.4](#84-dbscan-pada-marketing-campaign).

#### 4.7.2 Variabel dan grid search

Variabel pada tahap ini:

| Variabel | Arti |
|---|---|
| `nn` | Objek `NearestNeighbors(n_neighbors=MIN_SAMPLES)` untuk mencari tetangga terdekat |
| `jarak` | Matriks jarak tiap titik ke `MIN_SAMPLES` tetangga terdekatnya (titik itu sendiri dihitung sebagai tetangga pertama) |
| `k_dist` | `jarak[:, -1]` yang diurutkan naik. Ini kurva k-distance |
| `baris` | Daftar hasil grid search (`min_samples`, `eps`, jumlah cluster, jumlah noise, silhouette) |
| `m` (pada grid search) | Mask boolean `labels != -1`, yaitu titik yang bukan noise |
| `n_c` | Jumlah cluster hasil pencarian (tanpa noise) |

Pada dataset ini tekukan k-distance tidak setajam contoh buku teks (jarak mulai naik di kisaran 0,35 sampai 0,45), sehingga ditambah grid search:

| min_samples | eps | Jumlah cluster | Noise | Silhouette |
|---|---|---|---|---|
| 5 | 0,35 | 6 | 23 | 0,558 |
| 6 | 0,35 | 7 | 25 | 0,503 |
| **6** | **0,38** | **5** | **23** | **0,541** |
| 5 | 0,40 | 4 | 15 | 0,478 |
| 8 | 0,35 | 4 | 55 | 0,599 |
| 8 | 0,40 | 5 | 25 | 0,549 |

Silhouette dihitung **tanpa noise**, sehingga nilainya bisa tampak tinggi ketika noise banyak (misalnya `min_samples=8`, `eps=0,35` dengan 55 noise). Karena itu pilihan akhir mempertimbangkan juga jumlah noise yang wajar (11,5%) dan kecocokan dengan struktur data, bukan hanya Silhouette tertinggi. Dipilih **eps = 0,38** dan **min_samples = 6**.

### 4.8 Hasil DBSCAN

Hasilnya 150 core, 27 border, dan 23 noise. Silhouette = 0,541 dan Davies-Bouldin = 0,518 (noise dikeluarkan).

| Cluster | Jumlah | Usia (rata-rata) | Income (k$) | Spending | Interpretasi |
|---|---|---|---|---|---|
| 0 | 12 | 48,8 | 24,6 | 9,6 | Income rendah, belanja sangat rendah |
| 1 | 16 | 24,1 | 25,4 | 75,6 | Muda, income rendah, belanja tinggi |
| 2 | 95 | 42,5 | 52,8 | 47,1 | Pelanggan menengah |
| 3 | 31 | 32,6 | 80,2 | 83,6 | Income tinggi, belanja tinggi |
| 4 | 23 | 41,2 | 81,0 | 12,8 | Income tinggi, belanja rendah |
| -1 | 23 | 34,7 | 89,0 | 58,8 | Noise |

Noise terkumpul di wilayah income tinggi dengan spending sedang, yaitu zona perbatasan antara cluster 3 dan 4 yang kepadatannya rendah.

---

## 5. Metrik Evaluasi

### 5.1 Metrik khusus FCM

Metrik ini memakai matriks keanggotaan $U$ dan dihitung langsung pada kode (`pc`, `pe`, `xie_beni`).

**Partition Coefficient (PC)** (makin besar makin baik, maksimum 1):

$$PC = \frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{c} u_{ik}^2$$

Pada scikit-fuzzy nilai ini dikembalikan sebagai `fpc`.

**Partition Entropy (PE)** (makin kecil makin baik):

$$PE = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{c} u_{ik}\,\ln u_{ik}$$

Pada kode, `np.fmax(U, 1e-12)` dipakai agar $\ln 0$ tidak terjadi.

**Xie-Beni Index (XB)** (makin kecil makin baik):

$$XB = \frac{\displaystyle\sum_{i=1}^{n}\sum_{k=1}^{c} u_{ik}^{\,m}\,\lVert x_i - v_k \rVert^2}{n \cdot \displaystyle\min_{p \ne q} \lVert v_p - v_q \rVert^2}$$

Pembilang mengukur kekompakan cluster, penyebut mengukur jarak minimum antar centroid. Variabel pada kode `xie_beni`:

| Variabel | Arti |
|---|---|
| `D2` | Jarak kuadrat tiap data ke tiap centroid, bentuk $(n,c)$ |
| `sep` | Jarak kuadrat minimum antar dua centroid berbeda (pemisah antar cluster) |
| nilai kembali | Pembilang $\sum u^m d^2$ dibagi $n \cdot$ `sep` |

### 5.2 Metrik umum (FCM dan DBSCAN)

Keduanya dihitung pada label tegas. Untuk FCM dipakai label `argmax`, untuk DBSCAN titik noise dikeluarkan lebih dulu.

**Silhouette** (rentang -1 sampai 1, makin besar makin baik). Untuk titik $i$:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$$

dengan $a(i)$ rata-rata jarak $i$ ke titik lain dalam clusternya, dan $b(i)$ rata-rata jarak $i$ ke titik-titik di cluster terdekat lainnya. Skor akhir adalah rata-rata $s(i)$ seluruh titik.

**Davies-Bouldin Index (DBI)** (makin kecil makin baik):

$$DBI = \frac{1}{K}\sum_{i=1}^{K}\max_{j \ne i}\frac{\sigma_i + \sigma_j}{d(v_i, v_j)}$$

dengan $K$ jumlah cluster, $\sigma_i$ rata-rata jarak titik pada cluster $i$ ke centroidnya, dan $d(v_i, v_j)$ jarak antar centroid.

**Adjusted Rand Index (ARI)** dipakai hanya untuk membandingkan dua pelabelan (versi dari nol vs library pada DBSCAN). Nilai 1 berarti pengelompokan identik, nilai mendekati 0 berarti setara tebakan acak. ARI tidak bergantung pada nomor cluster.

---

## 6. Perbandingan FCM dan DBSCAN

Perbandingan di bawah untuk dataset Mall Customers. Perbandingan pada Marketing Campaign ada di [Bagian 8.5](#85-perbandingan-fcm-dan-dbscan-pada-marketing-campaign).

| Aspek | FCM | DBSCAN |
|---|---|---|
| Jumlah cluster | Ditentukan (c = 5) | Ditemukan otomatis (5) |
| Titik ambigu | Keanggotaan parsial (26 data < 0,6) | Noise (23 data) |
| Parameter | $c$, $m$ | $\varepsilon$, MinPts |
| Sensitivitas parameter | Sedang | **Tinggi** |
| Bentuk cluster | Cenderung bulat (berbasis centroid) | Bebas |
| Prediksi data baru | Mudah (`cmeans_predict`) | Tidak ada fungsi bawaan |
| Kompleksitas utama | $O(n \cdot c \cdot d)$ per iterasi | $O(n^2)$ untuk implementasi ini (matriks jarak penuh) |

Kedua algoritma menghasilkan lima segmen yang serupa: (1) muda, income rendah, belanja tinggi, (2) income rendah, belanja rendah, (3) menengah, (4) income tinggi, belanja tinggi, (5) income tinggi, belanja rendah. Ini menunjukkan struktur data pada dua fitur tersebut cukup stabil terhadap pilihan metode.

---

## 7. Mengganti Dataset: Checklist Penyesuaian

Saat dataset diganti, **fungsi algoritma tidak berubah**, tetapi **persiapan data dan nilai parameter harus ditinjau ulang**.

### 7.1 Yang tidak perlu diubah

Fungsi `fcm()`, `dbscan()`, metrik (`pc`, `pe`, `xie_beni`), perhitungan Silhouette dan Davies-Bouldin, serta kode visualisasi tetap sama, selama data masukan berupa matriks numerik ternormalisasi dengan dua fitur.

### 7.2 Yang harus disesuaikan

| Aspek | Mall Customers | Marketing Campaign | Tindakan |
|---|---|---|---|
| Pemuatan data | Dipisah koma | Dipisah **tab** | `pd.read_csv(..., sep='\t')` |
| Pemilihan fitur | Dua kolom numerik siap pakai | 29 kolom, banyak yang tidak cocok | Pilih fitur, buang penanda dan kolom konstan, buat fitur turunan |
| Missing value | Tidak ada | `Income` kosong pada 24 baris | `dropna` atau isi median |
| Outlier | Tidak bermasalah | `Income` 666.666 dan 3 tahun lahir sebelum 1920 | Buang dengan batas wajar |
| Skala fitur | Income k$, skor 1 sampai 100 | Income puluhan ribu, belanja sampai 2.525 | Normalisasi, dan sesuaikan data uji prediksi |
| Jumlah cluster FCM ($c$) | 5 | **2** (tiga metrik utama) | Pilih dari tabel metrik, jangan dibawa dari dataset lain |
| Rentang `eps` DBSCAN | 0,35 sampai 0,40 | 0,05 sampai 0,30 | Lihat k-distance, geser rentang grid |
| Garis `eps` pada plot | Angka tetap | Mengikuti variabel `EPS` | Hindari angka yang ditulis manual |
| Evaluasi DBSCAN | Selalu ada beberapa cluster | Bisa hanya 1 cluster | Beri pengaman sebelum Silhouette dan DBI |
| Ambang titik ambigu FCM | 0,6 | 0,6 | Baca sesuai $c$: pada $c=2$ keanggotaan maksimum terkecil adalah 0,5 |
| Memori DBSCAN dari nol | Matriks 200 × 200 | Matriks 2.212 × 2.212 (sekitar 39 MB) | Aman. Untuk puluhan ribu data pakai versi library |
| Folder keluaran | Tidak ada | `Marketing_Campaign/` | Buat dengan `os.makedirs(..., exist_ok=True)` |
| Judul dan komentar | "Mall Customers" | "Marketing Campaign" | Ganti, atau pakai f-string dari variabel |

### 7.3 Prinsip utama

**Parameter tidak boleh dibawa antar dataset.** Nilai $c$, $\varepsilon$, dan `min_samples` bergantung pada jumlah data, sebaran, dan skala fitur. Selalu ulangi pemilihan parameter pada dataset baru, dan terima bila hasilnya berbeda (misalnya $c=2$ alih-alih 5).

---

## 8. Studi Kasus: Marketing Campaign

### 8.1 Dataset

Dataset berisi 2.240 baris dan 29 kolom, dipisah tab.

| Jenis kolom | Contoh | Perlakuan |
|---|---|---|
| Penanda | `ID` | Dibuang |
| Konstan | `Z_CostContact` (selalu 3), `Z_Revenue` (selalu 11) | Dibuang karena tidak membedakan pelanggan |
| Kategorikal | `Education`, `Marital_Status` | Tidak dipakai (jarak Euclidean tidak cocok) |
| Biner | `AcceptedCmp1` sampai `5`, `Complain`, `Response` | Tidak dipakai |
| Tanggal | `Dt_Customer` (format `dd-mm-yyyy`) | Diubah menjadi lama menjadi pelanggan |
| Numerik | `Income`, `Recency`, `Mnt*`, `Num*`, `Year_Birth` | Kandidat fitur |

Masalah kualitas data:

| Masalah | Jumlah |
|---|---|
| `Income` kosong | 24 baris |
| `Income` di atas 200.000 (maksimum 666.666) | 1 baris |
| `Year_Birth` sebelum 1920 (usia sekitar 115 tahun) | 3 baris |
| `Marital_Status` tidak wajar: `Absurd` (2), `YOLO` (2), `Alone` (3) | 7 baris (hanya relevan bila kolom dipakai) |

### 8.2 Persiapan data

Fitur turunan yang dibuat:

| Fitur | Rumus | Keterangan |
|---|---|---|
| `Total_Spending` | $\sum$ kolom `Mnt*` | Total belanja enam kategori (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`) |
| `Age` | tahun `tgl_acuan` − `Year_Birth` | Usia saat data dikumpulkan |
| `Total_Children` | `Kidhome` + `Teenhome` | Jumlah anak di rumah |
| `Tenure_Days` | `tgl_acuan` − `Dt_Customer` (hari) | Lama menjadi pelanggan |

Aturan pembersihan: buang baris dengan `Income` kosong, `Income >= 200000`, dan `Age >= 100`. Sisa data: **2.212 baris**.

Variabel pada kode:

| Variabel | Arti |
|---|---|
| `DATA_PATH` | Lokasi file CSV |
| `OUT_DIR` | Folder keluaran (`Marketing_Campaign`) |
| `tgl_acuan` | Tanggal pendaftaran pelanggan terbaru (29 Juni 2014), dipakai sebagai "hari ini" saat data diambil |
| `mnt` | Daftar kolom yang namanya diawali `Mnt` |
| `fitur` | `['Income', 'Total_Spending']` |
| `X` | Matriks data asli $(2212, 2)$ |
| `Xs` | Matriks ternormalisasi (z-score) |
| `N` | Jumlah data setelah dibersihkan (2.212) |

Statistik fitur terpilih setelah dibersihkan:

| Fitur | Minimum | Median | Maksimum |
|---|---|---|---|
| `Income` | 1.730 | 51.371 | 162.397 |
| `Total_Spending` | 5 | 397 | 2.525 |

Korelasi `Income` dan `Total_Spending` sebesar **0,79**, jauh lebih tinggi daripada pasangan fitur pada Mall Customers. Ini berpengaruh besar pada hasil kedua algoritma.

### 8.3 FCM pada Marketing Campaign

Pemilihan $c$ dengan loop pada [Bagian 3.6.1](#361-membaca-loop-pemilihan-c-versi-scikit-fuzzy) ($m=2$, `maxiter=150`):

| c | Iterasi | FPC | PE | Silhouette | DBI | Data ambigu (< 0,6) |
|---|---|---|---|---|---|---|
| **2** | 22 | **0,845** | **0,264** | **0,591** | **0,583** | 116 (5,2%) |
| 3 | 91 | 0,736 | 0,474 | 0,480 | 0,731 | 290 (13,1%) |
| 4 | 80 | 0,681 | 0,603 | 0,438 | 0,780 | 370 |
| 5 | 129 | 0,634 | 0,719 | 0,400 | 0,826 | 559 (25,3%) |
| 6 | 150 | 0,600 | 0,811 | 0,377 | 0,902 | 694 |
| 7 | 150 | 0,573 | 0,888 | 0,360 | 0,973 | 778 |
| 8 | 148 | 0,555 | 0,954 | 0,366 | 0,901 | 835 |

Semua metrik memilih **$c=2$**. Pada $c=6$ dan $c=7$, iterasi mencapai batas `maxiter` sehingga belum tentu konvergen penuh. Nilai $c=5$ (yang dipakai pada Mall) tidak didukung: seperempat data berada di perbatasan cluster, dan dua cluster "income tinggi, belanja tinggi" saling tumpang tindih.

Profil cluster:

**$c=2$** (FPC 0,845):

| Cluster | Jumlah | Usia | Income | Total_Spending | Interpretasi |
|---|---|---|---|---|---|
| 0 | 914 | 47 | 72.371 | 1.221 | Income dan belanja tinggi |
| 1 | 1.298 | 44 | 37.585 | 175 | Income dan belanja rendah |

**$c=3$** (FPC 0,736), alternatif yang lebih bermanfaat untuk segmentasi bisnis:

| Cluster | Jumlah | Usia | Income | Total_Spending | Interpretasi |
|---|---|---|---|---|---|
| 0 | 673 | 49 | 59.364 | 632 | Menengah |
| 1 | 987 | 42 | 32.459 | 108 | Rendah |
| 2 | 552 | 46 | 77.797 | 1.470 | Tinggi |

Pilihan antara $c=2$ dan $c=3$ adalah keputusan analis. $c=2$ paling kuat secara statistik, sedangkan $c=3$ memberi pembagian rendah, menengah, tinggi yang lebih mudah dipakai dengan kualitas yang masih layak.

**Prediksi data baru.** Data uji harus berskala sama dengan data asli. Contoh yang bisa dipertanggungjawabkan adalah memakai persentil data sendiri:

```python
baru = np.array([
    df[fitur].quantile(0.90).values,    # profil tinggi: [79.803, 1.536]
    df[fitur].quantile(0.10).values,    # profil rendah: [24.081, 35]
])
u_baru, *_ = fuzz.cluster.cmeans_predict(
    scaler.transform(baru).T, cntr, m=M, error=1e-5, maxiter=150)
```

| Profil | c = 2 | c = 3 |
|---|---|---|
| Persentil 90 (tinggi) | 0,963 pada cluster 0 (tinggi) | 0,991 pada cluster 2 (tinggi) |
| Persentil 10 (rendah) | 0,958 pada cluster 1 (rendah) | 0,951 pada cluster 1 (rendah) |

Bila data uji memakai angka Mall (`[[70, 80], [30, 20]]`), kedua pelanggan sama-sama dianggap berpendapatan sangat rendah sehingga hasilnya identik dan tidak bermakna.

### 8.4 DBSCAN pada Marketing Campaign

**Skala k-distance** ($k=6$) jauh lebih kecil daripada Mall karena data lebih rapat:

| Persentil | k-distance |
|---|---|
| 50% | 0,056 |
| 90% | 0,122 |
| 95% | 0,164 |
| 99% | 0,419 |

Parameter Mall (`eps = 0,38`, `min_samples = 6`) berada mendekati ujung kurva (antara persentil 95 dan 99) dan menghasilkan **satu cluster berisi 2.198 titik dan 14 noise**. Akibatnya `silhouette_score` gagal dengan `ValueError: Number of labels is 1`, karena metrik tersebut butuh minimal 2 cluster.

**Grid search yang disesuaikan** ($\varepsilon$ dari 0,05 sampai 0,30, langkah 0,01; `min_samples` 4, 5, 6, 8, 10) dengan kolom tambahan ukuran cluster. Cuplikannya:

| min_samples | eps | Cluster | Noise | Cluster terbesar | Cluster ke-2 | Silhouette |
|---|---|---|---|---|---|---|
| 5 | 0,12 | 3 | 108 | 2.094 | 5 | 0,241 |
| 5 | 0,15 | 3 | 64 | 2.138 | 5 | 0,227 |
| 6 | 0,13 | 2 | 116 | 2.089 | 7 | 0,296 |
| 6 | 0,15 | 2 | 87 | 2.120 | 5 | 0,211 |
| 8 | 0,16 | 2 | 85 | 2.115 | 12 | 0,393 |
| 10 | 0,16 | 2 | 127 | 2.077 | 8 | 0,379 |

Pola yang muncul di hampir semua baris: **satu cluster raksasa plus cluster mungil berisi 5 sampai 12 titik**. Silhouette bisa tampak sedang sampai tinggi (misalnya 0,393), padahal cluster kedua hanya 12 pelanggan. Inilah sebabnya Silhouette DBSCAN (yang mengeluarkan noise) tidak boleh dipakai sendirian.

**Kriteria kelayakan segmentasi** pada kode:

| Konstanta | Nilai | Arti |
|---|---|---|
| `MIN_CLUSTER_SHARE` | 0,05 | Cluster ke-2 minimal 5% dari seluruh data |
| `MAX_NOISE_SHARE` | 0,20 | Noise maksimal 20% dari seluruh data |

Dari seluruh grid, **tidak ada konfigurasi yang lolos**. Pengujian tambahan dengan transformasi log (`log(Income)`, `log(Total_Spending)`) dan kombinasi fitur lain (`Age + Income + Total_Spending`, `Income + Total_Spending + Recency`) juga tidak menghasilkan konfigurasi yang bermakna. Pengujian ini terbatas pada kombinasi tersebut, tetapi polanya konsisten.

**Penyebab:** `Income` dan `Total_Spending` membentuk sebaran menerus (korelasi 0,79) tanpa celah kepadatan, sedangkan DBSCAN memisahkan cluster lewat celah kepadatan. Ini sejalan dengan FCM yang paling menyukai $c=2$ dan menunjukkan banyak data ambigu pada $c$ besar.

**Mode anomali.** Bila tidak ada konfigurasi layak, kode beralih ke `MODE = 'anomali'`: DBSCAN dipakai untuk menandai pelanggan yang tidak biasa, bukan untuk segmentasi. Parameternya `min_samples = 6` dan $\varepsilon$ = persentil 95 k-distance (0,164), sehingga sekitar 5% titik berada di luar jangkauan.

Hasilnya: 2.101 core, 50 border, dan 61 noise (2,8%).

| Kelompok | Jumlah | Income rata-rata | Belanja rata-rata | Income maks | Belanja maks |
|---|---|---|---|---|---|
| Normal (cluster 0) | 2.151 | 51.208 | 587 | 98.777 | 2.283 |
| Noise | 61 | 78.430 | 1.327 | 162.397 | 2.525 |

Noise berkumpul di tepi sebaran: pelanggan berbelanja sangat tinggi (di atas 2.000) dan pelanggan dengan income sangat tinggi tetapi belanja rendah. Kelompok ini menarik dianalisis terpisah, misalnya sebagai kandidat program loyalitas atau pelanggan berpotensi yang belum tergarap. Usia rata-rata hampir sama (44 dan 45), jadi usia tidak membedakan.

Variabel baru pada skrip:

| Variabel | Arti |
|---|---|
| `persen` | Dictionary persentil k-distance (50, 90, 95, 99) |
| `grid` | DataFrame hasil grid search, termasuk `cluster_1` dan `cluster_2` (ukuran dua cluster terbesar) |
| `ukuran` | Daftar ukuran cluster (tanpa noise) pada satu kombinasi parameter, urut dari terbesar |
| `layak` | Subset `grid` yang memenuhi kriteria kelayakan |
| `terbaik` | Baris `layak` dengan Silhouette tertinggi (bila ada) |
| `MODE` | `'segmentasi'` bila ada konfigurasi layak, `'anomali'` bila tidak |
| `EPS`, `MIN_SAMPLES` | Parameter final. Pada mode anomali, `EPS` berasal dari persentil 95 |

Evaluasi dilindungi dengan pengaman:

```python
n_c = len(set(labels[m]))
if n_c > 1:
    print('Silhouette:', round(silhouette_score(Xs[m], labels[m]), 3))
    print('Davies-Bouldin:', round(davies_bouldin_score(Xs[m], labels[m]), 3))
else:
    print(f'Hanya {n_c} cluster, metrik tidak dapat dihitung')
```

### 8.5 Perbandingan FCM dan DBSCAN pada Marketing Campaign

| Aspek | FCM | DBSCAN |
|---|---|---|
| Hasil utama | 2 atau 3 segmen yang jelas | 1 cluster besar dan 61 noise |
| Kecocokan dengan data | Baik, data berupa sebaran menerus | Kurang, tidak ada celah kepadatan |
| Titik ambigu | 116 data pada $c=2$, 290 pada $c=3$ | Noise 61 data (mode anomali) |
| Parameter | $c$ dipilih dari metrik | $\varepsilon$ dan `min_samples` sangat sensitif |
| Kegunaan | Segmentasi pelanggan | Deteksi pelanggan tidak biasa |

Kesimpulan: pada dataset ini FCM adalah alat segmentasi yang tepat, sedangkan DBSCAN lebih berguna sebagai deteksi anomali. Temuan bahwa DBSCAN tidak menghasilkan segmen bermakna adalah hasil yang sah dan layak dilaporkan.

---

## 9. Kesalahan Umum dan Perbaikannya

Masalah berikut ditemukan saat meninjau kode untuk dataset Marketing Campaign. Semuanya mudah terulang saat dataset diganti.

| Gejala | Penyebab | Perbaikan |
|---|---|---|
| Semua kolom menyatu menjadi satu kolom | File dipisah tab, dibaca dengan pemisah koma | `pd.read_csv(..., sep='\t')` |
| `ValueError: Number of labels is 1` pada `silhouette_score` | DBSCAN hanya menemukan 1 cluster, parameter dibawa dari dataset lain | Pengaman `if n_c > 1`, dan pilih ulang parameter |
| `OSError: Cannot save file into a non-existent directory` | Folder keluaran belum dibuat | `os.makedirs(OUT_DIR, exist_ok=True)` sebelum `savefig` dan `to_csv` |
| Prediksi data baru identik dan tidak bermakna | Data uji memakai skala dataset lain (`[70, 80]`) | Pakai nilai berskala sama, misalnya persentil data sendiri |
| Jumlah cluster tidak sesuai tabel metrik | `C = 5` dibawa dari dataset lain padahal metrik memilih 2 | Tetapkan $c$ dari tabel metrik, judul plot pakai f-string `c={C}` |
| Garis `eps` pada k-distance jauh di atas kurva | Garis ditulis manual (`0.38`) | `plt.axhline(EPS, ...)` mengikuti variabel |
| Silhouette DBSCAN tinggi tetapi hasil tidak berguna | Noise dikeluarkan, cluster kedua hanya beberapa titik | Cek ukuran cluster dan jumlah noise lewat kriteria kelayakan |
| Scatter plot menumpuk | 2.212 titik dengan `s=40` | `s=15, alpha=0.6` |
| Iterasi FCM mencapai 150 | Batas `maxiter` tercapai sebelum konvergen | Naikkan `maxiter`, atau tafsirkan hasil dengan hati-hati |

---

## 10. Daftar File dan Cara Menjalankan

| File | Isi |
|---|---|
| `fcm_mall_customers.py` | FCM dari nol, metrik validasi, pemilihan c, visualisasi |
| `fcm_library_skfuzzy.py` | FCM dengan scikit-fuzzy, termasuk prediksi data baru |
| `dbscan_mall_customers.py` | DBSCAN dari nol dan verifikasi dengan scikit-learn |
| `dbscan_library.py` | DBSCAN hanya dengan scikit-learn (Mall Customers) |
| `preprocessing_marketing_campaign.py` | Persiapan data Marketing Campaign: pemuatan, fitur turunan, pembersihan, normalisasi |
| `dbscan_marketing_campaign.py` | DBSCAN Marketing Campaign: grid search skala baru, pemilihan parameter otomatis, mode anomali, pengaman evaluasi |
| `hasil_*.csv`, `hasil_*.png`, `kdistance_*.png` | Keluaran tabel dan grafik untuk Mall Customers |
| `Marketing_Campaign/` | Folder keluaran Marketing Campaign (`DBScan_lib.png`, `DBScan_Kdistance_lib.png`, `DBScan_lib.csv`, dan keluaran FCM) |

Instalasi:

```bash
pip install numpy pandas scikit-learn matplotlib scikit-fuzzy
```

Letakkan file CSV di folder yang sama dengan skrip (atau ubah `DATA_PATH` / argumen `pd.read_csv(...)` sesuai nama file). Ingat bahwa `marketing_campaign.csv` dipisah tab. Lalu jalankan:

```bash
# Mall Customers
python fcm_mall_customers.py
python fcm_library_skfuzzy.py
python dbscan_mall_customers.py
python dbscan_library.py

# Marketing Campaign
python dbscan_marketing_campaign.py
```

---

## 11. Catatan dan Keterbatasan

- **Inisialisasi acak FCM.** Hasil dapat sedikit berbeda antar `seed`. Gunakan `seed` tetap atau jalankan beberapa kali dan pilih $J$ terkecil.
- **Nomor cluster tidak baku.** Perbedaan nomor antara versi dari nol dan library bukan kesalahan. Bandingkan lewat profil rata-rata tiap cluster, bukan nomornya.
- **Sensitivitas DBSCAN.** Dengan `min_samples=5`, menaikkan `eps` dari 0,35 ke 0,37 menurunkan jumlah cluster dari 6 menjadi 4 karena kelompok kecil income rendah menyatu. Pemilihan eps = 0,38 dan min_samples = 6 didasarkan pada k-distance plot dan grid search pada data dua fitur.
- **Silhouette pada DBSCAN bias ke atas** bila noise banyak, karena noise dikeluarkan dari perhitungan.
- **Menambah fitur.** Bila `Age` ikut dipakai, parameter DBSCAN perlu dicari ulang karena jarak pada dimensi lebih tinggi bergeser, dan visualisasi memerlukan 3D atau PCA. `Gender` memerlukan pendekatan khusus (misalnya K-Prototype) karena bersifat kategorikal.
- **Interpretasi segmen** (misalnya "target utama") adalah tafsiran dari rata-rata fitur, bukan hasil yang dihitung algoritma.
- **Parameter tidak dapat dibawa antar dataset.** $c$ terbaik pada Mall adalah 5, pada Marketing Campaign adalah 2. $\varepsilon$ yang cocok pada Mall (0,38) menghasilkan satu cluster pada Marketing Campaign.
- **DBSCAN tidak cocok untuk data dengan kepadatan menerus.** Pada Marketing Campaign, pengujian grid, transformasi log, dan kombinasi fitur lain tidak menghasilkan segmen bermakna. Pengujian ini terbatas pada kombinasi yang dicoba.
- **$\varepsilon$ dari persentil bukan "tekukan" sejati.** Pada mode anomali, $\varepsilon$ = persentil 95 k-distance adalah aturan yang dapat dijelaskan, bukan satu-satunya nilai yang benar. Persentil lain menggeser jumlah noise.
- **Batas iterasi FCM.** Nilai `n_iter` yang sama dengan `maxiter` menandakan konvergensi belum tercapai pada `error` yang ditetapkan.
- **Data uji prediksi** harus berskala sama dengan data asli dan ditransformasi dengan `scaler` yang sama.
