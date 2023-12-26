# Laporan Proyek Machine Learning

### Nama : Satrio Mukti Prayoga

### Nim : 211351137

### Kelas : Malam B

## Domain Proyek

Aplikasi streamlit ini dapat membantu tenaga penjualan agar pemasaran mereka lebih tepat sasaran dengan menganalisis penggunaan kartu kredit setiap orang dalam kumpulan data yang akan digunakan. Dengan demikian, tenaga penjualan atau pemasar dapat mengetahui pelanggan mana yang lebih berpotensi untuk membeli produk mereka dengan melihat pengelompokan pelanggan pada visualisasi algoritma K-means. Segmentasi atau pengelompokan dapat membantu dalam menyusun strategi pemasaran yang lebih efektif berdasarkan perilaku pengguna kartu kredit.

## Business Understanding

### Problem Statements

- Bagaimana cara meningkatkan penjualan agar pemasaran tepat sasaran?

### Goals

- Membuat segmentasi pelanggan berdasarkan perilaku penggunaan kartu kredit mereka, dengan menggunakan algoritma KMeans, sehingga penjualan dapat meningkat secara signifikan.

### Solution statements

Solution Statement 1: Menggunakan algoritma machine learning KMeans Clustering untuk membuat segmentasi pelanggan. Membandingkan hasil clustering tersebut dan memilih yang terbaik berdasarkan silhouette score.

Solution Statement 2: Membuat plot dan barplot untuk segmentasi pelanggan.

## Data Understanding

Data yang digunakan dalam proyek ini berasal dari sumber [kaggle](https://www.kaggle.com/) dan berisi informasi tentang perilaku penggunaan kartu kredit.

<br>

Link Dataset : [Card Usage](https://www.kaggle.com/datasets/noordeen/card-usage/)

### Variabel-variabel pada Dataset ini adalah sebagai berikut:

- CUST_ID: Identifikasi pemegang kartu kredit (Kategorikal)
- BALANCE: Jumlah saldo yang tersisa di akun mereka untuk melakukan pembelian
- BALANCE_FREQUENCY: Seberapa sering Saldo diperbarui, skor antara 0 dan 1 (1 = diperbarui secara sering, 0 = tidak diperbarui secara sering)
- PURCHASES: Jumlah pembelian yang dibuat dari akun
- ONEOFF_PURCHASES: Jumlah pembelian maksimum yang dilakukan sekaligus
- INSTALLMENTS_PURCHASES: Jumlah pembelian yang dilakukan secara cicilan
- CASH_ADVANCE: Uang tunai yang diberikan oleh pengguna
- PURCHASES_FREQUENCY: Seberapa sering pembelian dilakukan, skor antara 0 dan 1 (1 = sering dibeli, 0 = tidak sering dibeli)
- ONEOFF_PURCHASES_FREQUENCY: Seberapa sering pembelian dilakukan sekaligus (1 = sering dibeli, 0 = tidak sering dibeli)
- PURCHASES_INSTALLMENTS_FREQUENCY: Seberapa sering pembelian dalam cicilan dilakukan (1 = sering dilakukan, 0 = tidak sering dilakukan)
- CASH_ADVANCE_FREQUENCY: Seberapa sering uang tunai di muka dibayar
- CASH_ADVANCE_TRX: Jumlah transaksi yang dilakukan dengan "Cash in Advanced"
- PURCHASES_TRX: Jumlah transaksi pembelian yang dilakukan
- CREDIT_LIMIT: Batas Kartu Kredit untuk pengguna
- PAYMENTS: Jumlah pembayaran yang dilakukan oleh pengguna
- MINIMUM_PAYMENTS: Jumlah pembayaran minimum yang dilakukan oleh pengguna
- PRC_FULL_PAYMENT: Persentase pembayaran penuh yang dilakukan oleh pengguna
- TENURE: Masa pelayanan kartu kredit untuk pengguna

## Data Preparation

Dataset "Card Usage" didapat dari website [kaggle](https://www.kaggle.com/)

disini saya akan mengkoneksikan google colab ke kaggle menggunakan token dari akun saya :

```bash
from google.colab import files
files.upload()
```

disni saya akan membuat direktori untuk menyimpan file kaggle.json

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

saya akan mendownload file datasetnya dari kaggle :

```bash
!kaggle datasets download -d noordeen/card-usage
```

disini saya mengekstrak file dari dataset yang sudah saya download :

```bash
!unzip card-usage.zip
```

lalu saya akan membuat EDA, dengan menggunakan beberapa library, pertama saya akan import beberapa library yang akan dipakai :

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")
```

Disini saya akan memanggil dan menyimpan dataset di variabel df dengan menggunakan kode sebagai berikut :

```bash
df = pd.read_csv('CreditCardUsage.csv')
```

Setelah dataset disimpan di variabel df, saya akan melihat jumlah baris data dan featurenya.

```bash
df.shape
```

Saya juga melihat ada semua featurenya

```bash
df.columns
```

Setelah saya cek feature yang ada, saya juga ingin mengetahui 5 baris data awal pada setiap featurenya.

```bash
df.head()
```
