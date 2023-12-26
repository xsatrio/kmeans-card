import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# Suppress the deprecated warning
st.set_option('deprecation.showPyplotGlobalUse', False)

file_path = "CreditCardUsage.csv"
df = pd.read_csv(file_path)

# Sidebar
st.sidebar.title("Pengaturan Klaster")
num_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)
st.sidebar.markdown("---")

# Konten utama
st.title("Segmentasi Pelanggan Berdasarkan Penggunaan Kartu Kredit")
st.write("#### Nama : Satrio Mukti Prayoga \n#### NIM : 211351137 \n#### Malam B")
st.markdown("----")

st.write("Pada aplikasi streamlit kali ini saya ingin membantu para sales atau orang marketing, supaya pemasaran mereka tepat sasaran dengan cara mengetahui penggunaan kartu kredit pada setiap orang yang ada pada dataset yang akan saya gunakan, sehingga para sales atau orang marketing bisa mengetahui pelanggan mana, yang akan membeli produk mereka, dengan cara melihat cluster pelanggan pada visualisasi algoritma Kmeans. Segmentasi atau clustering dapat membantu dalam menyusun strategi marketing yang lebih efektif berdasarkan perilaku pengguna kartu kredit.")
st.markdown("----")


st.write("### Data Asli :")
st.write(df)

# Preprocessing
df.drop(['CUST_ID'], axis=1, inplace=True)

df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)

# Modeling
X = df.values[:]
X = np.nan_to_num(X)
scaled_X = StandardScaler().fit_transform(X)

# Menampilkan plot Elbow Method
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i).fit(scaled_X)
    inertia_values.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), inertia_values, marker='o')
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
st.sidebar.pyplot(fig)

kmeans = KMeans(n_clusters = num_clusters).fit(scaled_X)
print(kmeans.cluster_centers_)
kmeans_labels = kmeans.labels_

clusters_kmeans = kmeans.labels_ + 1
df["cluster"] = clusters_kmeans

cluster_labels = kmeans.labels_
silhouette_avg = silhouette_score(scaled_X, cluster_labels)

st.write("### Data Setelah Clustering : ")
st.write(f"Penggunaan num clusters {num_clusters}, mendapatkan nilai Silhouette Score sebesar {silhouette_avg:.2f}.")
st.write(df)

st.write("### Jumlah total data setiap cluster : ")
jumlah_data_per_cluster = df['cluster'].value_counts()
st.write(jumlah_data_per_cluster)

st.markdown("----")

# Visualisasi Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 0], y=scaled_X[:, 2], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

###
st.write("### Cluster Plot BALANCE & PURCHASES")
st.write("Scatter plot dibawah ini dapat membantu mengidentifikasi apakah pengguna dengan saldo yang tinggi cenderung melakukan pembelian dalam jumlah besar atau sebaliknya.")

plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 0], y=scaled_X[:, 2], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

for label in np.unique(clusters_kmeans):
    plt.annotate(label,
                 (scaled_X[clusters_kmeans == label, 0].mean(),
                  scaled_X[clusters_kmeans == label, 2].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.xlabel('BALANCE')
plt.ylabel('PURCHASES')
plt.title('Cluster Plot with Cluster Centers')
plt.legend()
st.pyplot()

df['Cluster'] = clusters_kmeans

cluster_stats = df.groupby('Cluster')[['BALANCE', 'PURCHASES']].sum()

cluster_stats.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Total')
plt.title('Total Balance and Purchases for Each Cluster')
st.pyplot()

st.markdown("----")

###
st.write("### Cluster Plot PURCHASES_FREQUENCY & ONEOFF_PURCHASES_FREQUENCY")
st.write("Scatter plot diabawah ini dapat membantu memahami hubungan antara jenis pembelian dan frekuensinya.")

plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 6], y=scaled_X[:, 7], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

for label in np.unique(clusters_kmeans):
    plt.annotate(label,
                 (scaled_X[clusters_kmeans == label, 6].mean(),
                  scaled_X[clusters_kmeans == label, 7].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.xlabel('PURCHASES_FREQUENCY')
plt.ylabel('ONEOFF_PURCHASES_FREQUENCY')
plt.title('Cluster Plot with Cluster Centers')
plt.legend()
st.pyplot()

cluster_stats_freq = df.groupby('Cluster')[['PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY']].mean()

cluster_stats_freq.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Mean Frequency')
plt.title('Mean Purchase Frequencies for Each Cluster')
st.pyplot()

st.markdown("----")

###
st.write("### Cluster Plot CASH_ADVANCE &PURCHASES")
st.write("Scatter plot dibawah ini dapat membantu melihat apakah pengguna yang sering mengambil uang tunai cenderung melakukan pembelian dalam jumlah besar atau sebaliknya.")

plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 5], y=scaled_X[:, 2], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

for label in np.unique(clusters_kmeans):
    plt.annotate(label,
                 (scaled_X[clusters_kmeans == label, 5].mean(),
                  scaled_X[clusters_kmeans == label, 2].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.xlabel('CASH_ADVANCE')
plt.ylabel('PURCHASES')
plt.title('Cluster Plot with Cluster Centers')
plt.legend()
st.pyplot()

cluster_stats_cash_purchase = df.groupby('Cluster')[['CASH_ADVANCE', 'PURCHASES']].mean()

cluster_stats_cash_purchase.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Mean Amount')
plt.title('Mean Cash Advance and Purchases for Each Cluster')
st.pyplot()

st.markdown("----")

###
st.write("### Cluster Plot INSTALLMENTS_PURCHASES & PURCHASES_INSTALLMENTS_FREQUENCY")
st.write("Scatter plot dibawah ini dapat membantu melihat apakah pengguna yang sering melakukan pembelian dalam bentuk cicilan juga melakukan pembelian cicilan dalam jumlah besar atau sebaliknya.")

plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 4], y=scaled_X[:, 8], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

for label in np.unique(clusters_kmeans):
    plt.annotate(label,
                 (scaled_X[clusters_kmeans == label, 4].mean(),
                  scaled_X[clusters_kmeans == label, 8].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.xlabel('INSTALLMENTS_PURCHASES')
plt.ylabel('PURCHASES_INSTALLMENTS_FREQUENCY')
plt.title('Cluster Plot with Cluster Centers')
plt.legend()
st.pyplot()

st.markdown("----")

###
st.write("### Cluster Plot BALANCE_FREQUENCY & PURCHASES_FREQUENCY")
st.write("Scatter plot dibawah ini dapat membantu melihat apakah ada korelasi antara keaktifan dalam menggunakan kartu kredit dan frekuensi pembelian.")

plt.figure(figsize=(10, 8))
sns.scatterplot(x=scaled_X[:, 1], y=scaled_X[:, 6], hue=clusters_kmeans, palette=sns.color_palette('hls', n_colors=len(np.unique(clusters_kmeans))), marker='o', s=50)

for label in np.unique(clusters_kmeans):
    plt.annotate(label,
                 (scaled_X[clusters_kmeans == label, 1].mean(),
                  scaled_X[clusters_kmeans == label, 6].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.xlabel('BALANCE_FREQUENCY')
plt.ylabel('PURCHASES_FREQUENCY')
plt.title('Cluster Plot with Cluster Centers')
plt.legend()
st.pyplot()

cluster_stats_balance_purchase = df.groupby('Cluster')[['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY']].mean()

cluster_stats_balance_purchase.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Mean Frequency')
plt.title('Mean Balance Frequency and Purchases Frequency for Each Cluster')
st.pyplot()

st.markdown("----")
