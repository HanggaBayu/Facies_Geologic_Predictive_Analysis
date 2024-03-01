# Laporan Proyek Machine Learning - Hangga Bayu Krisna
Proyek pertama predictive analytics untuk memenuhi submission dicoding
**Topik dari proyek ini adalah Facies Classification Using Various Machine Learning Methods**

## Domain Proyek
### Latar Belakang
Perkembangan teknologi yang begitu pesat membuat seorang peneliti dituntut untuk selalu beradaptasi terhadap hal tersebut tak terkecuali dalam dunia _oil and gas_. Dalam proses eksplorasi khususnya dalam tahap identifikasi reservoar. Dalam identifikasi reservoar ini diperlukan beberapa studi pendukung yang begitu krusial seperti salah satunya adalah identifikasi fasies, identifikasi ini meliputi pengelompokkan fasies berdasarkan jenis dan ketebalannya. Dengan adanya pengelompokkan data fasies ini diharapkan dapat digunakan untuk memperjelas perkiraan dari ketebalan reservoir, memahami sifat-sifat fisik reservoir dan memberikan saran dalam pelaksanaan pengeboran.Kemudian nantinya, pemahaman yang baik terhadap fasies memungkinkan identifikasi zona-zona yang memiliki potensi produksi hidrokarbon yang tinggi sehingga dapat memaksimalkan produksi dan meminimalkan risiko kegagalan pemboaran minyak dan gas bumi.Oleh sebab itu, proyek ini menginisiasi adanya sistem pemrograman python berbasis _machine learning_ untuk mengelompokkan data pengeboran  dalam hal  geologi (facies).

## Business Understanding

### Problem Statements
Permasalahan yang diselesaikan oleh proyek ini adalah:

- Dari beberapa fitur/log data, maka data apa yang paling berpengaruh terhadap kelas fasies?
- Bagaimana _best practice_ (alur kerja) dalam pengerjaan _machine learning_ untuk melakukan klasifikasi fasies batuan?
- Bagaimana hasil prediksi dapat digunakan dalam analisis kualitas reservoir yang berpotensi untuk dilakukan eksploitas/produksi?


### Goals
Tujuan yang ingin dicapai adalah:
- Mengetahui fitur yang paling berkorelasi dengan suatu kelas fasies.
- Membuat model _machine learning_ yang dapat mengklasifikasi fasies batuan dengan akurat.
- Hasil prediksi cukup valid dalam membantu analisis kualitas reservoir untuk dilakukannya produksi

### Solution statements
- Melakukan _exploratory data analyst_ baik _univariate_ dan _multivariate_ untuk melihat hubungan antar fitur, serta melakukan pengecekan _outlier_
- Mengajukan 2 atau lebih algoritma machine learning untuk dibandingkan hasil akurasinya
- Melakukan _searching_ parameter terbaik pada tiap algoritma untuk mendapatkan hasil pelatihan yang terbaik

## Data Understanding
Perkembangan teknologi yang begitu pesat membuat seorang peneliti dituntut untuk selalu beradaptasi terhadap hal tersebut tak terkecuali dalam dunia oil and gas. Dalam proses eksplorasi khususnya dalam tahap identifikasi reservoar. Dalam identifikasi reservoar ini diperlukan beberapa studi pendukung yang begitu krusial seperti identifikasi fasies, identifikasi ini meliputi pengelompokkan fasies berdasarkan jenis dan ketebalannya. Dengan adanya pengelompokkan data fasies ini diharapkan dapat digunakan untuk memperjelas perkiraan dari ketebalan reservoir dan memberikan saran dalam pelaksanaan pengeboran. 

Data yang digunakan bersumber dari SEG Competition (Hall, 2016) https://github.com/seg/2016-ml-contest yang diinisiasi oleh Hall. Dataset welllog facies_vector.csv yang digunakan oleh penulis. Data ini berisi 3232 entri dengan setiap entri terdiri dari 5 pengukuran log wireline, 2 variabel indikator, dan kedalaman pengukuran, dengan sebanyak 8 sumur di Lapangan Hugoton Barat Daya Kansas (Dubois et al., 2007). Variabel/fitur penting pada data dijelaskan sebagai berikut:

- Data Facies,
Data facies merupakan suatu data  satuan  batuan yang dapat dibedakan mulai dari litologi, struktur, dan geometri nya 
- Depth,
Depth atau  kedalaman merupakan data kedalaman secara realtime pada saat alat log dimasukkan ke dalam sumur.
- Formation
Merupakan data dari formasi batuan,  dalam data yang akan diolah dinyatakan dalam bentuk kode seperti A1 SH , C LM dll.
- Well Name,
Merupakan  data  nama dari sumur yang dimasuki oleh alat log
- GR ,
GR (Log Gamma Ray) Log Gamma Ray merupakan data yang diambil dari alat radioaktif berupa tingkat  radioaktivitas alami dari suatu batuan. 
- ILD_Log 10,
Merupakan data log yang terekam berdasarkan rekaman resistivitas batuan oleh alat log 
- DeltaPhi,
Data ini didapatkan dari hasil pengamatan radioaktif melalui pemancaran neutron secara kontinyu oleh alat ke dalam suatu formasi batuan.
- PHIND,
Data ini menunjukkan kandungan hidrogen dalam sebuah formasi. Nilai ini juga menunjukkan seberapa banyak fluida yang mengisi porositas batuan.
- PE,
Log photoelectric merupakan log yang mengukur faktor serapan fotoelektrik dari sebuah sumur.
- NM_M,
NM_M atau non-marine indicator merupakan data yang menunjukkan keterangan sebuah satuan litologi merupakan anggota batuan marine atau tidak.
- RELPOS,
Relpos atau Relative Position merupakan data yang menunjukkan posisi lapisan batuan pada sumur

Kelas-kelas fasies batuan yang terdapat pada data adalah:

- (SS) Nonmarine sandstone

- (CSiS) Nonmarine coarse siltstone

- (FSiS) Nonmarine fine siltstone

- (SiSH) Marine siltstone and shale

- (MS) Mudstone (limestone)

- (WS) Wackestone (limestone)

- (D) Dolomite

- (PS) Packstone-grainstone (limestone)

- (BS) Phylloid-algal bafflestone (limeston

### Exploratory Data Analyst
Telaah data (EDA) merupakan bagian dari data understanding (pemahaman data) yang secara umum terdiri: Pengecekan data dengan beragai visualisasi untuk mendapatkan informasi, _checking outlier_, _univariate analysis_, dan _multivariate analysis_
#### Pengecekan Data

- Mengecek jumlah pengukuran/entri pada masing-masing sumur

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/830f1a42-13a9-4ba6-bcd8-0d027a18b2a5)

Gambar 1. jumlah pengukuran pada masing-masing sumur

Pada Gambar 1 dapat dilihat jumlah pengukuran pada masing-masing sumur. Sumur yang memiliki data paling banyak adalah CROSS H CATTLE dengan jumlah sebesar 500 data, dan yang paling sedikit adalah Recruit F9 dengan jumlah sebesar 68.

- Melihat persebaran data fasies pada semua sumur

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/efca74b4-6ab7-4a50-9882-025a9adff346)

Gambar 2. Persebaran data fasies pada dataset

Gambar 2 menunjukkan fasies batuan paling banyak adalah fasies CSis sebanyak 738, dan paling sedikit adalah fasies D sebanyak 38

- Menampilkan plot data log pada sample sumur NEWBY

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/a641d113-46f9-4e3d-ba47-e97857d85b5d)

Gambar 3. Ploting data log dan fasies pada sumur NEWBY


_Plotting_ data pada Gambar 3 dilakukan untuk melihat korelasi antar fitur dengan label fasies terkait. Plot dilakukan pada sumur NEWBY

#### Mengecek Outlier (Pencilan)
_Checking_ dilakukan pada data-data numerik

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/6b976d3b-7593-44cc-8c77-d69ec8300dd4)

Gambar 4. Outlier pada data-data numerik

Pada Gambar 4 dapat dilihat terdapat indikasi pencilan pada fitur-fitur numerik, seperti GR, ILD_log10, DeltaPHI, PHIND, dan PE. Sementara pada fitur RELPOS tidak terindikasi adanya pencilan

#### Univariate Analysis
Ditampilkan pada beberapa fitur masukan:

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/daa18d63-a203-4d7d-8492-328be18a59d3)

Gambar 5. Univariate Analysis pada fitur numerik

Pada Gambar 5 dapat melihat distribusi pada pada masing-masing fitur numerik. Hal ini membantu memahami apakah data tersebut terdistribusi normal atau memiliki pola tertentu.

#### Multivariate Analysis
![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/5313bfe9-5793-4c41-9a37-60cad231e6a3)

Gambar 6. Multivariate Analysis pada fitur numerik 

Melalui visualsiasi pada Gambar 6 pemahaman terhadap struktur dan kompleksitas data diperoleh dengan lebih baik, yang dapat mendukung pengambilan keputusan yang lebih informasional dan berbasis bukti.


#### Correlation Matrix
![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/18d6ec6a-b17e-46bf-b50a-bfc46a2200df)

Gambar 7. Matriks Korelasi pada fitur numerik

Pada matriks korelasi seperti yang ditunjukkan oleh Gambar 7 kita dapat melihat korelasi positi dan negatif antar fitur. Selanjutnya karena tujuan kita adalah mengklasifikasi fasies, maka dapat kita lihat fitur-fitur yang secara signifikan berpengaruh terhadap fasies, yaiut NM_M, PE, dan GR yang berada di atas 0.5.

## Data Preparation
Dalam data preparation dilakukan beberapa proses penting yang mencakup:
1. Feature Augmentation dan Splitting Data
   
Feature Augmentation memiliki hasil akhirny, yaitu matriks fitur yang lebih besar dengan informasi tambahan dari jendela tetangga dan gradien. Selanjutnya data dibagi, menjadi 2 terlebih dahulu yaitu data untuk training (train dan test) (data selain sumur NEWBY), kemudian data untuk validasi (sumur NEWBY). Selanjutnya pada data training dilakukan konsep pembagian data menjadi dua bagian: data pelatihan (training data) dan data pengujian (testing data) yang digunakan dalam proses pelatihan.

Proses augmentasi dilakukan sehingga data dari yang sebelumnya memiliki 7 fitur, setelah augmentasi data memiliki 28 fitur , sehingga fitur yang digunakan dalam proses pelatihan semakin banyak.

2. Standarisasi
   
Suatu proses transformasi data agar memiliki rata-rata (mean) nol dan deviasi standar (standard deviation) satu


## Modeling
Selanjutnya pada proses ini, sebelum dilakukan percobaan pada beberapa algoritma dilakukan terlebih dahulu GridSearchCV ataupun RandomizedSearchCV untuk memperoleh algortima dengan parameter terbaik. Kedua metode tersebut dijelaskan sebagai berikut:

- GridSearchCV

GridSearchCV melakukan pencarian parameter dengan cara yang sistematis, dengan mencoba semua kombinasi nilai parameter yang telah ditentukan sebelumnya.
Algoritma menciptakan "grid" dari semua kemungkinan kombinasi parameter dan melakukan cross-validation untuk mengevaluasi performa model pada setiap kombinasi.

- RandomizedSearchCV:

RandomizedSearchCV melakukan pencarian parameter dengan cara yang acak, dengan mencoba sejumlah kombinasi parameter yang dipilih secara acak dari ruang parameter yang diberikan.
Jumlah iterasi pencarian parameter dapat diatur sebelumnya. Metode ini ebih efisien secara komputasional karena tidak mencoba semua kombinasi parameter sehingga cocok untuk ruang parameter yang besar

Algoritma-algoritma machine learning yang dilakukan dalam tugas kali ini adalah:

- Model (model): KNeighborsClassifier

KNN adalah algoritma klasifikasi berbasis instan yang menggunakan mayoritas kelas tetangga terdekat untuk memprediksi kelas data yang tidak diketahui.

Parameter yang Dicari (params): Jumlah tetangga (n_neighbors) dan jenis bobot tetangga (weights).

n_neighbors: Jumlah tetangga. Ini adalah jumlah tetangga terdekat yang akan digunakan oleh algoritma untuk menentukan kelas prediksi suatu titik data.

weights: Jenis bobot tetangga. Ini menentukan cara memberikan bobot kepada tetangga dalam perhitungan. Ada dua opsi umum:

'uniform': Semua tetangga memiliki bobot yang sama. Ini cocok digunakan ketika semua tetangga dianggap memiliki kontribusi yang setara.

'distance': Bobot inversi jarak. Tetangga yang lebih dekat memiliki pengaruh yang lebih besar daripada yang lebih jauh. Ini dapat berguna jika beberapa tetangga lebih relevan daripada yang lain dalam membuat prediksi


- Model (model): RandomForestClassifier
  
Random Forest adalah algoritma ensemble yang menggunakan beberapa pohon keputusan untuk membuat prediksi dan mengurangi overfitting.

Parameter yang Dicari (params): Kedalaman maksimum pohon (max_depth), jumlah pohon (n_estimators), dan nilai seed acak (random_state).

max_depth: Kedalaman maksimum pohon. Ini menentukan seberapa dalam setiap pohon keputusan dapat tumbuh. Pengaturan nilai yang terlalu tinggi dapat menyebabkan overfitting, sementara nilai yang terlalu rendah dapat menyebabkan model menjadi terlalu sederhana.

n_estimators: Jumlah pohon dalam ensemble. Semakin banyak pohon, semakin baik modelnya, tetapi ada batasan pada peningkatan kinerja dengan penambahan pohon. Ini karena setiap pohon tambahan menyumbang kurang dan kurang untuk meningkatkan performa keseluruhan.

random_state: Nilai seed acak. Ini memastikan reproduktibilitas model, artinya jika Anda menjalankan model dengan seed yang sama, Anda akan mendapatkan hasil yang sama. Ini penting untuk keperluan pembandingan dan reproduksi hasil


- Model (model): SVC (Support Vector Classification)
  
SVM adalah algoritma klasifikasi yang berusaha menemukan hyperplane terbaik untuk memisahkan kelas yang berbeda dalam ruang fitur.

Parameter yang Dicari (params): Parameter regularisasi (C) dan parameter kernel (gamma).

C: Parameter regularisasi. Ini mengontrol sejauh mana model mengenali dan memperhatikan setiap titik data pelatihan.
gamma: Parameter kernel. Ini mengontrol pengaruh titik data tunggal dan seberapa jauh dampaknya.

- Model (model): AdaBoostClassifier
  
AdaBoost adalah algoritma boosting yang menggabungkan beberapa model lemah untuk membuat model kuat.

Parameter yang Dicari (params): Tingkat pembelajaran (learning_rate), jumlah estimator (n_estimators), dan nilai seed acak (random_state).

learning_rate: Tingkat pembelajaran. Menentukan sejauh mana model belajar dari setiap model lemah yang ditambahkan

n_estimators: Jumlah estimator. Seperti pada RandomForestClassifier, ini menentukan berapa banyak model lemah (biasanya pohon keputusan dangkal) yang digunakan dalam ensemble.

random_state: Nilai seed acak. Sama seperti dalam RandomForestClassifier, ini untuk memastikan reproduktibilitas hasil.

- Model (model): XGBClassifier

XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang efisien dan kuat, sering digunakan dalam kompetisi data science.

Parameter yang Dicari (params): Kedalaman maksimum pohon (max_depth), tingkat pembelajaran (learning_rate), jumlah pohon (n_estimators), dan subsample ratio dari dataset (subsample).

max_depth: Kedalaman maksimum pohon. Seperti pada RandomForestClassifier, ini mengontrol kompleksitas pohon dan dapat membantu mencegah overfitting.

learning_rate: Tingkat pembelajaran. Sama seperti pada AdaBoost, menentukan sejauh mana model belajar dari setiap model tambahan.

n_estimators: Jumlah pohon. Jumlah pohon dalam ensemble, mirip dengan RandomForestClassifier dan AdaBoostClassifier.

subsample: Subsample ratio dari dataset. Ini adalah fraksi dari total data yang digunakan untuk melatih setiap pohon. Memilih nilai kurang dari 1.0 dapat membantu mengurangi overfitting.

Berikut adalah parameter terbaik dari masing-masing model

Tabel 1. Parameter terbaik masing-masing model

|     Algorithm | Best Parameters                                                               |
|--------------:|-------------------------------------------------------------------------------|
|           knn |                                     {'n_neighbors': 6, 'weights': 'distance'} |
| random_forest |                    {'random_state': 11, 'n_estimators': 1000, 'max_depth': 4} |
|           svm |                                                       {'gamma': 0.1, 'C': 10} |
|      AdaBoost |               {'random_state': 11, 'n_estimators': 25, 'learning_rate': 0.05} |
|       XGBoost | {'subsample': 0.6, 'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1} |


## Evaluation
Metrik evaluasi diterapkan pada masing-masing algoritma yang meliputi:

- Precision

Definisi: Precision mengukur seberapa banyak dari prediksi positif model yang benar-benar positif. Dihitung sebagai rasio antara True Positives (TP) dan total prediksi positif (TP + False Positives). Precision memberikan informasi tentang seberapa presisi model dalam mengklasifikasikan instans sebagai positif.

- Recall (Sensitivitas atau True Positive Rate):

Definisi: Recall mengukur seberapa banyak dari total instance yang sebenarnya positif yang berhasil ditemukan oleh model. Dihitung sebagai rasio antara True Positives (TP) dan total positif sebenarnya (TP + False Negatives). Recall memberikan informasi tentang seberapa baik model dapat menangkap semua instance positif yang sebenarnya.

- F1-Score

Definisi: F1-score adalah metrik yang menggabungkan precision dan recall menjadi satu nilai tunggal. Ini berguna ketika ingin mencari keseimbangan antara precision dan recall. F1-score memberikan informasi holistik tentang kinerja model dengan mempertimbangkan kedua aspek precision dan recall.

- Support

Definisi: Support adalah jumlah instance yang termasuk dalam suatu kelas pada data pengujian. Ini memberikan konteks tentang seberapa umum suatu kelas dalam dataset. Support tidak digunakan untuk mengukur kinerja model secara langsung, tetapi memberikan gambaran tentang seberapa besar dampak dari kelas tersebut pada evaluasi model.


Tabel 2. Metrik Akurasi Pada Algoritma XGBoost


|              | Precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| SS           | 0.82      | 0.96   | 0.88     | 24      |
| CSiS         | 0.73      | 0.83   | 0.78     | 59      |
| FSiS         | 0.79      | 0.58   | 0.67     | 38      |
| SiSh         | 0.75      | 0.30   | 0.43     | 10      |
| MS           | 0.62      | 0.76   | 0.68     | 17      |
| WS           | 0.61      | 0.61   | 0.61     | 36      |
| D            | 0.71      | 0.71   | 0.71     | 7       |
| PS           | 0.71      | 0.71   | 0.71     | 41      |
| BS           | 0.90      | 0.90   | 0.90     | 10      |
| accuracy     |           |        | 0.72     | 242     |
| macro avg    | 0.74      | 0.71   | 0.71     | 242     |
| weighted avg | 0.73      | 0.72   | 0.72     | 242     |

Mengambil hasil metrik akurasi pada algoritma XGBoost sebagai algoritma terbaik. dapat dilihat hasil akurasi sudah cukup baik yaitu sebesar 0.8 

Kemudian diperoleh hasil _accuracy_ pada data training dan juga testing pada masing-masing model:

Tabel 3. Hasil akurasi pada proses _training_

|           |   KNN   |  RandomForest |    SVM   | Boosting | XGBoost |
|:---------:|:-------:|:-------------:|:--------:|:--------:|:-------:|
| train_acc |   1.0   |    0.642726   |    1.0   |  0.47744 |   1.0   |
|  test_acc | 0.72314 |    0.607438   | 0.739669 | 0.458678 | 0.80165 |


![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/359f24c5-d859-4e6e-b335-f13a8f0dacf3)

Gambar 8. Perbandingan hasil akurasi pada tiap-tiap model

Dari Gambar 8 tersebut dapat dilihat bahwa algoritma terbaik adalah XGBoost, diikuti oleh SVM, KNN, RandomForest, dan AdaBoost.
Sehingga proses prediksi selanjutnya dilakukan menggunakan 3 algoritma terbaik yaitu, XGBoost, SVM, dan KNN dengan nilai akurasi sebesar 0.8, 0.73, dan 0.72 secara berurutan.


#### Hasil Prediksi pada Data NEWBY (Sumur Validasi)
- Prediksi dengan SVM
  
  ![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/f7f74b4a-8b43-48ff-906a-4d48e45e3cee)

Gambar 9. Hasil prediksi dengan algoritma SVM

 - Prediksi dengan KNN

![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/46115236-2aba-4344-9f28-4eb932e12431)

Gambar 10. Hasil prediksi dengan algoritma KNN

- Prediksi dengan XGBoost
![image](https://github.com/HanggaBayu/Daftar-Biodata-Siswa/assets/99377476/7ede7565-044a-4e35-bb03-c654cbaa9e4b)

Gambar 11. Hasil prediksi dengan algoritma XGBoost

Gambar 9, Gambar 10, dan Gambar 11 menunjukkan hasil prediksi terhadap label pada masing-masing algoritma terbaik. Dapat dilihat dari ketiganya sudah menunjukkan tren prediksi yang cukup baik

# PENUTUP
Hasil prediksi dari 3 algoritma terbaik sudah dapat memetakan fasies dengan cukup baik, sehingga algoritma-algoritma tersebut menjadi pilihan untuk digunakan dalam proses analisis fasies dalam rangka mengkarakterisasi reservoir yang memiliki potensi mengandung hidrokarbon, tentunya analisis ini harus juga disertai dengan _geologist_ agar dapat mmeberikan interpretasi yang valid dan benar baik secara praktikal dan teoretikal

### REFERENSI

Chen, J. and Zeng, Y., 2018. Application of machine learning in rock facies classification with physics-motivated feature augmentation. arXiv preprint arXiv:1808.09856.

Dubois, M., G. Bohling, and S. Chakrabarti, 2007, Comparison of four approaches to a rock facies classification problem: Computers and Geosciences, 33, 599-617.

Hall, B., 2016, Facies classification using machine learning: The Leading Edge, 35, 906-909.
