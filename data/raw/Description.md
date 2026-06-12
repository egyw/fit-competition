# Track 1 – Flood Prediction

## Overview

Dataset ini berisi data faktor-faktor risiko banjir dari berbagai wilayah, mencakup aspek iklim, geografis, lingkungan, infrastruktur, dan sosial. Tujuan utamanya adalah memprediksi **probabilitas terjadinya banjir** di suatu wilayah berdasarkan kombinasi faktor-faktor tersebut.

| Info | Detail |
|------|--------|
| **File** | `flood.csv` |
| **Jumlah Baris** | 50.000 |
| **Jumlah Kolom** | 21 (20 fitur + 1 target) |
| **Missing Values** | Tidak ada |
| **Task** | Regression |
| **Target** | `FloodProbability` (0.285 – 0.725) |

---

## Target Variable

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `FloodProbability` | `float64` | Probabilitas terjadinya banjir di suatu wilayah. Nilai mendekati **1.0** menunjukkan risiko banjir yang **tinggi**, sedangkan nilai mendekati **0.0** menunjukkan risiko yang **rendah**. |

---

## Feature Columns

Semua kolom fitur bertipe `int64` dengan rentang nilai **0 hingga ~16–19** (skor risiko). Semakin tinggi nilainya, semakin buruk atau berisiko kondisi faktor tersebut.

### 🌧️ Faktor Iklim & Hidrologi

| Kolom | Deskripsi |
|-------|-----------|
| `MonsoonIntensity` | Intensitas curah hujan musim hujan (monsun). Nilai tinggi mencerminkan curah hujan yang ekstrem dan berkontribusi langsung pada risiko banjir. |
| `ClimateChange` | Tingkat dampak perubahan iklim di wilayah tersebut, seperti meningkatnya frekuensi cuaca ekstrem dan anomali musim. |
| `Watersheds` | Kondisi pengelolaan daerah aliran sungai (DAS). Pengelolaan DAS yang buruk memperlambat penyerapan air dan meningkatkan limpasan permukaan. |

### 🏔️ Faktor Geografis & Lahan

| Kolom | Deskripsi |
|-------|-----------|
| `TopographyDrainage` | Kemampuan kontur topografi wilayah dalam mengalirkan air. Dataran rendah atau cekungan memiliki skor tinggi karena lebih rentan terhadap genangan. |
| `Landslides` | Risiko tanah longsor yang dapat menghalangi aliran sungai atau menyebabkan banjir bandang di daerah pegunungan. |
| `CoastalVulnerability` | Kerentanan wilayah pesisir terhadap banjir rob, gelombang pasang, atau naiknya permukaan laut. |
| `WetlandLoss` | Tingkat kehilangan lahan basah (rawa, mangrove). Lahan basah berfungsi sebagai penyangga alami yang menyerap kelebihan air. |
| `Siltation` | Tingkat sedimentasi atau endapan lumpur di sungai dan saluran air, yang memperkecil kapasitas tampung dan memperlambat aliran. |

### 🌲 Faktor Lingkungan & Penggunaan Lahan

| Kolom | Deskripsi |
|-------|-----------|
| `Deforestation` | Tingkat deforestasi atau penggundulan hutan. Hutan yang hilang mengurangi penyerapan air dan meningkatkan erosi tanah. |
| `Urbanization` | Tingkat urbanisasi wilayah. Permukaan beton dan aspal yang luas mengurangi infiltrasi air ke dalam tanah. |
| `AgriculturalPractices` | Kualitas praktik pertanian. Irigasi berlebihan, penggunaan pestisida, atau konversi lahan dapat memperparah limpasan air. |
| `Encroachments` | Tingkat pelanggaran tata ruang seperti pembangunan di bantaran sungai, daerah resapan air, atau sempadan danau. |

### 🏗️ Faktor Infrastruktur

| Kolom | Deskripsi |
|-------|-----------|
| `RiverManagement` | Kualitas pengelolaan sungai, mencakup normalisasi sungai, pembangunan tanggul, dan pengendalian debit aliran. |
| `DamsQuality` | Kondisi dan kualitas bendungan. Bendungan yang sudah tua atau kurang terawat berisiko jebol saat debit air tinggi. |
| `DrainageSystems` | Kualitas sistem drainase perkotaan. Saluran yang tersumbat atau kapasitasnya kecil memperparah banjir saat hujan deras. |
| `DeterioratingInfrastructure` | Kondisi infrastruktur secara umum (jembatan, kanal, tanggul) yang mengalami penurunan fungsi akibat kurangnya perawatan. |

### 👥 Faktor Sosial & Manajemen

| Kolom | Deskripsi |
|-------|-----------|
| `IneffectiveDisasterPreparedness` | Tingkat ketidaksiapan menghadapi bencana, termasuk lemahnya sistem peringatan dini, rencana evakuasi, dan respons darurat. |
| `PopulationScore` | Tekanan kepadatan populasi terhadap wilayah rawan banjir. Populasi padat di daerah berisiko memperburuk dampak banjir. |
| `InadequatePlanning` | Buruknya perencanaan tata kota dan tata wilayah yang tidak mempertimbangkan risiko banjir dalam penetapan zonasi. |
| `PoliticalFactors` | Pengaruh faktor politik seperti korupsi, lemahnya kebijakan mitigasi, atau kurangnya anggaran penanggulangan banjir. |

---

## Distribusi Data

```
FloodProbability:
  Min    : 0.285
  Max    : 0.725
  Mean   : 0.500
  Std    : 0.050
  Median : 0.500

Semua fitur:
  Rentang nilai : 0 – ~16 hingga 22 (tergantung kolom)
  Rata-rata     : ~5.0 per fitur
  Missing values: 0
```

---

## Catatan

- Dataset ini kemungkinan bersifat **sintetis** (dibuat secara komputasional), ditandai dengan distribusi fitur yang sangat seragam (rata-rata ~5 untuk semua fitur) dan tidak adanya missing values.
- Semua fitur menggunakan **skala skor ordinal** (bilangan bulat), bukan nilai pengukuran langsung.
- Task yang tepat adalah **regression** karena target `FloodProbability` bersifat kontinu, bukan kategorikal.

---

## Struktur File

```
Track 1/
└── Flood Prediction/
    └── Flood Prediction/
        └── flood.csv
```

---

## Contoh Penggunaan

```python
import pandas as pd

df = pd.read_csv("flood.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["FloodProbability"])
y = df["FloodProbability"]

print(f"Features : {X.shape[1]} kolom")
print(f"Samples  : {X.shape[0]} baris")
print(f"Target   : {y.min():.3f} – {y.max():.3f}")
```