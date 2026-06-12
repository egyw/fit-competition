# Track 5 EDA Report

## 1. Executive Summary
- All five sheets are stored as Excel tables with numeric-looking values imported as text, so type coercion is required before analysis.
- The workbook contains 226 duplicated rows across the five sheets, with the farmer and retail sheets contributing the largest shares.
- Unnamed placeholder columns are present in multiple sheets, especially the farmer and retail sheets, and should be removed from any modeling dataset.
- The heaviest missingness is concentrated in Farmer Data at column Unnamed: 9, where 575 rows are missing (100.0%).
- The strongest relationship appears in Rice Miller Data between Amount of milled rice (Kg) and nilaiberashasilgiling with correlation 0.999.
- The retail sheet is the weakest on the margin proxy, with an average spread of -2,595,926 IDR and only 2.9% positive rows.

## 2. Dataset Overview
| sheet | rows | columns | memory_kb | numeric | categorical | duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| Farmer Data | 575 | 13 | 209.11 | 8 | 2 | 168 |
| Rice Miller Data | 117 | 8 | 33.77 | 7 | 1 | 6 |
| Middlemen Data | 116 | 9 | 37.90 | 8 | 1 | 6 |
| Wholesaler Data | 116 | 7 | 29.59 | 6 | 1 | 6 |
| Retail Data | 149 | 10 | 43.53 | 6 | 2 | 40 |

## 3. Data Quality Assessment
| sheet | missing_columns | constant_columns | near_constant_columns | duplicate_rows | unnamed_columns |
| --- | --- | --- | --- | --- | --- |
| Farmer Data | 13 | 3 | 0 | 168 | 4 |
| Rice Miller Data | 8 | 0 | 0 | 6 | 0 |
| Middlemen Data | 9 | 0 | 0 | 6 | 0 |
| Wholesaler Data | 7 | 0 | 0 | 6 | 0 |
| Retail Data | 10 | 2 | 0 | 40 | 3 |

### Farmer Data
- Constant columns: Unnamed: 9, Unnamed: 10, Unnamed: 11
- Near-constant columns: none
- Invalid value checks: Land lease value (IDR): parse failures=4, negatives=0; Labor cost (IDR): parse failures=4, negatives=0; Seed purchase value (IDR): parse failures=4, negatives=0; Fertilizer purchase value (IDR): parse failures=4, negatives=0; Pesticide purchase value (IDR): parse failures=4, negatives=0; Equipment rent value (IDR): parse failures=4, negatives=0; Production value (IDR): parse failures=4, negatives=0

### Rice Miller Data
- Constant columns: none
- Near-constant columns: none
- Invalid value checks: Number of machines (unit): parse failures=4, negatives=0; Value of milled grains (IDR): parse failures=4, negatives=0; Amount of milled rice (Kg): parse failures=4, negatives=0; Labor cost (IDR): parse failures=4, negatives=0; Supporting equipment cost (IDR): parse failures=4, negatives=0; nilaiberashasilgiling: parse failures=4, negatives=0; Total revenue of milling machine (IDR): parse failures=4, negatives=0

### Middlemen Data
- Constant columns: none
- Near-constant columns: none
- Invalid value checks: Total rice purchase (kg): parse failures=4, negatives=0; Total rice purchase (IDR): parse failures=4, negatives=0; Building rent cost (IDR): parse failures=4, negatives=0; Labor cost (IDR): parse failures=4, negatives=0; Supporting equipment cost (IDR): parse failures=4, negatives=0; Value of rice sold (IDR): parse failures=4, negatives=0; Total precipitation (%): parse failures=4, negatives=0; Precipitation quality (%): parse failures=4, negatives=0

### Wholesaler Data
- Constant columns: none
- Near-constant columns: none
- Invalid value checks: Value of rice purchase (IDR): parse failures=4, negatives=0; Building rent cost (IDR): parse failures=4, negatives=0; Labor cost (IDR): parse failures=4, negatives=0; Supporting equipment cost (IDR): parse failures=4, negatives=0; Precipitation quality (%): parse failures=4, negatives=0; Value of rice sold (IDR): parse failures=4, negatives=0

### Retail Data
- Constant columns: Unnamed: 7, Unnamed: 8
- Near-constant columns: none
- Invalid value checks: Value of rice purchase (IDR): parse failures=4, negatives=0; Building rent cost (IDR): parse failures=4, negatives=0; Labor cost (IDR): parse failures=4, negatives=0; Supporting equipment cost (IDR): parse failures=4, negatives=0; Value of rice sold (IDR): parse failures=4, negatives=0; Precipitation quality (%): parse failures=4, negatives=0

## 4. Descriptive Statistics

### Farmer Data - Numerical Features
| feature | count | mean | median | mode | std | variance | min | max | range | q1 | q2 | q3 | iqr | skewness | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Land area (m2) | 404 | 4,484.93 | 2,115.00 | 10,000.00 | 5,378.95 | 28,933,097 | 2.00 | 30,000.00 | 29,998.00 | 1,200.00 | 2,115.00 | 6,000.00 | 4,800.00 | 2.30 | 6.02 |
| Land lease value (IDR) | 400 | 8,914,214 | 4,012,500 | 12,000,000 | 16,833,421 | 283,364,079,206,903 | 100,000.00 | 255,000,000 | 254,900,000 | 2,000,000 | 4,012,500 | 12,000,000 | 10,000,000 | 9.25 | 120.57 |
| Labor cost (IDR) | 400 | 2,461,494 | 980,000.00 | 500,000.00 | 3,485,303 | 12,147,340,326,402 | 55,000.00 | 32,000,000 | 31,945,000 | 500,000.00 | 980,000.00 | 3,225,000 | 2,725,000 | 3.53 | 19.08 |
| Seed purchase value (IDR) | 400 | 1,021,360 | 500,000.00 | 2,300,000 | 1,231,584 | 1,516,798,680,421 | 23,000.00 | 6,900,000 | 6,877,000 | 276,000.00 | 500,000.00 | 1,380,000 | 1,104,000 | 2.35 | 6.29 |
| Fertilizer purchase value (IDR) | 400 | 5,890,721 | 2,918,500 | 13,000,000 | 6,986,652 | 48,813,301,271,402 | 130,000.00 | 39,000,000 | 38,870,000 | 1,560,000 | 2,918,500 | 7,800,000 | 6,240,000 | 2.31 | 6.05 |
| Pesticide purchase value (IDR) | 400 | 681,450.62 | 296,180.00 | 1,300,000 | 901,390.00 | 812,503,937,766 | 13,000.00 | 5,040,000 | 5,027,000 | 162,890.00 | 296,180.00 | 910,000.00 | 747,110.00 | 2.39 | 5.82 |
| Equipment rent value (IDR) | 400 | 1,130,256 | 655,000.00 | 350,000.00 | 1,255,069 | 1,575,199,353,721 | 127,000.00 | 7,500,009 | 7,373,009 | 384,550.00 | 655,000.00 | 1,300,000 | 915,450.00 | 2.87 | 10.02 |
| Production value (IDR) | 400 | 41,321,811 | 24,525,000 | 13,050,000 | 48,037,546 | 2,307,605,821,547,320 | 1,620,000 | 267,300,000 | 265,680,000 | 11,275,000 | 24,525,000 | 51,400,000 | 40,125,000 | 2.36 | 6.41 |

### Farmer Data - Categorical Features
| feature | unique_categories | top_category | top_frequency | top_percentage |
| --- | --- | --- | --- | --- |
| dmu | 115 | 1 | 5 | 1.23 |
| Unnamed: 12 | 1 | Count Beli | 1 | 100.00 |

### Rice Miller Data - Numerical Features
| feature | count | mean | median | mode | std | variance | min | max | range | q1 | q2 | q3 | iqr | skewness | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Number of machines (unit) | 105 | 2.23 | 2.00 | 1.00 | 1.29 | 1.66 | 1.00 | 6.00 | 5.00 | 1.00 | 2.00 | 3.00 | 2.00 | 0.8555 | -0.0189 |
| Value of milled grains (IDR) | 105 | 254,488,858 | 120,000,000 | 300,000,000 | 643,741,482 | 414,403,095,861,507,008 | 3,000,000 | 5,000,000,000 | 4,997,000,000 | 25,000,000 | 120,000,000 | 275,000,000 | 250,000,000 | 6.26 | 41.57 |
| Amount of milled rice (Kg) | 105 | 31,853.79 | 14,000.00 | 38,412.00 | 83,495.03 | 6,971,420,104 | 300.00 | 600,000.00 | 599,700.00 | 3,000.00 | 14,000.00 | 35,000.00 | 32,000.00 | 6.12 | 39.20 |
| Labor cost (IDR) | 105 | 9,541,283 | 5,392,557 | 3,595,038 | 12,227,605 | 149,514,332,913,855 | 1,200,000 | 65,000,000 | 63,800,000 | 3,000,000 | 5,392,557 | 8,987,595 | 5,987,595 | 2.75 | 7.58 |
| Supporting equipment cost (IDR) | 105 | 1,599,538 | 1,599,538 | 1,599,538 | 1,421,076 | 2,019,457,590,237 | 150,000.00 | 6,000,000 | 5,850,000 | 500,000.00 | 1,599,538 | 2,000,000 | 1,500,000 | 1.56 | 2.25 |
| nilaiberashasilgiling | 105 | 302,877,292 | 126,000,000 | 5,400,000 | 781,751,597 | 611,135,559,758,900,480 | 2,902,899 | 5,689,681,959 | 5,686,779,060 | 29,028,990 | 126,000,000 | 336,349,226 | 307,320,236 | 6.08 | 38.75 |
| Total revenue of milling machine (IDR) | 105 | 24,844,133 | 10,400,000 | 31,200,000 | 63,369,942 | 4,015,749,537,558,976 | 312,000.00 | 468,000,000 | 467,688,000 | 2,600,000 | 10,400,000 | 28,600,000 | 26,000,000 | 6.14 | 39.73 |

### Rice Miller Data - Categorical Features
| feature | unique_categories | top_category | top_frequency | top_percentage |
| --- | --- | --- | --- | --- |
| dmu | 34 | 1 | 5 | 4.42 |

### Middlemen Data - Numerical Features
| feature | count | mean | median | mode | std | variance | min | max | range | q1 | q2 | q3 | iqr | skewness | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Total rice purchase (kg) | 104 | 10,895.19 | 10,000.00 | 10,000.00 | 14,721.74 | 216,729,637 | 700.00 | 120,000.00 | 119,300.00 | 2,650.00 | 10,000.00 | 10,000.00 | 7,350.00 | 4.56 | 29.43 |
| Total rice purchase (IDR) | 104 | 81,484,053 | 70,000,000 | 83,000,000 | 108,286,594 | 11,725,986,371,121,704 | 6,225,000 | 960,000,000 | 953,775,000 | 25,362,500 | 70,000,000 | 83,000,000 | 57,637,500 | 5.57 | 42.20 |
| Building rent cost (IDR) | 104 | 2,889,423 | 3,200,000 | 3,200,000 | 2,157,440 | 4,654,547,236,744 | 300,000.00 | 20,000,000 | 19,700,000 | 1,500,000 | 3,200,000 | 3,200,000 | 1,700,000 | 5.17 | 38.71 |
| Labor cost (IDR) | 104 | 5,272,504 | 5,000,000 | 1,007,069 | 3,643,604 | 13,275,849,081,076 | 500,000.00 | 15,106,035 | 14,606,035 | 2,014,139 | 5,000,000 | 7,200,000 | 5,185,861 | 0.7835 | -0.1669 |
| Supporting equipment cost (IDR) | 104 | 1,511,643 | 1,000,000 | 1,500,000 | 1,838,285 | 3,379,291,951,779 | 50,000.00 | 10,000,000 | 9,950,000 | 500,000.00 | 1,000,000 | 1,500,000 | 1,000,000 | 2.63 | 7.14 |
| Value of rice sold (IDR) | 104 | 95,375,096 | 88,000,000 | 88,000,000 | 130,039,442 | 16,910,256,382,437,266 | 5,896,000 | 1,056,000,000 | 1,050,104,000 | 22,785,000 | 88,000,000 | 88,000,000 | 65,215,000 | 4.54 | 29.08 |
| Total precipitation (%) | 104 | 14.53 | 10.00 | 0.0000 | 15.23 | 231.84 | 0.0000 | 50.00 | 50.00 | 3.75 | 10.00 | 20.75 | 17.00 | 0.9659 | -0.2848 |
| Precipitation quality (%) | 104 | 15.11 | 10.00 | 0.0000 | 15.79 | 249.22 | 0.0000 | 50.00 | 50.00 | 1.50 | 10.00 | 25.00 | 23.50 | 0.8751 | -0.5374 |

### Middlemen Data - Categorical Features
| feature | unique_categories | top_category | top_frequency | top_percentage |
| --- | --- | --- | --- | --- |
| dmu | 34 | 1 | 5 | 4.46 |

### Wholesaler Data - Numerical Features
| feature | count | mean | median | mode | std | variance | min | max | range | q1 | q2 | q3 | iqr | skewness | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value of rice purchase (IDR) | 104 | 392,759,423 | 114,000,000 | 114,000,000 | 1,085,302,734 | 1,177,882,025,137,528,064 | 9,000,000 | 8,000,000,000 | 7,991,000,000 | 88,500,000 | 114,000,000 | 197,750,000 | 109,250,000 | 5.49 | 32.78 |
| Building rent cost (IDR) | 104 | 5,866,827 | 4,000,000 | 4,000,000 | 5,874,408 | 34,508,670,416,355 | 1,500,000 | 36,000,000 | 34,500,000 | 4,000,000 | 4,000,000 | 5,000,000 | 1,000,000 | 3.44 | 12.72 |
| Labor cost (IDR) | 104 | 8,799,615 | 8,000,000 | 8,000,000 | 7,000,109 | 49,001,524,122,479 | 200,000.00 | 40,560,000 | 40,360,000 | 4,725,000 | 8,000,000 | 10,125,000 | 5,400,000 | 2.00 | 5.14 |
| Supporting equipment cost (IDR) | 104 | 7,926,923 | 3,000,000 | 2,000,000 | 19,425,552 | 377,352,083,644,511 | 200,000.00 | 175,000,000 | 174,800,000 | 1,500,000 | 3,000,000 | 6,175,000 | 4,675,000 | 6.87 | 55.28 |
| Precipitation quality (%) | 104 | 15.30 | 10.00 | 0.0000 | 15.81 | 250.11 | 0.0000 | 50.00 | 50.00 | 5.00 | 10.00 | 20.00 | 15.00 | 0.9274 | -0.3890 |
| Value of rice sold (IDR) | 104 | 647,413,817 | 180,000,000 | 180,000,000 | 1,513,287,532 | 2,290,039,155,959,161,600 | 9,000,000 | 10,500,000,000 | 10,491,000,000 | 99,750,000 | 180,000,000 | 485,000,000 | 385,250,000 | 4.61 | 23.73 |

### Wholesaler Data - Categorical Features
| feature | unique_categories | top_category | top_frequency | top_percentage |
| --- | --- | --- | --- | --- |
| dmu | 30 | 1 | 4 | 4.76 |

### Retail Data - Numerical Features
| feature | count | mean | median | mode | std | variance | min | max | range | q1 | q2 | q3 | iqr | skewness | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value of rice purchase (IDR) | 102 | 2,450,515 | 2,612,500 | 2,695,000 | 389,908.33 | 152,028,507,207 | 585,000.00 | 3,240,000 | 2,655,000 | 2,450,000 | 2,612,500 | 2,695,000 | 245,000.00 | -1.88 | 4.82 |
| Building rent cost (IDR) | 102 | 187,450.98 | 200,000.00 | 200,000.00 | 57,258.94 | 3,278,586,682 | 50,000.00 | 480,000.00 | 430,000.00 | 150,000.00 | 200,000.00 | 200,000.00 | 50,000.00 | 1.16 | 6.42 |
| Labor cost (IDR) | 102 | 2,676,961 | 3,000,000 | 3,000,000 | 958,021.67 | 917,805,523,199 | 200,000.00 | 5,600,000 | 5,400,000 | 3,000,000 | 3,000,000 | 3,000,000 | 0.0000 | -0.8652 | 1.63 |
| Supporting equipment cost (IDR) | 102 | 204,156.86 | 200,000.00 | 200,000.00 | 97,635.69 | 9,532,727,626 | 30,000.00 | 500,000.00 | 470,000.00 | 150,000.00 | 200,000.00 | 300,000.00 | 150,000.00 | 0.4793 | 0.3985 |
| Value of rice sold (IDR) | 102 | 2,923,157 | 3,000,000 | 3,000,000 | 498,961.07 | 248,962,153,368 | 700,000.00 | 3,600,000 | 2,900,000 | 3,000,000 | 3,000,000 | 3,000,000 | 0.0000 | -1.48 | 3.57 |
| Precipitation quality (%) | 102 | 0.1270 | 0.1000 | 0.0500 | 0.1110 | 0.0123 | 0.0500 | 0.4500 | 0.4000 | 0.0500 | 0.1000 | 0.1500 | 0.1000 | 1.55 | 1.24 |

### Retail Data - Categorical Features
| feature | unique_categories | top_category | top_frequency | top_percentage |
| --- | --- | --- | --- | --- |
| dmu | 33 | 1 | 5 | 4.55 |
| Unnamed: 9 | 1 | Count Beli | 1 | 100.00 |

## 5. Missing Value Analysis

### Farmer Data
| column | missing_count | missing_pct |
| --- | --- | --- |
| Unnamed: 9 | 575 | 100.00 |
| Unnamed: 10 | 575 | 100.00 |
| Unnamed: 11 | 575 | 100.00 |
| Unnamed: 12 | 574 | 99.83 |
| Land area (m2) | 171 | 29.74 |
| Land lease value (IDR) | 171 | 29.74 |
| Labor cost (IDR) | 171 | 29.74 |
| Seed purchase value (IDR) | 171 | 29.74 |
| Fertilizer purchase value (IDR) | 171 | 29.74 |
| Pesticide purchase value (IDR) | 171 | 29.74 |
| Equipment rent value (IDR) | 171 | 29.74 |
| Production value (IDR) | 171 | 29.74 |
| dmu | 167 | 29.04 |
- Highest missingness: Unnamed: 9: 100.0%, Unnamed: 10: 100.0%, Unnamed: 11: 100.0%

### Rice Miller Data
| column | missing_count | missing_pct |
| --- | --- | --- |
| Number of machines (unit) | 8 | 6.84 |
| Value of milled grains (IDR) | 8 | 6.84 |
| Amount of milled rice (Kg) | 8 | 6.84 |
| Labor cost (IDR) | 8 | 6.84 |
| Supporting equipment cost (IDR) | 8 | 6.84 |
| nilaiberashasilgiling | 8 | 6.84 |
| Total revenue of milling machine (IDR) | 8 | 6.84 |
| dmu | 4 | 3.42 |
- Highest missingness: Number of machines (unit): 6.8%, Value of milled grains (IDR): 6.8%, Amount of milled rice (Kg): 6.8%

### Middlemen Data
| column | missing_count | missing_pct |
| --- | --- | --- |
| Total rice purchase (kg) | 8 | 6.90 |
| Total rice purchase (IDR) | 8 | 6.90 |
| Building rent cost (IDR) | 8 | 6.90 |
| Labor cost (IDR) | 8 | 6.90 |
| Supporting equipment cost (IDR) | 8 | 6.90 |
| Value of rice sold (IDR) | 8 | 6.90 |
| Total precipitation (%) | 8 | 6.90 |
| Precipitation quality (%) | 8 | 6.90 |
| dmu | 4 | 3.45 |
- Highest missingness: Total rice purchase (kg): 6.9%, Total rice purchase (IDR): 6.9%, Building rent cost (IDR): 6.9%

### Wholesaler Data
| column | missing_count | missing_pct |
| --- | --- | --- |
| dmu | 32 | 27.59 |
| Value of rice purchase (IDR) | 8 | 6.90 |
| Building rent cost (IDR) | 8 | 6.90 |
| Labor cost (IDR) | 8 | 6.90 |
| Supporting equipment cost (IDR) | 8 | 6.90 |
| Precipitation quality (%) | 8 | 6.90 |
| Value of rice sold (IDR) | 8 | 6.90 |
- Highest missingness: dmu: 27.6%, Value of rice purchase (IDR): 6.9%, Building rent cost (IDR): 6.9%

### Retail Data
| column | missing_count | missing_pct |
| --- | --- | --- |
| Unnamed: 7 | 149 | 100.00 |
| Unnamed: 8 | 149 | 100.00 |
| Unnamed: 9 | 148 | 99.33 |
| Value of rice purchase (IDR) | 43 | 28.86 |
| Building rent cost (IDR) | 43 | 28.86 |
| Labor cost (IDR) | 43 | 28.86 |
| Supporting equipment cost (IDR) | 43 | 28.86 |
| Value of rice sold (IDR) | 43 | 28.86 |
| Precipitation quality (%) | 43 | 28.86 |
| dmu | 39 | 26.17 |
- Highest missingness: Unnamed: 7: 100.0%, Unnamed: 8: 100.0%, Unnamed: 9: 99.3%

## 6. Outlier Analysis

### Farmer Data
| feature | iqr_outliers | iqr_outlier_pct | z_outliers | z_outlier_pct |
| --- | --- | --- | --- | --- |
| Land area (m2) | 32 | 7.92 | 5 | 1.24 |
| Land lease value (IDR) | 20 | 5.00 | 4 | 1.00 |
| Labor cost (IDR) | 27 | 6.75 | 8 | 2.00 |
| Seed purchase value (IDR) | 31 | 7.75 | 6 | 1.50 |
| Fertilizer purchase value (IDR) | 32 | 8.00 | 5 | 1.25 |
| Pesticide purchase value (IDR) | 35 | 8.75 | 14 | 3.50 |
| Equipment rent value (IDR) | 36 | 9.00 | 13 | 3.25 |
| Production value (IDR) | 36 | 9.00 | 12 | 3.00 |
- IQR outliers identify unusually large or small values relative to the middle 50% of the data.
- Z-score outliers are the values more than 3 standard deviations from the mean.

### Rice Miller Data
| feature | iqr_outliers | iqr_outlier_pct | z_outliers | z_outlier_pct |
| --- | --- | --- | --- | --- |
| Number of machines (unit) | 0 | 0.0000 | 0 | 0.0000 |
| Value of milled grains (IDR) | 4 | 3.81 | 2 | 1.90 |
| Amount of milled rice (Kg) | 3 | 2.86 | 2 | 1.90 |
| Labor cost (IDR) | 13 | 12.38 | 4 | 3.81 |
| Supporting equipment cost (IDR) | 7 | 6.67 | 4 | 3.81 |
| nilaiberashasilgiling | 3 | 2.86 | 2 | 1.90 |
| Total revenue of milling machine (IDR) | 3 | 2.86 | 2 | 1.90 |
- IQR outliers identify unusually large or small values relative to the middle 50% of the data.
- Z-score outliers are the values more than 3 standard deviations from the mean.

### Middlemen Data
| feature | iqr_outliers | iqr_outlier_pct | z_outliers | z_outlier_pct |
| --- | --- | --- | --- | --- |
| Total rice purchase (kg) | 12 | 11.54 | 1 | 0.9615 |
| Total rice purchase (IDR) | 11 | 10.58 | 1 | 0.9615 |
| Building rent cost (IDR) | 5 | 4.81 | 1 | 0.9615 |
| Labor cost (IDR) | 2 | 1.92 | 0 | 0.0000 |
| Supporting equipment cost (IDR) | 10 | 9.62 | 4 | 3.85 |
| Value of rice sold (IDR) | 14 | 13.46 | 1 | 0.9615 |
| Total precipitation (%) | 5 | 4.81 | 0 | 0.0000 |
| Precipitation quality (%) | 0 | 0.0000 | 0 | 0.0000 |
- IQR outliers identify unusually large or small values relative to the middle 50% of the data.
- Z-score outliers are the values more than 3 standard deviations from the mean.

### Wholesaler Data
| feature | iqr_outliers | iqr_outlier_pct | z_outliers | z_outlier_pct |
| --- | --- | --- | --- | --- |
| Value of rice purchase (IDR) | 15 | 14.42 | 2 | 1.92 |
| Building rent cost (IDR) | 26 | 25.00 | 3 | 2.88 |
| Labor cost (IDR) | 9 | 8.65 | 2 | 1.92 |
| Supporting equipment cost (IDR) | 15 | 14.42 | 2 | 1.92 |
| Precipitation quality (%) | 8 | 7.69 | 0 | 0.0000 |
| Value of rice sold (IDR) | 12 | 11.54 | 3 | 2.88 |
- IQR outliers identify unusually large or small values relative to the middle 50% of the data.
- Z-score outliers are the values more than 3 standard deviations from the mean.

### Retail Data
| feature | iqr_outliers | iqr_outlier_pct | z_outliers | z_outlier_pct |
| --- | --- | --- | --- | --- |
| Value of rice purchase (IDR) | 21 | 20.59 | 1 | 0.9804 |
| Building rent cost (IDR) | 7 | 6.86 | 1 | 0.9804 |
| Labor cost (IDR) | 33 | 32.35 | 1 | 0.9804 |
| Supporting equipment cost (IDR) | 0 | 0.0000 | 2 | 1.96 |
| Value of rice sold (IDR) | 45 | 44.12 | 1 | 0.9804 |
| Precipitation quality (%) | 11 | 10.78 | 0 | 0.0000 |
- IQR outliers identify unusually large or small values relative to the middle 50% of the data.
- Z-score outliers are the values more than 3 standard deviations from the mean.

## 7. Univariate Analysis

### Farmer Data
- Land lease value (IDR) is positively skewed with skewness 9.25, suggesting a long upper tail and a small number of very large observations.
- Labor cost (IDR) is positively skewed with skewness 3.53, suggesting a long upper tail and a small number of very large observations.
- Equipment rent value (IDR) is positively skewed with skewness 2.87, suggesting a long upper tail and a small number of very large observations.
- Land area (m2) is the most left-skewed feature in this sheet with skewness 2.30.

### Rice Miller Data
- Value of milled grains (IDR) is positively skewed with skewness 6.26, suggesting a long upper tail and a small number of very large observations.
- Total revenue of milling machine (IDR) is positively skewed with skewness 6.14, suggesting a long upper tail and a small number of very large observations.
- Amount of milled rice (Kg) is positively skewed with skewness 6.12, suggesting a long upper tail and a small number of very large observations.
- Number of machines (unit) is the most left-skewed feature in this sheet with skewness 0.86.

### Middlemen Data
- Total rice purchase (IDR) is positively skewed with skewness 5.57, suggesting a long upper tail and a small number of very large observations.
- Building rent cost (IDR) is positively skewed with skewness 5.17, suggesting a long upper tail and a small number of very large observations.
- Total rice purchase (kg) is positively skewed with skewness 4.56, suggesting a long upper tail and a small number of very large observations.
- Labor cost (IDR) is the most left-skewed feature in this sheet with skewness 0.78.

### Wholesaler Data
- Supporting equipment cost (IDR) is positively skewed with skewness 6.87, suggesting a long upper tail and a small number of very large observations.
- Value of rice purchase (IDR) is positively skewed with skewness 5.49, suggesting a long upper tail and a small number of very large observations.
- Value of rice sold (IDR) is positively skewed with skewness 4.61, suggesting a long upper tail and a small number of very large observations.
- Precipitation quality (%) is the most left-skewed feature in this sheet with skewness 0.93.

### Retail Data
- Precipitation quality (%) is positively skewed with skewness 1.55, suggesting a long upper tail and a small number of very large observations.
- Building rent cost (IDR) is positively skewed with skewness 1.16, suggesting a long upper tail and a small number of very large observations.
- Supporting equipment cost (IDR) is positively skewed with skewness 0.48, suggesting a long upper tail and a small number of very large observations.
- Value of rice purchase (IDR) is the most left-skewed feature in this sheet with skewness -1.88.

## 8. Bivariate Analysis

### Farmer Data
- Strongest correlations: Land area (m2) vs Fertilizer purchase value (IDR) = 0.999; Land area (m2) vs Seed purchase value (IDR) = 0.991; Seed purchase value (IDR) vs Fertilizer purchase value (IDR) = 0.990; Land area (m2) vs Pesticide purchase value (IDR) = 0.934; Fertilizer purchase value (IDR) vs Pesticide purchase value (IDR) = 0.933
- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.

### Rice Miller Data
- Strongest correlations: Amount of milled rice (Kg) vs nilaiberashasilgiling = 0.999; Amount of milled rice (Kg) vs Total revenue of milling machine (IDR) = 0.995; Value of milled grains (IDR) vs Amount of milled rice (Kg) = 0.995; Value of milled grains (IDR) vs Total revenue of milling machine (IDR) = 0.995; nilaiberashasilgiling vs Total revenue of milling machine (IDR) = 0.993
- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.

### Middlemen Data
- Strongest correlations: Total rice purchase (kg) vs Value of rice sold (IDR) = 0.999; Total rice purchase (kg) vs Total rice purchase (IDR) = 0.981; Total rice purchase (IDR) vs Value of rice sold (IDR) = 0.978; Total precipitation (%) vs Precipitation quality (%) = 0.942; Labor cost (IDR) vs Value of rice sold (IDR) = 0.344
- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.

### Wholesaler Data
- Strongest correlations: Value of rice purchase (IDR) vs Value of rice sold (IDR) = 0.913; Supporting equipment cost (IDR) vs Value of rice sold (IDR) = 0.752; Value of rice purchase (IDR) vs Supporting equipment cost (IDR) = 0.734; Value of rice purchase (IDR) vs Building rent cost (IDR) = 0.509; Building rent cost (IDR) vs Value of rice sold (IDR) = 0.430
- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.

### Retail Data
- Strongest correlations: Value of rice purchase (IDR) vs Value of rice sold (IDR) = 0.612; Supporting equipment cost (IDR) vs Value of rice sold (IDR) = 0.336; Value of rice purchase (IDR) vs Supporting equipment cost (IDR) = 0.322; Supporting equipment cost (IDR) vs Precipitation quality (%) = -0.256; Building rent cost (IDR) vs Precipitation quality (%) = 0.218
- Multicollinearity risk is high in the farmer and rice miller sheets because several cost and production fields move almost in lockstep.

## 9. Multivariate Analysis
| sheet | avg_revenue | avg_costs | avg_margin | positive_margin_pct |
| --- | --- | --- | --- | --- |
| Farmer Data | 41,321,811 | 13,982,258 | 21,222,316 | 90.50 |
| Rice Miller Data | 24,844,133 | 9,998,173 | 13,703,312 | 51.43 |
| Middlemen Data | 95,375,096 | 81,727,524 | 4,217,474 | 32.69 |
| Wholesaler Data | 647,413,817 | 372,385,259 | 232,061,029 | 58.65 |
| Retail Data | 2,923,157 | 3,778,164 | -2,595,926 | 2.94 |
- The retail sheet shows the weakest margin proxy, while middlemen and wholesalers show the strongest spreads.
- The supply chain appears to accumulate value downstream, but the retail economics are much tighter and potentially under pressure.

## 10. Visualization Insights
- Histogram plots show that most monetary variables are right-skewed with a small number of very large transactions.
- Boxplots confirm numerous high-end outliers in the cost and revenue measures, especially in the farmer, middlemen, and wholesaler sheets.
- Correlation heatmaps show very tight relationships between production or transaction volume and the associated revenue fields.
- Scatter plots of the strongest correlated pairs reinforce the near-linear movement of volume and value variables.
- Missing-value bars highlight the blank placeholder columns and the partial row loss common to all sheets.

## 11. Feature Engineering Opportunities
- Total cost features by sheet: sum all input and operating cost columns to create a single economic burden measure.
- Margin proxy: revenue minus costs, useful as a target or health indicator.
- Cost-to-revenue ratio: helps normalize scale differences between farmers, millers, middlemen, wholesalers, and retailers.
- Yield or throughput features: land area to production value for farmers, rice quantity to machine revenue for millers.
- Supplier/market spread features: purchase value versus sales value for middlemen, wholesalers, and retail actors.

## 12. Modeling Readiness
- Data cleanliness is moderate: the numeric content is usable after coercion, but type normalization and column pruning are mandatory.
- The strongest modeling risks are duplicate rows, blank columns, and multicollinearity among cost and revenue fields.
- There is no obvious datetime or boolean target in the workbook, so the data is better suited to regression, anomaly detection, or descriptive segmentation than direct supervised classification.
- Before modeling, remove empty columns, deduplicate rows, standardize numeric types, and review the business meaning of the apparent extreme outliers.

## 13. Recommendations
- Remove the Unnamed placeholder columns immediately.
- Convert all numeric-looking fields to numeric types and keep dmu as a record identifier.
- Investigate the duplicate rows in the farmer and retail sheets before training any model.
- Treat extreme high-value transactions as potential special cases rather than automatic errors.
- Build derived margin and ratio features before any forecasting or optimization task.
- Use the workbook primarily for operational analytics, profitability analysis, and outlier detection.

## 14. Final Conclusion
This workbook contains a useful supply-chain view of rice production and trading across five stages, but it requires structured cleaning before advanced analytics. The dominant patterns are heavy right-skew, strong revenue-volume relationships, duplicated rows, and several empty placeholder columns. Once standardized, the dataset is suitable for business performance analysis and margin-based feature engineering.

## Generated Assets
- reports/track_5_eda/assets/farmer_data_histograms.png
- reports/track_5_eda/assets/farmer_data_boxplots.png
- reports/track_5_eda/assets/farmer_data_correlation.png
- reports/track_5_eda/assets/farmer_data_scatter_1.png
- reports/track_5_eda/assets/farmer_data_scatter_2.png
- reports/track_5_eda/assets/farmer_data_missing.png
- reports/track_5_eda/assets/rice_miller_data_histograms.png
- reports/track_5_eda/assets/rice_miller_data_boxplots.png
- reports/track_5_eda/assets/rice_miller_data_correlation.png
- reports/track_5_eda/assets/rice_miller_data_scatter_1.png
- reports/track_5_eda/assets/rice_miller_data_scatter_2.png
- reports/track_5_eda/assets/rice_miller_data_missing.png
- reports/track_5_eda/assets/middlemen_data_histograms.png
- reports/track_5_eda/assets/middlemen_data_boxplots.png
- reports/track_5_eda/assets/middlemen_data_correlation.png
- reports/track_5_eda/assets/middlemen_data_scatter_1.png
- reports/track_5_eda/assets/middlemen_data_scatter_2.png
- reports/track_5_eda/assets/middlemen_data_missing.png
- reports/track_5_eda/assets/wholesaler_data_histograms.png
- reports/track_5_eda/assets/wholesaler_data_boxplots.png
- reports/track_5_eda/assets/wholesaler_data_correlation.png
- reports/track_5_eda/assets/wholesaler_data_scatter_1.png
- reports/track_5_eda/assets/wholesaler_data_scatter_2.png
- reports/track_5_eda/assets/wholesaler_data_missing.png
- reports/track_5_eda/assets/retail_data_histograms.png
- reports/track_5_eda/assets/retail_data_boxplots.png
- reports/track_5_eda/assets/retail_data_correlation.png
- reports/track_5_eda/assets/retail_data_scatter_1.png
- reports/track_5_eda/assets/retail_data_scatter_2.png
- reports/track_5_eda/assets/retail_data_missing.png
- reports/track_5_eda/assets/sheet_level_comparison.png