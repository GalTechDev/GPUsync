# GPUsync Global Benchmark Summary Report

- **Périphérique de calcul** : Intel(R) Graphics (`opencl`)
- **Date de génération** : 2026-07-22 12:20:27

## DTW
*Dynamic Time Warping 1-to-N distance computation using dtw_gpu_engine*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.0027s | GPUsync | 0.0028s | **0.96x** |
| 100,000 | 0.0279s | GPUsync | 0.0041s | **6.87x** |
| 1,000,000 | 0.3195s | GPUsync | 0.0426s | **7.50x** |
| 2,000,000 | 0.6179s | GPUsync | 0.0730s | **8.46x** |

## KMeans
*K-Means Cluster Assignment Phase (N points, K=16, D=8) using kmeans_engine*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 50,000 | 0.0307s | GPUsync | 0.0119s | **2.58x** |
| 250,000 | 0.1517s | GPUsync | 0.0440s | **3.45x** |

## XMeans
*Standard X-Means Multidimensional Clustering (N points, D=8, K_min=2, K_max=16)*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.0199s | GPUsync | 0.0305s | **0.65x** |
| 50,000 | 0.1382s | GPUsync | 0.0736s | **1.88x** |

## MatrixMult
*2D Dense Matrix Multiplication (N x N) - Comparison of 6 variants*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 128 | 0.0003s | OpenCL Tiled | 0.0044s | **0.08x** |
|  |  | OpenCL 2D Tiling | 0.0054s | **0.06x** |
|  |  | CUDA Double Buffering | Skipped | N/A |
|  |  | CUDA WMMA | Skipped | N/A |
|  |  | CUDA WMMA 2D | Skipped | N/A |
|  |  | CUDA Direct cuBLAS | Skipped | N/A |
| 256 | 0.0005s | OpenCL Tiled | 0.0052s | **0.10x** |
|  |  | OpenCL 2D Tiling | 0.0048s | **0.11x** |
|  |  | CUDA Double Buffering | Skipped | N/A |
|  |  | CUDA WMMA | Skipped | N/A |
|  |  | CUDA WMMA 2D | Skipped | N/A |
|  |  | CUDA Direct cuBLAS | Skipped | N/A |
| 512 | 0.0017s | OpenCL Tiled | 0.0112s | **0.16x** |
|  |  | OpenCL 2D Tiling | 0.0072s | **0.24x** |
|  |  | CUDA Double Buffering | Skipped | N/A |
|  |  | CUDA WMMA | Skipped | N/A |
|  |  | CUDA WMMA 2D | Skipped | N/A |
|  |  | CUDA Direct cuBLAS | Skipped | N/A |

## KNN-DTW
*K-Nearest Neighbors Dynamic Time Warping window search*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.0692s | Single Query | 0.0234s | **2.95x** |
|  |  | Multi-Query (Q=4) | 0.0228s | **3.04x** |
| 25,000 | 0.1815s | Single Query | 0.0536s | **3.39x** |
|  |  | Multi-Query (Q=4) | 0.0453s | **4.01x** |

## KShape
*KShape Time Series Clustering*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 0.2206s | GPUsync Hybrid Fit | 0.0920s | **2.40x** |
| 250 | 0.4687s | GPUsync Hybrid Fit | 0.1606s | **2.92x** |

## LOESS
*Local Regression (LOWESS) Smoothing*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.3753s | GPUsync LOESS Fit | 0.0092s | **40.91x** |
| 25,000 | 0.8545s | GPUsync LOESS Fit | 0.0080s | **106.80x** |

## STOMP
*Matrix Profile Computation using STOMP*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.1548s | GPUsync STOMP Fit | 0.0747s | **2.07x** |
| 25,000 | 0.3518s | GPUsync STOMP Fit | 0.3249s | **1.08x** |

## GR-Soft-DTW
*Graph-Regularized Soft-DTW Forward/Backward/HVP*

| Taille N | Temps CPU (s) | Variante | Temps GPU (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 50 | 1.2337s | GPUsync FWD+BWD+HVP | 0.0294s | **41.98x** |
| 75 | 1.1103s | GPUsync FWD+BWD+HVP | 0.0197s | **56.40x** |

