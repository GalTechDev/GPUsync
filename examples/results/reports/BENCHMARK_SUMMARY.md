# GPUsync Global Benchmark Summary Report

- **Périphérique de calcul** : Intel(R) Graphics (`opencl`)
- **Date de génération** : 2026-07-21 16:09:40

## DTW
*Dynamic Time Warping 1-to-N distance computation*

| Taille N | Temps CPU (s) | GPUsync (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 10,000 | 0.1006s | 0.0039s | **25.64x** |
| 50,000 | 0.0969s | 0.0050s | **19.27x** |
| 100,000 | 0.0966s | 0.0067s | **14.39x** |

## KMeans
*K-Means Cluster Assignment Phase (N points, K=16, D=8)*

| Taille N | Temps CPU (s) | GPUsync (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 50,000 | 0.0363s | 0.0037s | **9.83x** |
| 250,000 | 0.1332s | 0.0070s | **18.98x** |
| 500,000 | 0.2727s | 0.0106s | **25.64x** |

## MatrixMult
*2D Dense Matrix Multiplication (N x N)*

| Taille N | Temps CPU (s) | GPUsync (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 128 | 0.0003s | 0.0017s | **0.16x** |
| 256 | 0.0013s | 0.0025s | **0.53x** |
| 512 | 0.0018s | 0.0053s | **0.35x** |

