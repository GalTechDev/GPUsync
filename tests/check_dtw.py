import math

import numpy as np
import pandas as pd


def dtw_window(s1, s2, w):
    n = len(s1)
    m = len(s2)
    w = max(w, abs(n - m))

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m + 1, i + w + 1)):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return math.sqrt(dp[n, m])


df = pd.read_excel("tests/RomaneDataSet_OXYGENE.xlsx")
data = df.iloc[:, 1:].values.astype(np.float32)

print("Valeur python (Série 0 vs Série 1, w=5):", dtw_window(data[0], data[1], 5))
print("Valeur python (Série 0 vs Série 2, w=5):", dtw_window(data[0], data[2], 5))
print("Valeur python (Série 0 vs Série 0, w=5):", dtw_window(data[0], data[0], 5))

# Charger la matrice calculée
mat = pd.read_csv("tests/dtw_matrix_oxygen.csv", index_col=0).values
print("Valeur GPU (Série 0 vs Série 1):", mat[0, 1])
print("Valeur GPU (Série 0 vs Série 2):", mat[0, 2])
print("Valeur GPU (Série 0 vs Série 0):", mat[0, 0])
