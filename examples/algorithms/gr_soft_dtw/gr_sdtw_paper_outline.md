# Plan de Rédaction & Structure du Papier de Recherche : GR-S-DTW

## Titre Proposé
**Efficient Second-Order Alignment for Graph-Regularized Soft-DTW**  
*(ou : Graph-Regularized Soft-DTW with $\mathcal{O}(NM)$ Analytical Hessian-Vector Products)*

---

## 1. Abstract (Résumé)

La métrique *Soft-Dynamic Time Warping* (Soft-DTW) est largement adoptée comme fonction de perte différentiable pour l'alignement de séries temporelles. Cependant, elle aligne les dimensions de manière indépendante, ignorant les contraintes physiques ou topologiques reliant les capteurs dans des applications réelles (IoT, réseaux sismiques, biomécanique). Dans ce travail, nous présentons le **Graph-Regularized Soft-DTW** (GR-S-DTW), une extension qui introduit une pénalité de lissage spatial sur les matrices d'alignement de capteurs adjacents dans un graphe, tout en autorisant un déphasage temporel modélisé physiquement. 

Bien que cette formulation introduise des dépendances au second ordre (gradients de gradients), nous résolvons le verrou computationnel historique d'explosion mémoire en $\mathcal{O}(N^4)$ de l'Autograd standard. Nous dérivons un algorithme analytique de produit Hessienne-Vecteur (HVP) en $\mathcal{O}(NM)$ sous forme de programmation dynamique à trois passes. Implémenté sur GPU, notre noyau élimine les plantages mémoire (OOM) et surpasse les performances d'Autograd de plusieurs ordres de grandeur. Nous démontrons l'efficacité et la robustesse de GR-S-DTW pour l'alignement et le débruitage de signaux sur des réseaux de capteurs simulés et réels.

---

## 2. Structure Détaillée de l'Article

### Section 1 : Introduction
*   **Contexte** : Alignement temporel élastique, importance du DTW et transition vers sa version différentiable (Soft-DTW) pour le Deep Learning.
*   **Limitation des approches actuelles** : La Soft-DTW multivariée classique aligne les signaux de manière indépendante ou globale sans préserver les relations topologiques (voisinage de capteurs) ou propager des retards physiques.
*   **Le Défi Mathématique** : L'ajout d'une régularisation sur les matrices d'alignement (gradients de premier ordre de la perte) nécessite un double backward (Hessienne). L'Autograd classique génère des graphes de calcul géants conduisant à des plantages mémoire *Out Of Memory* (OOM) dès $N=100$.
*   **Contributions** :
    1. Formulation de la perte de régularisation par graphe avec déphasage temporel modélisé par une matrice tridiagonale $P_{\Delta}$.
    2. Dérivation formelle d'un algorithme HVP analytique exact en temps et espace linéaires par rapport à la grille de programmation dynamique ($\mathcal{O}(NM)$).
    3. Conception d'un noyau GPU performant parallélisé en anti-diagonales.
    4. Validation empirique sur des benchmarks de performance et des tâches d'alignement de signaux.

### Section 2 : Related Work (État de l'art)
*   **Alignement temporel et Soft-DTW** : *Cuturi (2017)*, *Blondel et al. (2021)* (Soft-DTW divergences).
*   **Régularisation sur graphes et transport optimal** : Fused Gromov-Wasserstein, régularisation de graphes pour l'apprentissage multitâche.
*   **Optimisation système de la programmation dynamique** : Travaux sur l'accélération GPU des algorithmes de type Smith-Waterman ou DTW (parallélisation par anti-diagonales).

### Section 3 : Graph-Regularized Soft-DTW
*   **Formulation mathématique** de la perte globale $\mathcal{L}(X, Y)$.
*   **Modélisation du déphasage** : Rôle de la matrice $P_{\Delta_{u,v}}$ pour contraindre la vitesse de propagation physique de l'information entre deux capteurs voisins $u$ et $v$.
*   **Propriétés de la perte** : Continuité, différentiabilité par rapport aux séries d'entrée $X$ et $Y$.

### Section 4 : Algorithme HVP Analytique (Méthodologie)
*   *Note : Cette section s'appuiera directement sur les démonstrations de [hvp_mathematical_derivation.md](file:///c:/Users/moreaum/Documents/Programme/kalyx/GPUsync/examples/algorithms/gr_soft_dtw/hvp_mathematical_derivation.md).*
*   Explication des 3 passes :
    1. **Forward** : Calcul de $Z$ et $Q$ (coûts cumulés et probabilités de transition).
    2. **Backward** : Calcul de l'alignement $A$.
    3. **Double Backward** : Propagation forward de la perturbation $H$ (adjointe d'alignement), calcul des dérivées de la softmax, et remontée rétrograde de $B$ (adjointe de coût) qui donne le HVP final.
*   Preuve théorique de la complexité spatiale et temporelle optimale $\mathcal{O}(NM)$.

### Section 5 : Noyau GPU & Stratégie de Parallélisation
*   **Parallélisation par anti-diagonales** : Comment contourner la dépendance séquentielle de la DP (une cellule a besoin de son voisin du dessus, de gauche, et de diagonale) en traitant en parallèle les cellules situées sur une même anti-diagonale de la matrice.
*   **Architecture de batching** : Distribution des différents capteurs/lots (dimension $D$) sur les blocs de threads GPU pour saturer le matériel.
*   **Détail d'implémentation** : Mention de l'utilisation du compilateur JIT **GPUsync** comme outil backend permettant de générer ces noyaux hautement optimisés directement depuis Python.

### Section 6 : Expériences & Évaluation
*   **Expérience 1 : Benchmark Système (Vitesse et Mémoire)**
    *   Comparaison de temps de calcul et d'empreinte mémoire entre :
        1. Notre HVP GPU.
        2. PyTorch Autograd (CPU/GPU).
        3. JAX (avec compilation JIT JAX).
    *   Axe X : Longueur de séquence $N$ ($10 \to 1000$), Axe Y : Temps (ms) / Mémoire (MB).
*   **Expérience 2 : Robustesse sur données synthétiques**
    *   Simulation de signaux sinusoïdaux propagés le long d'un graphe linéaire avec du bruit.
    *   Montrer que GR-S-DTW reconstruit l'alignement et filtre le bruit plus efficacement que l'Independent Soft-DTW.
*   **Expérience 3 : Cas d'usage réel**
    *   *Option MoCap* : Alignement de mouvements de squelettes humains (le graphe représente les articulations physiques).
    *   *Option Réseau de Capteurs* : Alignement de signaux physiques avec délai de propagation.

---

## 3. Feuille de Route pour la Finalisation (Roadmap)

1.  **Valider les équations sur de grandes tailles** : S'assurer que le script de test `test_phase_c.py` et `test_phase_d_gpusync.py` passent pour des tailles croissantes sans divergences numériques.
2.  **Lancer le script de benchmark** : Exécuter [benchmark.py](file:///c:/Users/moreaum/Documents/Programme/kalyx/GPUsync/examples/algorithms/gr_soft_dtw/benchmark.py) et générer les graphiques de performances (temps et mémoire).
3.  **Créer le cas d'usage appliqué** : Écrire un petit script Python simulant des données de capteurs (ex: ondes se propageant sur un graphe linéaire) pour démontrer visuellement la supériorité de la perte GR-S-DTW par rapport à la Soft-DTW classique.
