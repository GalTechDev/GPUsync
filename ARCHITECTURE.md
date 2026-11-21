# GPUsync v2 — Architecture

> Statut : **M6 terminé** (20/11/2026). Ce document fixe les décisions avant d'écrire le cœur en Rust.
> Il s'appuie sur le prototype Python (`GPUsync-python/`), testé sur Apple M5 Pro (OpenCL 1.2), le 19/11/2026.

### Avancement

| Jalon | État | Validé sur |
|---|---|---|
| M0 Squelette | ✅ workspace Cargo, PyO3 0.30 (abi3-py310) + maturin 1.16, `gs.devices()` / `gs.backends()` via wgpu 30 et opencl3 0.12 (chargement dynamique) | Mac M5 Pro (Python 3.15) ; PC Windows RTX 4080 SUPER + UHD 770 (Python 3.10) |
| M1 MVP | ✅ `@gs.kernel` élément par élément, IR + interpréteur, émetteurs OpenCL C et WGSL, `gs.Array`, zéro-copie OpenCL | Mac : Metal, OpenCL, cpu (75 tests) ; PC : Vulkan et OpenCL sur RTX 4080 SUPER et UHD 770, cpu (104 tests) |
| M1.5 Transferts et choix automatique | ✅ f64 via Vulkan, besoins `f64` / `f64_math`, classement des périphériques, option `compat` ; moteur de transferts (Direct, Résident, Flux) pour OpenCL et wgpu ; zéro-copie Metal (`NoCopy` via wgpu-hal) ; `out=`, `np.memmap`, `.stream()`, `gs.OutOfMemory`, `gs.set_memory_budget` | Mac : 109 tests ; PC : 164 tests ; essais à gros volume ci-dessous |
| M2 Index et formes | ✅ `gs.index()`, dimensions nommées (`gs.f32["n", "t"]`, axes sans nom `gs.f32[:, :]`), `x.shape[k]`, `len(x)`, plusieurs sorties (`-> (gs.f32["n"], gs.i32["n"])`, `out=` en tuple), `gs.local`, réductions de `gs.Array` sur l'hôte ; moteur à plusieurs sorties ; perte du GPU par le driver (TDR) : erreur claire et réouverture | Mac : 152 tests ; PC : 228 tests ; DTW identique au prototype |
| M3 Out-of-core avec index | ✅ grille découpée par blocs de lignes (premier axe de la sortie) dans tous les modes ; entrées lues seulement à la ligne courante découpées, les autres gardées entières ; lancements cadencés (~0,1 s chacun) | Mac : 172 tests ; PC : 263 tests ; essais à gros volume ci-dessous |
| M4 Async et interop | ✅ `await y`, `await kernel(...)`, `mean, std = await stats(x)` ; appels depuis plusieurs threads (`asyncio.to_thread`) ; DLPack (export et import, mémoire hôte) ; `y.save("r.npy")` par blocs ; même code en script, asyncio et Jupyter (notebook exécuté par nbclient) ; cache disque étudié et écarté (mesures ci-dessous) | Mac : 196 tests ; PC : 301 tests |
| M5 Mode explicite | ✅ `@gs.kernel(grid="x")` ou `grid=("n", "n")`, sorties `gs.Out[gs.i32[10]]` (valeur de départ : `gs.Out[gs.f32["m"], gs.inf]`), `gs.atomic_add` / `atomic_min` / `atomic_max` en i32, u32, f32 (f32 émulé par comparaison-échange en OpenCL 1.2 et en WGSL), indices multiples `(a, b)`, accumulation avec `out=` ; écriture hors limites jamais faite et signalée (`gs.OutOfBounds`) | Mac : 239 tests ; PC : 364 tests |
| M6 CUDA | ✅ émetteur CUDA C (même générateur qu'OpenCL C, autre dialecte), backend CUDA à liaisons dynamiques maison (driver + NVRTC), moteur de transferts commun à OpenCL et CUDA (trait `Gpu`), réserve de buffers, cache disque du PTX, NVRTC par le toolkit ou `pip install nvidia-cuda-nvrtc-cu12` | PC : 430 tests, dont les 61 tests différentiels par périphérique sur `cuda:0` (parité) ; Mac : 240 tests (CUDA absent, signalé) |

#### Entiers 64 bits : `gs.i64`, `gs.u64` (21/11/2026)

- IR, interpréteur, OpenCL (`long`), CUDA (`long long`), WGSL avec l'extension `SHADER_INT64` (Vulkan, DX12,
  Metal MSL 2.3+) ; capacité `int64` par périphérique, repli `compat` sinon.
- Conversions exactes entre entiers (l'interpréteur passait par un f64 : sans perte en 32 bits, plus en 64).
- Élargissement implicite sans perte, comme les entiers de Python : i32 → i64, u32 → i64, u32 → u64 ; les autres
  mélanges restent explicites. `gs.Out` en 64 bits refusé pour l'instant (atomiques 64 bits non portables).
- **Metal** : naga 30 traduit la division et le reste 64 bits en MSL invalide (« call to 'select' is
  ambiguous ») ; le générateur les remplace, sur Metal seulement, par une division par décalages (64 tours) :
  juste mais lente, préférer OpenCL sur Mac pour ces calculs.
- Premier usage : le crible profond des nombres premiers (`exemples/premiers_gpusync`), diviseurs jusqu'à 2⁴⁰.
- Testé sur Mac (Metal, OpenCL, `cpu:0`) ; PC non validé (agent distant hors ligne).

#### Spécialisation des tailles (21/11/2026, §5.4)

| DTW 1 contre N, 2 M séries | sans | avec | gain |
|---|---|---|---|
| Mac, OpenCL | 27,3 ms | 5,1 ms | × 5,3 |
| Mac, Metal | 35,6 ms | 12,0 ms | × 3,0 |
| RTX, CUDA | 32,4 ms | 21,4 ms | × 1,5 |
| RTX, OpenCL | 36,2 ms | 24,7 ms | × 1,5 |
| RTX, Vulkan | 55,4 ms | 28,8 ms | × 1,9 |
| UHD 770, OpenCL | 691 ms | 30,7 ms | × 22 |
| UHD 770, Vulkan | 2490 ms | 62,3 ms | × 40 |

DTW tous contre tous 4000 × 4000 (fenêtre 5) : OpenCL Mac 1142 → 575 ms, CUDA 419 → 364 ms, OpenCL RTX
426 → 370 ms ; Metal et Vulkan inchangés (bornes de la fenêtre non constantes, tableaux indexés par une
variable : restent en mémoire). Contre le prototype, à 2 M : × 6,8 sur le Mac (OpenCL), × 14 sur la RTX
(`tools/compare_v1/results/dtw1n_v1_v2_spec.png`).

Corrigé au passage : après `for k in range(n)`, `k` valait `n` (au lieu de `n − 1`) partout, interpréteur
compris, et le réaffecter dans le corps changeait l'itération. Un compteur caché mène désormais la boucle,
comme en Python (test ajouté).

#### Indices au-delà de 2³¹ éléments (21/11/2026)

Dans un kernel avec index, l'indice aplati `x[i, k]` était calculé en 32 bits : un tableau de plus de
2³¹ − 1 éléments (8 Gio en f32), gardé entier sur le périphérique, faisait lire au mauvais endroit, sans
erreur ou par un « illegal memory access » (vérifié sur la RTX en CUDA, 8 Gio). Désormais :

- une **variante à indices 64 bits** (OpenCL `long`, CUDA `long long`) est compilée à la demande, seulement
  pour un appel dont un tableau dépasse 2³¹ − 1 éléments : la variante 32 bits, plus rapide, reste la norme ;
- l'interpréteur `cpu:0` calcule toujours ses indices en 64 bits ;
- wgpu refuse avec une erreur claire (pas d'entiers 64 bits en WGSL ; de toute façon, un buffer wgpu ne
  dépasse pas quelques Gio) ;
- tests : variante forcée sur tous les tests d'index (seuil interne `GPUSYNC_WIDE_INDEX_LIMIT`), et vrai cas de
  8 Gio en option (`GS_BIG=1`), passé en CUDA et sur `cpu:0`.

Ajout au passage : `GPUSYNC_DEVICE` choisit le périphérique par défaut sans toucher au code (après
`with gs.device(...)` et `gs.set_default_device(...)`).

#### Comparaison avec le prototype v1 (`tools/compare_v1`, 21/11/2026)

Mêmes calculs, même GPU, même API de calcul des deux côtés, chaque mesure dans un processus neuf ; le
prototype tourne dans son propre environnement (`.venv-v1` : numpy, pyopencl, llvmlite, cupy). Résultats
identiques partout (DTW 391 : écart nul au référentiel ; DTW 2000 : sommes égales au bit près).

| mesure | Mac, OpenCL (v1 → v2) | PC RTX, CUDA (CuPy → `cuda:0`) | PC RTX, OpenCL (v1 → v2) |
|---|---|---|---|
| import de la bibliothèque | 42 → 6 ms | 229 → 7 ms | 234 → 6 ms |
| 1er appel (compilation) | 8,3 → 3,3 ms | 59 → 52 ms | **35 → 66 ms** |
| surcoût par appel | 3,28 → 0,25 ms (× 13) | 0,28 → 0,08 ms (× 3,3) | 0,67 → 0,06 ms (× 11) |
| 10 M éléments | 4,7 → 0,9 ms (× 5) | **10,0 → 11,3 ms** | 28,6 → 13,9 ms (× 2) |
| 1 Gio (durée ; mémoire en plus) | 0,09 → 0,03 s ; non mesurable sous macOS | 0,40 s ; 1,02 Gio → 0,31 s ; 0,27 Gio | **0,28 s ; 0,01 Gio → 0,38 s ; 0,28 Gio** |
| DTW 391 séries OXYGENE | 959 → 2,5 ms (× 380) | 60 → 0,95 ms (× 63) | 1241 → 1,0 ms (× 1218) |
| DTW 2000 × 64 | 5023 → 286 ms (× 18) | 568 → 104 ms (× 5,5) | 2484 → 105 ms (× 24) |

En gras : là où le prototype fait mieux. Le DTW 391 du prototype est mesuré comme son propre script
(`run_xlsx_dtw.py` : 25 appels de 16 références, avec la machinerie hors mémoire à chaque appel) ; le DTW
2000 reflète mieux le débit des kernels. Pistes : sur GPU dédié en OpenCL, le prototype transfère 1 Gio sans
tampon épinglé intermédiaire (plus rapide, presque sans mémoire en plus) ; en CUDA, CuPy copie 10 M éléments
directement depuis la mémoire de l'hôte. Les deux reviennent au « dernier `memcpy` » de §9.3.

##### DTW 1 contre N, jusqu'à 2 M séries (benchmark DTW du prototype)

Repris de `examples/algorithms/dtw/benchmark.py` du prototype : une référence, N cibles, séries de 30 points
complétées à 32 par des zéros (le DTW porte donc sur 32 points), sans fenêtre. Variantes du prototype : « OOC
Manuel » (blocs de 20 000, deux en vol) et « OOC Auto » (`compute_dtw_matrix`). Résultats vérifiés entre
versions ; une variante dont les résultats diffèrent n'est pas retenue dans le gain.

| N | Mac, v1 Manuel / Auto (OpenCL) | Mac, v2 OpenCL / Metal | RTX, v1 Manuel (OpenCL) | RTX, v2 CUDA / OpenCL / Vulkan |
|---|---|---|---|---|
| 100 000 | 7,5 / 9,4 ms | 3,2 / 3,4 ms | 18,4 ms | 1,8 / 2,5 / 2,3 ms |
| 500 000 | 20,7 / 17,7 ms | 7,1 / 9,5 ms | 83,3 ms | 9,2 / 11,8 / 12,9 ms |
| 1 000 000 | 40,1 / 20,5 ms | 14,1 / 18,5 ms | 204 ms | 16,8 / 19,5 / 26,3 ms |
| 2 000 000 | 87,7 / 35,6 ms | 27,2 / 36,2 ms | 307 ms | 32,4 / 35,7 / 55,3 ms |

- **Mac** : v2 OpenCL 1,3 à 2,5 fois plus rapide que la meilleure variante du prototype ; Metal à égalité à 2 M.
- **RTX** : v2 7 à 12 fois plus rapide que la seule variante correcte du prototype.
- **Défauts du prototype trouvés** : en CUDA, ce kernel ne compile pas (`mad`, fonction OpenCL, émise telle
  quelle en CUDA) ; en OpenCL sur la RTX, « OOC Auto » rend des distances fausses sans erreur (premières valeurs
  toutes identiques, sommes décalées de 2,6e-4) alors qu'elle est juste sur le Mac.
- **Piste pour v2** : sur ce kernel très calculatoire, le prototype déroule le DTW avec une taille constante (32)
  et garde la ligne en registres ; v2 boucle sur une taille lue à l'exécution avec un tableau local. La
  spécialisation sur les tailles connues à l'appel (§5.4) comblerait cet écart.

#### Mesures M6 (RTX 4080 SUPER, à chaud ; mêmes scripts qu'aux jalons précédents)

| | `cuda:0` | `opencl:0` | `vulkan:0` |
|---|---|---|---|
| surcoût par appel, 1000 éléments (`tools/probe/cuda_bench.py`) | **0,097 ms** | 0,140 ms | 0,164 ms |
| 10 M éléments, aller-retour | 11,3 ms | 13,9 ms | 11,2 ms |
| DTW 4000 × 4000, t = 64 | 419 ms | 423 ms | 1129 ms |
| 24 Gio en RAM, mode par blocs (`bench_m15`) | 7,1 s, 0,60 Gio en plus | 5,9 s, 0,57 Gio | 7,3 s, 0,31 Gio |
| `pairwise_dist` 72 000², sortie 19,3 Gio (`bench_m3`) | 3,9 s, 0,18 Gio en plus | 3,9 s | 3,8 s |
| somme par case f32, 100 M (`bench_m5`) | **66 ms** (atomique natif) | 363 ms | 472 ms |
| histogramme 16 cases, 100 M | 48 ms | 50 ms | 59 ms |
| 1er appel d'un processus (`bench_m4`) : sans cache / avec cache PTX | 152 ms / 62 ms | 58 ms / 68 ms | 44 ms / 48 ms |

#### Décisions et constats de M6

- **Liaisons maison plutôt que `cudarc`** (écart au plan) : `cudarc` panique quand une bibliothèque manque,
  fige à la compilation les noms de bibliothèques d'une version de CUDA, et ne cherche ni dans `CUDA_PATH` ni
  dans les roues pip. Une trentaine de fonctions suffisent (driver et NVRTC), chargées par `libloading` :
  aucune panique, une erreur claire, rien à installer à la compilation (le crate compile aussi sur macOS).
- **NVRTC** n'est pas dans le driver. Recherche : `GPUSYNC_NVRTC`, roues pip (`nvidia-cuda-nvrtc-cu12`,
  dossiers transmis par le package Python), toolkit (`CUDA_PATH*`), PATH. S'il manque, le backend CUDA est
  indisponible avec la commande `pip install` à lancer, et le choix automatique passe à Vulkan. Testé : sans
  toolkit, avec la roue seule.
- **Moteur commun** : les modes Résident, par blocs et par blocs de lignes sont écrits une fois, pour OpenCL et
  CUDA (trait interne `Gpu` : allocation, tampons épinglés, copies et lancements chaînés par événements).
- **Surcoût par appel** : 0,325 ms au départ, à cause de `cuMemAlloc` / `cuMemFree` (qui synchronise tout le
  GPU) à chaque appel. Une réserve de buffers (rendus avec un événement par stream, réutilisés une fois ces
  événements passés) le ramène à 0,097 ms, le plus bas des trois backends sur la RTX.
- **Cache disque du PTX** : la compilation NVRTC coûte ~90 ms par kernel et par processus ; le PTX est gardé
  dans le cache utilisateur (`GPUSYNC_CACHE_DIR`, `GPUSYNC_CACHE=0` pour le couper), le driver mettant déjà
  en cache la suite. Fichier abîmé : recompilé sans erreur.
- **Non fait à M6** : mode Direct pour CUDA (Jetson, mémoire partagée : traité comme un GPU dédié, aucun
  matériel pour tester) ; export DLPack de buffers CUDA (le résultat est exporté depuis la mémoire hôte) ;
  `cuMemHostRegister` sur les tableaux NumPy (le `memcpy` vers l'anneau épinglé reste, comme en OpenCL).

#### Mesures M5 (`tools/bench_m5.py`, 100 M éléments, à chaud, résultat lu et vérifié)

| | NumPy | `metal:0` | `opencl:0` Mac | `vulkan:0` RTX | `opencl:0` RTX | `vulkan:1` UHD | `opencl:1` UHD |
|---|---|---|---|---|---|---|---|
| histogramme, 16 cases (i32) | 239 ms (Mac) / 396 ms (PC) | 22 ms | 5,3 ms | 60 ms | 50 ms | 193 ms | 117 ms |
| somme par case, 1000 cases (f32 émulé) | 104 ms (Mac) / 351 ms (PC) | 161 ms | 146 ms | 473 ms | 229 ms | 297 ms | 148 ms |

Les atomiques entiers sont natifs et rapides, même très disputés (16 cases). Les atomiques flottants, émulés
par comparaison-échange, ne battent pas NumPy sur ce cas très disputé ; sur la RTX, l'envoi de 800 Mo par PCIe
pèse aussi (~100 ms). Pistes : atomiques flottants natifs (`cl_ext_float_atomics`, Metal `atomic_float`, CUDA
à M6), agrégation par groupe de travail en mémoire locale.

#### Décisions et constats de M5 (D7 tranchée)

- **Syntaxe validée sur 5 cas réels** (histogramme, histogramme 2D, somme / min / max par case, comptage de
  paires sur une grille `("n", "n")`) : `grid=` + `gs.Out[...]` + `gs.atomic_*`. Deux ajouts : la valeur de
  départ (`gs.Out[gs.f32["m"], gs.inf]`, indispensable pour un minimum) et `out=` qui part du contenu donné
  (accumulation sur plusieurs appels, ex. un flux de blocs).
- **Écritures directes refusées**, sans exception `unsafe_writes` : toute écriture dans un `gs.Out` passe par
  un atomique, et un `gs.Out` est en écriture seule. Aucun cas réel n'a demandé autre chose.
- **Bornes vérifiées** : chaque atomique teste son indice ; une écriture hors limites n'est jamais faite (pas de
  mémoire corrompue), elle est notée dans un petit tableau caché `gs_err`, et la lecture du résultat lève
  `gs.OutOfBounds` (sortie en cause, exemple d'indice, forme). Les autres écritures sont faites. Le cas
  classique : `int(x * 10)` pour `x = 1.0`.
- **Moteur** : les `gs.Out` restent entiers sur le périphérique (sans copie en mode Direct) pendant que les
  entrées sont découpées par blocs de lignes comme au M3. Un défaut corrigé au passage : sur GPU dédié en mode
  par blocs, la réutilisation d'un emplacement n'attendait la fin du calcul qu'à travers le retour de ses
  sorties ; sans sortie par case, des comptes se perdaient (vu sur la RTX en OpenCL). L'attente est maintenant
  explicite.
- **f64** : refusé pour les `gs.Out` (pas d'atomiques 64 bits portables).

#### Mesures M4 (`tools/bench_m4.py`, processus neufs)

| Périphérique | 1er appel du processus | appel suivant | 1er appel d'un 2e processus |
|---|---|---|---|
| Mac `metal:0` | 5,3 ms | 0,7 ms | 5,2 ms |
| Mac `opencl:0` | 5,1 ms | 0,6 ms | 5,0 ms |
| PC `vulkan:0` (RTX) | 59 ms | 0,4 ms | 51 ms |
| PC `opencl:0` (RTX) | 69 ms | 0,2 ms | 62 ms |
| PC `opencl:1` (UHD 770) | 506 ms | 3,9 ms | 7,8 ms |

Le premier appel paie surtout l'**ouverture du périphérique** (40 à 75 ms sur le PC) ; la compilation d'un
kernel ne coûte que 0,1 à 2 ms une fois le périphérique ouvert. Les drivers ont déjà leur cache de binaires :
l'UHD 770 passe de 506 ms à 7,8 ms dans un deuxième processus, NVIDIA de même (`ComputeCache`).
`gs.devices()` coûte ~340 ms sur le PC (découverte Vulkan et OpenCL par les drivers), ~35 ms sur le Mac.

#### Décisions et constats de M4

- **Cache disque des kernels : non fait.** Les mesures ci-dessus montrent qu'il ne ferait rien gagner :
  la compilation est déjà mise en cache par les drivers, et le reste du premier appel est l'ouverture du
  périphérique. Repris à M6 pour CUDA : NVRTC coûte ~90 ms par kernel, d'où un cache disque du PTX.
- **`await`** : un appel met toujours le travail en file de façon synchrone (les erreurs arrivent à l'appel,
  y compris dans Jupyter) ; `await y` attend la fin et la lecture sur un thread de l'exécuteur asyncio (GIL
  libéré), puis rend `y`. Plusieurs sorties : un tuple attendable (`Results`). Ce qui s'exécute dans l'appel
  lui-même (mode par blocs, `cpu:0`) se met hors de la boucle avec `await asyncio.to_thread(f, x)` : les
  appels sont sûrs depuis plusieurs threads (testé sur 8 threads, même kernel, même périphérique).
- **Lancements cadencés, version 2** : les lancements ne sont chronométrés que jusqu'à une mesure fiable
  (≥ 20 ms) ; les suivants partent en file sans attente. L'appel rend la main plus tôt, `await` a du sens
  même pour de gros calculs, et le DTW 4000 × 4000 sur l'UHD 770 reste à 56,6 s sans perte du GPU.
- **DLPack** : export de la mémoire hôte du résultat (sans copie ; `np.from_dlpack`, `torch.from_dlpack`),
  import de tout objet DLPack lisible depuis l'hôte. Un tenseur resté sur un GPU donne une erreur qui
  suggère `.cpu()`. L'échange direct de buffers GPU attend le backend CUDA (M6).
- **Jupyter** : testé pour de vrai (notebook exécuté par nbclient, kernels définis dans une cellule,
  `await` au niveau de la cellule). Dépendances de développement ajoutées : `nbclient`, `ipykernel`.

#### Mesures M3 (`tools/bench_m3.py`, 64 lignes vérifiées contre NumPy)

| Cas | Machine, périphérique | Mode | Durée | Mémoire en plus | Résultat |
|---|---|---|---|---|---|
| `pairwise_dist` 72 000 × 72 000 (t = 16), sortie 19,3 Gio > 16 Go de VRAM | PC, `vulkan:0` (RTX) | par blocs (`flux`) | 3,8 s | 0,34 Gio | ✓ |
| idem | PC, `opencl:0` (RTX) | par blocs (`flux`) | 3,9 s | 0,16 Gio | ✓ |
| idem | PC, `opencl:1` (UHD 770) | Direct | 29,5 s | 0 | ✓ |
| idem | PC, `vulkan:1` (UHD 770) | par blocs (`flux`) | 34,5 s | 0,51 Gio | ✓ |
| `pairwise_dist` 56 000 × 56 000, sortie 11,7 Gio > allocation max (3,3 Gio) | Mac, `opencl:0` | Direct | 1,2 s | 0 | ✓ |
| idem | Mac, `metal:0` | Direct | 0,6 s | 0,01 Gio | ✓ |
| DTW 4000 × 4000 (t = 64), perdu par le driver à M2 (TDR) | PC, `vulkan:1` (UHD 770) | Résident, lancements cadencés | 56,7 s | — | ✓ |

#### Décisions et constats de M3

- **Découpage par lignes** : le frontend repère, dans le source, les tableaux lus seulement à la ligne
  courante (premier indice = la variable `i` de `i, … = gs.index()`, jamais réaffectée, et même première
  dimension que la sortie). Ceux-là sont découpés avec la sortie ; leur premier indice devient `i − inrow`.
  Les autres (ex. `series[j, k]` dans `pairwise_dist`) restent entiers sur le périphérique.
- **Trois paramètres cachés du moteur** : `row0` (première ligne du lancement), `inrow` (première ligne des
  entrées découpées présentes), `obase` (case de départ dans le buffer de sortie). Le même code sert à tous
  les modes : Direct (blocs importés sans copie, débuts alignés sur 16 Kio pour Metal), Résident (buffers
  entiers, plusieurs lancements) et par blocs (OpenCL : deux emplacements, le retour d'un bloc recouvre le
  calcul du suivant ; wgpu : à la suite).
- **Lancements cadencés** : au-delà de 65 536 cases, un premier petit lancement est chronométré, puis chaque
  lancement vise ~0,1 s. Plus aucun lancement ne s'approche du délai de garde des drivers ; coût mesuré :
  ~1 ms sur un DTW de 391 × 391 (Metal : 3,0 → 4,1 ms), invisible au-delà.
- **Entrée trop grosse pour rester entière** : `gs.UnsupportedFeature` qui la nomme, donne sa taille et la
  place disponible, et explique comment la rendre découpable.

#### Mesures M2 (`tools/bench_m2.py`, à chaud, résultat lu sur l'hôte)

DTW (fenêtre 5) entre toutes les paires de séries ; écart avec le prototype ≤ 2e-7 en relatif (identique au
bit près en OpenCL sur le Mac) sur les 391 séries `OXYGENE` (`tests/data`).

| Cas | `metal:0` | `opencl:0` Mac | `vulkan:0` RTX | `opencl:0` RTX | `vulkan:1` UHD | `opencl:1` UHD | `cpu:0` |
|---|---|---|---|---|---|---|---|
| 391 × 391, séries de 21 | 3,0 ms | 1,9 ms | 1,3 ms | 0,9 ms | 81 ms | 103 ms | 6,3 s (Mac) |
| 4000 × 4000, séries de 64 | 1,41 s | 1,12 s | 1,13 s | 0,42 s | perdu (TDR) | 73 s | — |

#### Décisions et constats de M2

- **Un seul bloc pour les kernels avec index** (levé à M3) : la grille couvrait toute la sortie ; au-delà du
  budget ou d'un buffer, erreur claire renvoyant au M3.
- **Dimensions cachées** : chaque dimension distincte (nommée, ou `param.axe` sans nom) devient un paramètre
  `i32` caché, fourni par le runtime depuis les formes ; les indices `x[i, k]` sont aplatis en ordre C.
- **TDR Windows** : sur l'UHD 770, un DTW 4000×4000 dépasse ~2 s en Vulkan ; le driver perd le périphérique et
  wgpu paniquait à l'appel suivant. Désormais : `gs.DeviceError` explicite, puis le périphérique et les kernels
  compilés sont rouverts au prochain appel. OpenCL sur le même GPU n'est pas interrompu (73 s). Les
  lancements cadencés de M3 suppriment ce cas (56,7 s en Vulkan, sans perte).
- **Vulkan 2,7 fois plus lent qu'OpenCL sur la RTX** pour le DTW (tableaux locaux de 257 éléments par case) :
  piste à mesurer (initialisation et contrôles de bornes des tableaux `var<function>` en WGSL).

#### Mesures M1.5 (`tools/bench_m15.py`, résultat vérifié en entier)

| Cas (§9.8) | Machine, périphérique | Mode | Durée | Mémoire en plus | Résultat |
|---|---|---|---|---|---|
| 4 Gio en RAM (> allocation max 3,3 Gio) | Mac, `opencl:0` | Direct | 0,5 s | ≈ 0 | ✓ |
| 4 Gio en RAM | Mac, `metal:0` | Direct | 0,3 s | 0,14 Gio | ✓ |
| 40 Gio en memmap, `out=` memmap (> 26 Go de RAM) | Mac, `metal:0` | Direct | 35 s (disque) | 4,1 Gio de cache de fichiers, récupérable | ✓ |
| 2 Gio (> liaison max 1 Gio) | PC, `vulkan:1` (UHD 770) | Flux | 1,0 s | 1,24 Gio | ✓ |
| 2 Gio | PC, `vulkan:0` (RTX) | Flux | 0,9 s | 0,31 Gio | ✓ |
| 2 Gio | PC, `opencl:0` (RTX) | Résident | 0,9 s | 0,38 Gio | ✓ |
| 2 Gio | PC, `opencl:1` (UHD 770) | Direct | 1,0 s | 0,10 Gio | ✓ |
| 24 Gio en RAM (> 16 Go de VRAM) | PC, `vulkan:0` (RTX) | Flux | 7,3 s (7,0 Go/s) | 0,31 Gio | ✓ |
| 24 Gio en RAM | PC, `opencl:0` (RTX) | Flux | 5,9 s (8,7 Go/s) | 0,57 Gio | ✓ |
| 20 Gio en memmap, `out=` memmap | PC, `vulkan:0` et `opencl:0` (RTX) | Flux | 48 et 51 s (disque) | pages des deux fichiers, récupérables | ✓ |

Mémoire en plus : pic de mémoire physique (résidente sur macOS, working set sous Windows), moins l'entrée et la
sortie en RAM. Sous macOS, la mémoire épinglée par OpenCL `USE_HOST_PTR` n'apparaît pas dans la mémoire
résidente : seul le mode Direct de Metal y est mesurable. Comparaison M1 (1 Gio, avant le moteur) : 3,01 Gio
en Metal, 5,23 Gio en Vulkan sur la RTX, 3,35 Gio en OpenCL sur la RTX.

#### Décisions et constats de M1.5

- **Numérotation stable** : l'ordre des plateformes OpenCL change d'un environnement à l'autre (constaté sous
  Windows : RTX puis UHD 770, puis l'inverse). Les périphériques sont désormais triés (dédiés, intégrés, CPU,
  puis nom) : `opencl:0` désigne toujours le même GPU.
- **Metal `NoCopy`** accepte aussi les pages d'un fichier `memmap` : 40 Gio traités en mode Direct.
- **wgpu sur la RTX** : une liaison est limitée à ~2 Gio, d'où le mode Flux dès 2 Gio même dans le budget.
- **Mémoire bornée** : sur la RTX, 0,31 Gio en plus à 2 Gio comme à 24 Gio en Vulkan (0,38 et 0,57 Gio en
  OpenCL) : le surcoût ne dépend plus du volume.
- **Outil distant** (v7) : la CLI attend le retour d'un agent injoignable (10 min) et l'agent réessaie
  pendant 5 min. Reste un problème non résolu : les tâches de plus de ~12 min via l'agent Windows se figent
  (l'agent ne donne plus signe de vie, sans plantage ni mise en veille du PC). Les essais longs ont été
  lancés en processus détaché.

#### Mesures M1 (`tools/bench_m1.py`)

Surcoût par appel, de bout en bout (appel + `np.asarray`), 1 000 éléments — objectif ≤ 0,3 ms :

| | Metal M5 Pro | OpenCL M5 Pro | Vulkan RTX | Vulkan UHD 770 | OpenCL RTX | OpenCL UHD 770 |
|---|---|---|---|---|---|---|
| ms / appel | 0,196 | 0,193 | 0,095 | 0,183 | 0,272 | 0,149 |

Prototype v1 sur le M5 Pro (OpenCL) : 3,07 ms ; pyopencl brut : 0,30 ms.

Pic mémoire supplémentaire pour 1 Gio d'entrée (la sortie seule compte pour 1,00) — objectif ×1 en OpenCL sur le Mac :

| | Metal M5 Pro | OpenCL M5 Pro | Vulkan RTX | OpenCL RTX | OpenCL UHD 770 |
|---|---|---|---|---|---|
| Gio | 3,01 | **1,00** (zéro-copie) | 5,23 | 3,35 | **1,00** (zéro-copie) |

Mesure : mémoire résidente (macOS), mémoire engagée (Windows). Sur 10 M éléments, OpenCL en zéro-copie
traite 83 Go/s sur le M5 Pro contre 9,5 Go/s pour Metal (copies).

#### Mesures f64 via wgpu (sonde sur le PC, wgpu 30)

| GPU / API | `SHADER_F64` | Résultat |
|---|---|---|
| RTX 4080 SUPER, Vulkan | ✅ | arithmétique, `//`, `%`, comparaisons, `sqrt`, `floor`, `ceil`, `abs`, `min`, `max`, `fma`, conversions : **précision f64** (≤ 2·10⁻¹⁶) |
| RTX 4080 SUPER, Vulkan | ✅ | `exp` : accepté mais **calculé en précision f32** (1,7·10⁻⁷), sans erreur ; `log` : **le pilote plante** (violation d'accès) |
| RTX 4080 SUPER, Vulkan | ✅ | paramètre f64 par `bitcast<f64>(vec2<u32>)` : refusé par naga ; par un buffer `array<f64>` : correct |
| RTX 4080 SUPER et UHD 770, DX12 | ❌ | — |
| UHD 770, Vulkan ; M5 Pro, Metal | ❌ | pas d'unités f64 dans le matériel |

Les fonctions transcendantes en f64 (GLSL.std.450 ne les définit qu'en 16 et 32 bits) sont donc **refusées
avant compilation** sur Vulkan : sinon, résultat faux silencieux ou plantage.

#### Décisions prises pendant M1

- **Sémantique** : `//` et `%` suivent Python et NumPy (arrondi vers −∞, signe du diviseur), y compris sur
  le GPU ; division entière par zéro → 0 (comme NumPy) ; `/` entre entiers donne un f32 ; `**` réservé aux
  flottants ; `pow` corrigé en WGSL pour une base négative.
- **Typage** : un littéral prend le type de son contexte (`x * 2` reste f32) ; une locale prend le type de ses
  affectations (`acc = 0` puis `acc += x` → f32) ; aucune promotion implicite vers f64.
- **f64 via wgpu** : à M1, `Device.f64` vaut `False` pour Metal, Vulkan et DX12. C'était trop restrictif :
  wgpu active le f64 sur Vulkan (`SHADER_F64`), à condition d'éviter les fonctions mathématiques. Corrigé
  à M1.5 (§6.2, mesures ci-dessous).
- **`cpu:0`** : l'interpréteur de référence est exposé comme périphérique, toujours disponible, en dernier.
- **Taille des buffers wgpu** : limitée à ~1 Gio sur l'UHD 770 en Vulkan. Un tableau trop grand lève
  `gs.UnsupportedFeature` (avant correction, wgpu renvoyait un résultat faux sans erreur). Le découpage
  automatique viendra avec l'out-of-core (M3).
- **Entrée D10** : wgpu copie et consomme 3 à 5 fois la taille des données ; OpenCL en zéro-copie, aucune.
  Sur Mac, Metal reste le défaut (même surcoût par appel), mais les gros volumes ont intérêt à passer par
  OpenCL : à trancher avec l'out-of-core (M3).

Écarts au plan, à M0 : seuls les crates nécessaires existent (`gpusync-runtime`, `gpusync-backend-wgpu`,
`gpusync-backend-opencl`, `gpusync-py`) ; `gpusync-ir`, `gpusync-frontend` et `gpusync-codegen` seront créés
en M1. Sous Windows, un GPU visible en Vulkan et en DX12 n'est listé qu'une fois (Vulkan). D2 tranchée :
Python ≥ 3.10, testé aux deux bouts (3.10 sur le PC, 3.15 sur le Mac).

---

## 1. Objectifs

**GPUsync permet à un dev Python d'écrire un kernel GPU en Python, de l'appeler comme une fonction, et d'obtenir un résultat correct sur n'importe quel GPU, sans rien connaître au GPU.**

Objectifs mesurables :

| # | Objectif | Critère |
|---|---|---|
| O1 | Installation triviale | `pip install gpusync` réussit sur macOS, Linux et Windows sans driver GPU particulier ; l'import ne plante jamais |
| O2 | Correction | Résultats identiques à NumPy (à la tolérance flottante près) sur tous les backends ; **jamais** de résultat faux silencieux |
| O3 | Overhead faible | ≤ 0,3 ms par appel sur un petit kernel (M5 Pro). Prototype : 3,1 ms ; pyopencl brut : 0,3 ms |
| O4 | Portabilité | Metal, Vulkan, DX12 (via wgpu) et OpenCL ≥ 1.2 dès le MVP ; CUDA ensuite |
| O5 | Out-of-core | Données plus grandes que la mémoire GPU traitées avec le même appel |
| O6 | Lisibilité | Un message d'erreur dit ce qui ne va pas, où, et comment le corriger |
| O7 | Pas de duplication sur mémoire unifiée | Sur Apple Silicon, GPU intégrés et Jetson, un tableau NumPy n'existe qu'une fois en RAM, en entrée comme en sortie, quand le backend le permet (voir §8.4) |

### Non-objectifs (pour l'instant)

- Remplacer PyTorch / JAX : pas d'autodiff, pas de bibliothèque de tenseurs complète.
- Accepter du Python arbitraire dans un kernel : on supporte un **sous-ensemble** clairement documenté.
- Fusion automatique de kernels et ordonnancement multi-GPU avancé (prévus en fin de feuille de route, pas dans le MVP).

---

## 2. Leçons du prototype

| Constat dans GPUsync v1 | Décision pour la v2 |
|---|---|
| 4 transpileurs écrits à la main, chacun depuis l'AST Python (CUDA, OpenCL, LLVM, PTX). LLVM échoue déjà sur `min`, `get_local_id` | **Une seule IR typée**, plusieurs émetteurs simples qui partent de l'IR, pas de l'AST |
| 5 façons de choisir le matériel (`Scheduler`, `SwarmScheduler`, `MultiDeviceScheduler`, `use_gpu`, scheduler par défaut) | **Un seul concept : `Device`**, choisi automatiquement par défaut |
| Boucle asyncio en arrière-plan + `auto_sync` + callback OpenCL par étape : ~1,1 ms par attente, ~3 ms par appel | **Le runtime Rust gère files et attentes.** Pas de boucle asyncio interne. On n'attend qu'à la lecture des données |
| `auto_sync` renvoie une coroutine dès qu'une boucle tourne → `y = f(x)` casse dans Jupyter ; un `await` oublié passe inaperçu | **`gs.Array`** : un seul objet à la fois attendable et lisible directement |
| Files en round-robin : upload, kernel et relecture sur des files différentes → synchro obligatoire à chaque étape | **Une file ordonnée par périphérique** et dépendances suivies par le runtime |
| Sorties toujours en float32 ; un tableau float64 donne des résultats faux sans erreur | **Types obligatoires dans la signature**, vérifiés à l'appel, capacités du périphérique vérifiées à la compilation |
| `cupy-cuda12x` en dépendance obligatoire → install impossible sur Mac | **Aucune dépendance GPU côté Python.** Les bibliothèques natives sont chargées dynamiquement au démarrage |
| `AsyncBuffer.__del__` remet le buffer au pool immédiatement, même si une opération l'utilise encore ; pool jamais borné | Un buffer ne retourne au pool **qu'après la fin des opérations en attente** ; pool avec budget mémoire |
| Callback OpenCL Apple : ~1,1 ms ; attente directe (`wait`) : ~0,25 ms | Attente **directe** sur l'événement (GIL libéré), pas de callbacks |
| Copie systématique hôte ↔ GPU, même quand la mémoire est unifiée (Apple Silicon) : données en double et copies inutiles | **Zéro-copie sur mémoire unifiée.** Mesuré sur M5 Pro, 256 Mo : 7,4 ms avec `USE_HOST_PTR` contre 33,3 ms avec copie |
| ✅ Écrire ses kernels en Python | **Conservé**, c'est la raison d'être de la lib |
| ✅ Out-of-core en une ligne, y compris depuis un générateur | **Conservé et généralisé** (voir §8) |
| ✅ Garder sur le GPU les arguments constants en out-of-core | **Généralisé** : toute entrée qui ne dépend pas de la dimension découpée reste résidente |
| ✅ Pool de buffers : plus rapide que pyopencl brut sur 10 M éléments | **Conservé**, dans le runtime Rust |

---

## 3. Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────────┐
│ Python : package `gpusync` (mince)                                    │
│   @gs.kernel · types gs.f32[...] · gs.Array · gs.devices() · erreurs │
└───────────────┬──────────────────────────────────────────────────────┘
                │ PyO3 (abi3)
┌───────────────▼──────────────────────────────────────────────────────┐
│ gpusync-py : liaison Python                                           │
│   lit l'`ast` Python → KernelAst (structure Rust neutre)              │
│   convertit NumPy / DLPack ↔ buffers · libère le GIL · __await__       │
└───────┬──────────────────────────────────────────┬───────────────────┘
        │                                          │
┌───────▼───────────────┐                ┌─────────▼──────────────────┐
│ gpusync-frontend      │                │ gpusync-runtime             │
│  KernelAst → IR        │                │  Device · Stream · Buffer   │
│  inférence de types    │                │  pool mémoire · cache       │
│  dimensions nommées    │                │  graphe différé · events    │
│  validation            │                │  out-of-core                │
└───────┬───────────────┘                └─────────┬──────────────────┘
        │                                          │ trait Backend
┌───────▼───────────────┐      ┌───────────────────┼───────────────────┐
│ gpusync-ir            │      │                   │                   │
│  IR typée structurée   │  ┌───▼──────────┐ ┌──────▼───────┐ ┌─────────▼────┐
│  capacités requises    │  │ backend-wgpu │ │backend-opencl│ │ backend-cuda │
│  interpréteur CPU      │  │ Metal/Vulkan │ │  OpenCL ≥1.2 │ │   (plus tard)│
└───────┬───────────────┘  │ /DX12        │ │              │ │              │
        │                  └──────▲───────┘ └──────▲───────┘ └─────────▲────┘
┌───────▼───────────────┐         │ WGSL           │ OpenCL C          │ CUDA C
│ gpusync-codegen       ├─────────┴────────────────┴───────────────────┘
│  IR → WGSL / OpenCL C │
│  / CUDA C             │
└───────────────────────┘
```

### Workspace Cargo

```
gpusync/
├── Cargo.toml                    # workspace
├── pyproject.toml                # maturin
├── crates/
│   ├── gpusync-ir/               # types, IR, capacités, interpréteur — aucune dépendance GPU
│   ├── gpusync-frontend/         # KernelAst → IR (aucune dépendance à PyO3)
│   ├── gpusync-codegen/          # IR → texte de kernel, un module par cible
│   ├── gpusync-runtime/          # trait Backend, Device, Stream, Buffer, pool, graphe, out-of-core
│   ├── gpusync-backend-wgpu/
│   ├── gpusync-backend-opencl/
│   ├── gpusync-backend-cuda/     # jalon M6
│   └── gpusync-py/               # cdylib PyO3 : seul crate qui dépend de Python
├── python/gpusync/               # couche Python pure : décorateur, annotations de types, __init__
└── tests/                        # tests Python de bout en bout
```

**Règle :** seul `gpusync-py` dépend de PyO3. Le cœur est testable en Rust pur, et réutilisable un jour depuis Rust directement.

---

## 4. API Python

### 4.1 Le modèle mental

> **Un kernel s'écrit du point de vue d'une case de la sortie.**
> « Je suis la case `(i, j)`. Je lis ce dont j'ai besoin dans les entrées, et je renvoie ma valeur. »

- Chaque thread **lit** où il veut dans les entrées.
- Chaque thread **écrit une seule case** : la sienne, via `return`.
- Donc : aucun conflit d'écriture, aucun verrou, résultat déterministe.
- **La grille (nombre de threads) est la forme de la sortie.**

### 4.2 Kernel élément par élément

```python
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std
```

- Sans `gs.index()` dans le corps, le kernel est appliqué à chaque élément : `x` désigne la valeur courante.
- La sortie a la forme de l'entrée. `gs.f32[:]` = une dimension sans nom.
- Les scalaires (`mean`, `std`) sont passés tels quels à tous les threads.

### 4.3 Kernel avec index et dimensions nommées

```python
@gs.kernel
def pairwise_dist(series: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
    i, j = gs.index()
    acc = 0.0
    for k in range(series.shape[1]):
        d = series[i, k] - series[j, k]
        acc += d * d
    return gs.sqrt(acc)

dist = pairwise_dist(series)      # (2000, 365) → n=2000, t=365 → sortie (2000, 2000)
```

À l'appel :
1. Les dimensions nommées sont liées depuis les formes des entrées (`n=2000`, `t=365`).
2. Une même dimension doit avoir la même taille partout, sinon `gs.ShapeMismatch` avant tout lancement.
3. La sortie est allouée à la forme déclarée ; la grille vaut cette forme.
4. Une dimension de sortie absente des entrées est une erreur de déclaration (un paramètre `dims={"k": 10}`
   pourra la fournir plus tard).
5. Un axe sans nom (`gs.f32[:, :]`) a une taille libre, lisible par `x.shape[k]` (`k` négatif accepté) ;
   `len(x)` vaut `x.shape[0]`. Toutes les sorties portent des noms.
6. Indexation : `x[i, k]` (un indice par dimension, ordre C) ; un indice hors limites donne un résultat
   indéfini sur GPU et une erreur dans l'interpréteur `cpu:0`.

Tableau local à chaque case : `buf = gs.local(gs.f32, 64)` (taille constante, mis à zéro), indexé par
`buf[k]` en lecture et en écriture.

### 4.4 Plusieurs sorties

```python
@gs.kernel
def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
    i = gs.index()
    s = 0.0; s2 = 0.0
    for k in range(x.shape[1]):
        s += x[i, k]; s2 += x[i, k] * x[i, k]
    m = s / x.shape[1]
    return m, gs.sqrt(s2 / x.shape[1] - m * m)

mean, std = stats(series)
```

Les sorties d'un même kernel doivent avoir la même forme (une grille unique). Sinon : deux kernels, ou le mode explicite (§4.5).

### 4.5 L'exception : grille explicite et écritures atomiques (M5)

Quand le calcul part naturellement de l'entrée (histogramme, scatter) :

```python
@gs.kernel(grid="data")                       # un thread par élément de `data`
def histogram(data: gs.f32["n"], counts: gs.Out[gs.i32[10]]):
    i = gs.index()
    gs.atomic_add(counts, gs.min(int(data[i] * 10), 9), 1)

counts = histogram(data)                      # gs.Out : alloué et mis à zéro par la lib

@gs.kernel(grid=("n",))                       # grille : la dimension « n » (pas toute la forme de xy)
def per_bin(xy: gs.f32["n", 2], idx: gs.i32["n"], lo: gs.Out[gs.f32["m"], gs.inf], ref: gs.i32["m"]):
    i = gs.index()
    gs.atomic_min(lo, idx[i], xy[i, 0])       # valeur de départ gs.inf

acc = np.zeros(10, np.int32)
for bloc in blocs:
    histogram(bloc, out=acc)                  # out= : part du contenu donné (accumulation)
```

- `grid=` désigne une entrée (toutes ses dimensions) ou un tuple de dimensions nommées (ou entières).
- Les sorties sont des paramètres `gs.Out[type]` ou `gs.Out[type, départ]`, absents de l'appel ; leurs
  dimensions sont des entiers ou des noms de dimensions d'entrée. Plusieurs `gs.Out` : un tuple de résultats.
- Écriture seulement par `gs.atomic_add`, `gs.atomic_min`, `gs.atomic_max` (i32, u32, f32), indice entier ou
  tuple ; un `gs.Out` ne se lit pas dans le kernel. `return` sans valeur termine le thread.
- Indice hors limites : écriture non faite, `gs.OutOfBounds` à la lecture du résultat.

### 4.6 `gs.Array` : synchrone, asynchrone, chaînable

Un appel de kernel **met le travail en file et rend la main immédiatement**. Il renvoie un `gs.Array` : un handle vers des données sur le GPU, avec un événement de fin en attente.

```python
y = normalize(x, 0.0, 1.0)         # rend la main tout de suite
print(y[:5])                        # lecture → attend la fin et rapatrie
np.asarray(y); y.numpy()            # idem
y = await normalize(x, 0.0, 1.0)    # attend sans bloquer la boucle asyncio
z = other(normalize(x, 0.0, 1.0))   # chaînage : rien ne repasse par la RAM
```

| Opération sur `gs.Array` | Effet |
|---|---|
| passer à un autre kernel | aucune attente, dépendance enregistrée |
| `await y` | attente et lecture sur un thread de l'exécuteur asyncio, puis rend `y` (M4) |
| `np.asarray(y)`, `y.numpy()`, `y[...]`, `print(y)` | attente bloquante GIL libéré, puis relecture. Sur mémoire unifiée : **vue** sur la mémoire du buffer, sans copie |
| `y.shape`, `y.dtype`, `y.device` | immédiat (connu sans attendre) |
| `y.sum()`, `y.max(axis=…)`, `y.mean()` | réductions, calculées sur l'hôte par NumPy après lecture (kernels internes plus tard) |
| `y.save(path)` | fichier `.npy` écrit bloc par bloc depuis le GPU |
| `torch.from_dlpack(y)`, `np.from_dlpack(y)` | export DLPack de la mémoire hôte, sans copie |

Le même code fonctionne à l'identique dans un script, dans Jupyter et dans du code async.

**Règle d'utilisation :** un tableau NumPy passé à un kernel ne doit pas être modifié tant que le résultat n'a pas été lu (ou attendu). Sur mémoire unifiée, le GPU lit directement cette mémoire : la modifier pendant le calcul change le résultat. CUDA impose la même règle. `gs.debug()` la vérifie (somme de contrôle de l'entrée avant et après le kernel).

### 4.7 Out-of-core

```python
big = np.memmap("signal_50GB.f32", dtype=np.float32, mode="r")
out = normalize(big, 0.0, 1.0)              # même appel, découpage automatique
out.save("normalise.f32")

for bloc in normalize.stream(generateur(), mean=0.0, std=1.0):
    traiter(bloc)                           # np.ndarray par bloc
```

- Tableau ou `memmap` : même appel. Le résultat est un `gs.Array` « out-of-core » (matérialisé en RAM ou sur disque à la lecture).
- Générateur : `.stream()` renvoie un itérateur de résultats, pour ne jamais concaténer en RAM.
- Détails du découpage : §8.

### 4.8 Périphériques

```python
gs.devices()
# [Device('Apple M5 Pro', backend='metal', f64=False, unified_memory=True),
#  Device('Apple M5 Pro', backend='opencl', f64=False, unified_memory=True, memory='17.8 Go')]
# Sur le PC : Device('NVIDIA GeForce RTX 4080 SUPER', backend='vulkan', f64=True, f64_math=False, …)

with gs.device("opencl"):            # pour un bloc de code
    ...
normalize(x, 0.0, 1.0, device="opencl:0")   # pour un appel
```

#### Choix automatique (M1.5)

Sans `device=` ni `with gs.device(...)`, GPUsync choisit le périphérique **pour chaque kernel**, selon ce qu'il
exige (§6.2) et ce que le matériel sait faire. L'utilisateur n'a jamais à choisir entre Vulkan et OpenCL.

1. **Filtrer** les périphériques capables : `f64`, `f64_math`, et plus tard atomiques, mémoire locale…
2. **Classer** les restants :

| Rang | Catégorie | Backend préféré | Pourquoi |
|---|---|---|---|
| 1 | GPU dédié | CUDA (M6) → Vulkan → OpenCL → DX12 | le plus puissant ; Vulkan a l'overhead le plus bas (0,095 ms contre 0,27 ms en OpenCL sur la RTX) |
| 2 | GPU intégré | Metal (Mac) ; sinon OpenCL → Vulkan | zéro-copie : Metal `NoCopy`, OpenCL `USE_HOST_PTR` |
| 3 | CPU | OpenCL (POCL) → `cpu:0` | dernier recours |

3. Si seul `cpu:0` convient (ex. f64 avec `exp` sur un Mac), le calcul s'y fait, avec un avertissement
   `gs.PerformanceWarning` émis une fois par kernel : le résultat est correct, mais lent.

#### Opération indisponible : option `compat`

```python
@gs.kernel                     # compat="auto" (défaut)
@gs.kernel(compat="strict")
```

Quand le périphérique prévu (choisi par l'utilisateur, ou premier du classement) ne sait pas exécuter une
opération du kernel (ex. `gs.exp` en f64 sur Vulkan) :

| Mode | Comportement |
|---|---|
| `auto` | avertissement `gs.CompatibilityWarning` (une fois par kernel, il nomme l'opération et les deux périphériques), puis **tout le kernel** s'exécute en **OpenCL**, de préférence sur le même GPU physique (ex. `vulkan:0` → `opencl:0` sur la RTX). Si aucun périphérique OpenCL ne convient : `cpu:0`, avec le même avertissement |
| `strict` | `gs.UnsupportedFeature`, sans changer de périphérique ; le message liste les périphériques compatibles |

Règles :
- **Pas de simulation** : aucune opération n'est émulée en logiciel (ex. f64 en deux f32 sur Metal), ni
  découpée entre deux backends au milieu d'un kernel. C'est le kernel entier qui change de périphérique.
- Les données ne bougent qu'une fois : une entrée déjà sur le périphérique prévu (`gs.Array`) est transférée
  vers le périphérique de repli, puis le résultat y reste.
- `y.device` indique toujours où le calcul a réellement eu lieu.

Le choix est mis en cache par kernel et visible : `y.device`, `gs.default_device(kernel)`. Exemples sur le PC :

| Kernel | Choisi |
|---|---|
| f32 | `vulkan:0` (RTX) |
| f64 arithmétique | `vulkan:0` (RTX, `SHADER_F64`) |
| f64 avec `gs.exp` | `opencl:0` (RTX) |

Avec `device=` explicite, le périphérique demandé est utilisé tel quel ; s'il ne convient pas,
`gs.UnsupportedFeature` liste les périphériques compatibles.

Le backend « CPU » est l'interpréteur de l'IR (lent, toujours disponible) ; POCL via OpenCL est l'option CPU rapide.

### 4.9 Débogage et introspection

```python
pairwise_dist.source("opencl")     # code généré pour une cible
pairwise_dist.source("wgsl")
pairwise_dist.ir()                 # IR lisible

with gs.debug():                   # exécution par l'interpréteur de l'IR
    pairwise_dist(series[:10])     # exceptions Python, index hors bornes avec la valeur de (i, j)
```

### 4.10 Erreurs

| Exception | Quand | Exemple de message |
|---|---|---|
| `gs.KernelSyntaxError` | construction non supportée dans le kernel | `ligne 4 : les listes Python ne sont pas supportées dans un kernel ; utilisez gs.local(gs.f32, 16)` |
| `gs.TypeMismatch` | dtype d'un argument ≠ signature | `argument 'x' attendu f32[:], reçu float64[:]. Utilisez x.astype(np.float32) ou @gs.kernel(cast="auto")` |
| `gs.ShapeMismatch` | dimensions nommées incohérentes | `dimension 'n' incohérente : a → 10, b → 12` |
| `gs.UnsupportedFeature` | capacité absente du périphérique | `le kernel 'precise' utilise f64, non supporté par 'Apple M5 Pro' (metal)` |
| `gs.DeviceError` | erreur du driver | message du driver + code généré + périphérique |
| `gs.OutOfMemory` | même un bloc minimal ne tient pas | taille demandée, budget disponible, entrées résidentes en cause |

**Règle :** aucune conversion implicite de dtype par défaut. `cast="auto"` est explicite et documenté.

---

## 5. Frontend : du Python à l'IR

### 5.1 Récupérer l'AST

Au moment de la décoration :
1. `inspect.getsource` + `ast.parse` côté Python (comme le prototype).
2. `gpusync-py` parcourt les nœuds `ast` via PyO3 et construit un **`KernelAst`** : une structure Rust qui ne couvre que le sous-ensemble supporté.

Pourquoi ne pas parser en Rust : l'`ast` de l'interpréteur courant correspond toujours à sa grammaire, et on évite de maintenir un parseur Python.

Résolution des noms à la décoration :
- `gs.*`, `math.*` → fonctions intrinsèques.
- Constantes `int` / `float` / `bool` des globales ou de la closure → inlinées (remplace `constants=` du prototype).
- Fonctions décorées `@gs.func` → fonctions device, inlinées ou émises séparément.
- Tout autre nom → `gs.KernelSyntaxError` avec la ligne.

Annotations : évaluées dans les globales de la fonction, y compris avec `from __future__ import annotations`.

### 5.2 Sous-ensemble supporté (MVP)

| Supporté | Non supporté |
|---|---|
| arithmétique, comparaisons, `and`/`or`/`not`, bit à bit (`& \| ^ ~ << >>`), opérateur ternaire | listes, dicts, sets, chaînes |
| variables locales scalaires | récursion |
| `if` / `elif` / `else`, `for … in range(…)`, `while`, `break`, `continue` | `try`, `with`, générateurs, `lambda` |
| indexation `a[i]`, `a[i, j]`, `a.shape[k]` | appels de fonctions Python ordinaires |
| tuples pour `gs.index()` et les retours multiples | slicing dans le kernel |
| `gs.local(type, taille)` : tableau local de taille constante | allocation dynamique |
| intrinsèques : `gs.sqrt`, `gs.exp`, `gs.min`, `gs.abs`, `gs.mulhi`, … ; `math.*` équivalents | |

### 5.3 Typage

- Les **paramètres** sont typés par la signature. Pas de signature → `gs.KernelSyntaxError`.
- Les **locales** sont inférées à la première affectation ; une réaffectation incompatible est une erreur.
- **Les littéraux s'adaptent au contexte** : `0.0` dans un contexte f32 est un f32. Indispensable car en OpenCL C, `3.0` est un double, ce qui force du fp64 ou échoue sur les GPU sans fp64.
- Promotions : `i32 op f32 → f32`, jamais vers f64 implicitement.
- Mode élément par élément : le frontend réécrit `x` en `x[idx]` pour chaque tableau de la forme de la sortie.

**Opérations bit à bit et `gs.mulhi`.** `&`, `|`, `^`, `~` suivent le complément à deux (comme Python pour
les valeurs représentables) ; sur deux booléens, `&`, `|`, `^` donnent un booléen. `<<` et `>>` gardent le
type de la valeur décalée (le compte y est converti) ; le compte est lu comme non signé et un compte ≥ la
largeur donne 0 (ou −1 pour `>>` d'un négatif), là où C et les GPU le réduiraient modulo la largeur. Un
compte constant s'écrit directement, sinon une petite fonction teste la largeur ; `<<` d'un signé passe
par le non signé (débordement indéfini en C, erreur de l'évaluation constante en WGSL).
`gs.mulhi(a, b)` rend la partie haute du produit complet (2 × 32 ou 2 × 64 bits) : `mul_hi` en OpenCL,
`__umulhi` / `__mulhi` / `__umul64hi` / `__mul64hi` en CUDA, produits partiels sur des moitiés en WGSL.
Avec elle, une multiplication modulaire (Montgomery) tient en arithmétique 32 bits, sans `%` 64 bits,
émulé et lent sur GPU (un étage de NTT mesuré va ×1.5 à ×7 plus vite).

### 5.4 Spécialisation

Un `@gs.kernel` est un **modèle**. Une version compilée est identifiée par :

```
(hash de l'IR, dtypes, rangs, cible, capacités du périphérique)
```

Les tailles des dimensions ne font en général **pas** partie de la clé : elles sont passées en paramètres,
pour ne pas recompiler à chaque taille. **Exception (21/11/2026)** : dans un kernel avec index, une dimension
qui n'est pas un axe de la grille (ex. la longueur `t` des séries) et vaut au plus 1024 devient une constante
du code généré. Le compilateur peut alors dérouler les boucles et garder les tableaux locaux en registres ;
WGSL étant réécrit par naga (boucles que Metal et Vulkan ne déroulent plus), le générateur déroule lui-même
les boucles les plus intérieures à bornes constantes (≤ 64 tours), en propageant les locales constantes
(`t = len(ref)`). Les axes de la grille restent des paramètres (changer `n` ne recompile rien) ; au plus 8
variantes spécialisées par kernel et par périphérique, au-delà la version générique sert.
`GPUSYNC_SPECIALIZE=0` la désactive, `GPUSYNC_DUMP_DIR` écrit chaque code compilé (débogage).

Tableaux non contigus (vues NumPy) : copiés en contigu à l'upload dans le MVP ; strides en paramètres plus tard si le besoin est mesuré.

---

## 6. IR

### 6.1 Forme

IR **structurée** : un arbre d'instructions typées (`Let`, `Assign`, `If`, `For`, `While`, `Store`, `Return`, `AtomicRmw`, `Barrier`) et d'expressions typées. Pas de SSA ni de graphe de contrôle : toutes les cibles sont des langages structurés de la famille du C, donc l'IR garde leur forme et les émetteurs restent simples.

Types : `bool`, `i32`, `u32`, `i64`, `f16`, `f32`, `f64`, tableaux `T[dims]` (globaux, en lecture ou en écriture), tableaux locaux de taille fixe.

### 6.2 Capacités requises

Une passe calcule ce que le kernel exige :

```rust
struct Requirements { f64: bool, f64_math: bool, i64: bool, f16: bool, atomics_f32: bool, local_mem_bytes: u32, … }
```

- `f64` : le kernel manipule des f64 (arithmétique, comparaisons, `//`, `%`, `sqrt`, `floor`, `ceil`, `abs`,
  `min`, `max`, `fma`, conversions).
- `f64_math` : il appelle en plus une fonction transcendante en f64 (`exp`, `exp2`, `log`, `log2`, `sin`,
  `cos`, `tan`, `tanh`, `atan2`, `pow`, `**`).

| Périphérique | `f64` | `f64_math` |
|---|---|---|
| OpenCL avec `cl_khr_fp64`, CUDA, `cpu:0` | ✅ | ✅ |
| Vulkan avec `SHADER_F64` | ✅ | ❌ (résultat faux silencieux ou plantage du pilote, mesuré) |
| Metal, DX12, GPU sans f64 | ❌ | ❌ |

Comparées aux capacités du périphérique **avant** la génération de code → `gs.UnsupportedFeature` clair au lieu d'une erreur de driver ou d'un résultat faux. Elles servent aussi au choix automatique du périphérique (§4.8).

### 6.3 Interpréteur de référence

`gpusync-ir` contient un interpréteur de l'IR (séquentiel, en Rust). Il sert :
- d'**oracle de tests** pour tous les backends ;
- de mode `gs.debug()` (vérification des bornes, valeur de l'index fautif) ;
- de backend « CPU » de secours toujours disponible.

---

## 7. Génération de code

Un trait commun, une implémentation par cible :

```rust
trait Emitter {
    fn emit(&self, kernel: &ir::Kernel, layout: &LaunchLayout) -> Result<String>;
}
```

| Cible | Version de base | Points d'attention |
|---|---|---|
| **OpenCL C** | 1.2 | suffixe `f` sur les littéraux f32 ; `#pragma OPENCL EXTENSION cl_khr_fp64` seulement si requis ; pas d'atomiques flottants en 1.2 → émulation par boucle CAS ; garde de bornes |
| **WGSL** | wgpu | f64 seulement sur Vulkan avec `SHADER_F64` (extension de naga : type `f64`, littéraux `1.5lf`, paramètres f64 dans un buffer `array<f64>` séparé), sans fonctions transcendantes ; pas de i64 ; taille max d'un buffer et d'une liaison limitée → tailles de bloc (§9.2) |
| **CUDA C** | NVRTC (M6) | littéraux `f`, `__restrict__`, garde de bornes |

Règles communes :
- La grille est arrondie au multiple de la taille de groupe ; chaque kernel commence par une **garde de bornes** (`if (i >= n) return;`).
- La taille de groupe est choisie par le backend selon les limites du périphérique (comme l'auto-tuning du prototype), et mise en cache.
- Le code généré est déterministe (même IR → même texte), ce qui rend le cache et les tests de référence fiables.

---

## 8. Runtime

### 8.1 Concepts

| Concept | Rôle |
|---|---|
| `Device` | un périphérique d'un backend + ses capacités + son budget mémoire |
| `Stream` | une file **ordonnée** par périphérique (MVP : une seule) |
| `Buffer` | mémoire sur le périphérique, issue du pool |
| `Event` | fin d'une opération ; attendu directement, jamais par callback |
| `Op` | upload, kernel, relecture, copie : enregistré dans le stream |

### 8.2 Trait `Backend`

```rust
trait Backend {
    fn devices(&self) -> Vec<DeviceInfo>;
    fn compile(&self, dev: &Device, src: &KernelSource) -> Result<CompiledKernel>;
    fn alloc(&self, dev: &Device, bytes: usize) -> Result<RawBuffer>;
    fn upload(&self, s: &Stream, dst: &RawBuffer, src: &[u8]) -> Result<Event>;
    fn launch(&self, s: &Stream, k: &CompiledKernel, args: &[Arg], grid: Grid) -> Result<Event>;
    fn download(&self, s: &Stream, src: &RawBuffer, dst: &mut [u8]) -> Result<Event>;
    fn wait(&self, e: &Event) -> Result<()>;

    // Mémoire unifiée — optionnel : `None` = non supporté, le runtime revient à la copie
    fn wrap_host(&self, dev: &Device, ptr: *mut u8, len: usize) -> Option<RawBuffer>; // hôte → GPU sans copie
    fn host_view(&self, buf: &RawBuffer) -> Option<HostSlice>;                         // GPU → hôte sans copie
}
```

`DeviceInfo` expose `unified_memory: bool` (en OpenCL : `CL_DEVICE_HOST_UNIFIED_MEMORY`), ainsi que l'alignement requis pour le zéro-copie.

Bibliothèques Rust envisagées (versions et fonctionnalités à vérifier au démarrage du projet) :
- wgpu : `wgpu`.
- OpenCL : `opencl3` / `cl3`, avec **chargement dynamique** de la bibliothèque OpenCL : si elle est absente, le backend est juste indisponible.
- CUDA : liaisons maison, chargement dynamique de `libcuda` / `nvcuda` et de NVRTC (M6 : `cudarc` écarté).

### 8.3 Exécution d'un appel

```
pairwise_dist(series)
  │ Python : wrapper léger
  ├─ 1. clé de spécialisation (dtypes, rangs, device) → cache ; sinon frontend + codegen + compile
  ├─ 2. lier les dimensions (n=2000, t=365), vérifier les formes
  ├─ 3. GIL libéré ─────────────────────────────── Rust
  │     ├─ si `series` est un np.ndarray : zéro-copie si possible (§8.4), sinon upload
  │     │  (rien à faire si c'est déjà un gs.Array)
  │     ├─ alloue la sortie depuis le pool (en mémoire partagée si unifiée)
  │     └─ enregistre le kernel dans le stream, après l'upload
  └─ 4. renvoie gs.Array(buffer, événement de fin)        ← aucune attente ici

np.asarray(dist)
  └─ GIL libéré : enregistre la relecture, attend son événement, renvoie le tableau NumPy
```

Un appel synchrone complet ne fait donc **qu'une seule attente**, au moment de la relecture.

### 8.4 Mémoire

- **Pool** par périphérique, par classes de taille (puissances de 2 au-delà d'un seuil), avec un budget (par défaut une fraction de la mémoire du périphérique).
- Un buffer libéré par Python (`gs.Array` détruit) retourne au pool **seulement après** l'événement de sa dernière opération.

#### Mémoire séparée (GPU dédié)

- Entrée `np.ndarray` : upload vers un buffer du pool.
- Sortie : relue dans un nouveau `np.ndarray` (ou dans `out=` si fourni).

#### Mémoire unifiée (Apple Silicon, GPU intégrés Intel / AMD, Jetson)

Le GPU accède directement à la RAM de l'hôte : copier reviendrait à dupliquer les données dans la même mémoire physique. Le runtime utilise donc le zéro-copie quand le backend le permet.

| Sens | Stratégie | Repli |
|---|---|---|
| Entrée `np.ndarray` | `wrap_host` : le buffer GPU **utilise la mémoire du tableau** | copie si mémoire non alignée, non contiguë, ou backend sans zéro-copie |
| Sortie | buffer alloué **en mémoire partagée** ; `np.asarray(y)` renvoie une **vue** sur cette mémoire (`host_view`) | relecture par copie |

Mesures sur Apple M5 Pro, OpenCL 1.2, 256 Mo :

| Méthode | Upload + kernel | Tableau NumPy modifié sans copie | Mémoire |
|---|---|---|---|
| `CL_MEM_COPY_HOST_PTR` | 33,3 ms | non | ×2 |
| `CL_MEM_USE_HOST_PTR` | 7,4 ms | oui (lecture du buffer = même adresse que le tableau) | ×1 |

Les gros tableaux NumPy (`np.ones`, `np.empty`) étaient déjà alignés sur une page (16 Ko) : NumPy les obtient directement du système. Les petits tableaux ne le sont pas toujours ; leur copie est peu coûteuse. Pour garantir le zéro-copie, `gs.empty(...)` / `gs.zeros(...)` renvoient des tableaux NumPy alloués alignés.

Durée de vie : une vue NumPy sur une sortie **garde le buffer en vie** (objet `base` du tableau). Le buffer ne retourne au pool qu'après destruction de la vue.

Support par backend :

| Backend | Zéro-copie | Mécanisme |
|---|---|---|
| OpenCL | ✅ mesuré | `CL_MEM_USE_HOST_PTR` en entrée, `CL_MEM_ALLOC_HOST_PTR` + map en sortie |
| Metal natif | ✅ | `newBufferWithBytesNoCopy` (mémoire alignée sur une page), `storageModeShared` |
| wgpu → Metal | ✅ faisable | `newBufferWithBytesNoCopy`, confié à wgpu par `wgpu-hal` (`metal::Device::buffer_from_raw`, vérifié dans wgpu 30) |
| wgpu → Vulkan / DX12 | ❌ | wgpu-hal 30 n'importe pas de mémoire hôte comme buffer (seulement `texture_from_raw`). Le matériel le permet pourtant (`VK_EXT_external_memory_host` présent sur la RTX 4080 SUPER et l'UHD 770, alignement 4 Kio) : il faudrait un backend Vulkan natif |
| CUDA (Jetson, Grace) | ✅ | `cudaHostRegister` ou mémoire managée |

Mesuré à M1 : wgpu copie et consomme 3 à 5 fois la taille des données. La réponse est le moteur de transferts (§9) : zéro-copie là où l'import est possible, transferts par blocs bornés ailleurs.

### 8.5 Threads et GIL

- Toute opération longue (compilation, attente, copie) se fait **GIL libéré**.
- `__await__` : attente sur un thread de l'exécuteur asyncio (`run_in_executor`), GIL libéré pendant l'attente
  et la lecture (M4).
- Aucune boucle asyncio interne à la lib.

### 8.6 Cache de compilation

- En mémoire, par processus.
- Sur disque : écarté à M4 pour OpenCL et wgpu après mesure (les drivers mettent déjà les binaires en cache ;
  compilation de 0,1 à 2 ms). CUDA (M6) : cache du PTX produit par NVRTC (152 → 62 ms au premier appel).

---

## 9. Moteur de transferts (out-of-core intégré)

**Principe : un seul chemin d'exécution.** Tout appel est planifié en blocs ; un tableau qui tient en mémoire
n'est qu'un plan à un bloc. Il n'y a **aucune limite de taille**, seulement des budgets mémoire et des
contraintes de taille de bloc. L'out-of-core n'est pas un mode spécial : c'est la gestion normale des transferts.

### 9.1 Trois modes, choisis automatiquement à chaque appel

| Mode | Quand | Transferts | Mémoire en plus |
|---|---|---|---|
| **Direct** (zéro-copie) | GPU à mémoire unifiée, mémoire importable (OpenCL `USE_HOST_PTR`, Metal `NoCopy`) | aucun : le GPU lit et écrit les tableaux NumPy | 0 |
| **Résident** | GPU dédié, l'appel tient dans le budget VRAM | un transfert DMA par entrée ; le résultat reste en VRAM (chaînage gratuit) | anneau de transfert (RAM, borné) |
| **Flux** | GPU dédié au-delà du budget VRAM ; ou source non importable (memmap refusé, générateur) ; ou wgpu sans import | un transfert par bloc, recouvert par le calcul | anneau de transfert (RAM, borné) + quelques blocs (VRAM) |

Les trois modes exécutent le même code généré : un bloc est vu par le kernel comme un tableau entier.

### 9.2 Blocs

- **Taille d'un bloc** = min(cible par défaut 64 Mio par tableau ; taille max d'un buffer ou d'une liaison du
  périphérique ; budget ÷ (nombre de tampons × nombre de tableaux)), arrondie aux alignements (page de
  16 Kio sur Apple Silicon, 4 Kio ailleurs ; 256 o pour les décalages de liaison wgpu).
- **Les limites matérielles deviennent des tailles de bloc** : 1 Gio par liaison sur l'UHD 770 en Vulkan,
  3,3 Gio par allocation sur le M5 Pro en OpenCL. Aucune n'est plus une erreur.
- **Élément par élément** : un bloc est une sous-plage d'indices de toutes les entrées et de la sortie.
- **Avec index** : blocs de lignes de la sortie (son premier axe) ; les entrées lues seulement à la ligne
  courante suivent ce découpage, les autres restent entières (§9.7).

### 9.3 Transferts sur GPU dédié

- **Anneau de transfert** : K tampons (3 par défaut) de la taille d'un bloc, en mémoire épinglée (OpenCL
  `CL_MEM_ALLOC_HOST_PTR` mappé ; wgpu : buffers `MAP_WRITE` / `MAP_READ` réutilisés). Remplis par `memcpy`
  depuis la source sur un thread Rust (GIL libéré) ; le DMA se fait à pleine vitesse depuis la mémoire épinglée.
- **Pipeline** : pendant que le bloc *k* est calculé, le bloc *k+1* est envoyé et le bloc *k−1* revient
  **directement dans sa destination finale** (tableau NumPy, memmap). OpenCL : une file de transferts et une
  file de calcul synchronisées par événements, donc un vrai recouvrement. wgpu : une seule file ; le
  recouvrement se fait entre les `memcpy` CPU et le travail du GPU.
- **Budget VRAM** : 50 % de la mémoire du périphérique par défaut (OpenCL : `CL_DEVICE_GLOBAL_MEM_SIZE` ;
  Vulkan : tas mémoire lus via wgpu-hal), réglable par `gs.set_memory_budget()`.
- **Pool** : tampons et buffers de blocs réutilisés d'un appel à l'autre, aucune allocation par appel.
- **Coût en RAM borné** : de l'ordre de 3 tampons × 64 Mio × (entrées + sortie), soit 0,4 à 0,8 Gio quel que
  soit le volume (contre 3 à 5 fois le volume mesuré à M1).
- **Le dernier `memcpy`** (source → tampon épinglé) disparaît avec CUDA (M6 : `cudaHostRegister`, DMA direct
  depuis le tableau NumPy) ou un backend Vulkan natif (`VK_EXT_external_memory_host`).

### 9.4 Mémoire unifiée

- Mode Direct **par bloc** : chaque bloc est une sous-plage du tableau NumPy importée sans copie
  (`USE_HOST_PTR` en OpenCL, `newBufferWithBytesNoCopy` en Metal via wgpu-hal). Aucune copie, quel que soit
  le volume ; les blocs servent seulement à respecter la taille max d'une allocation.
- Vulkan via wgpu sur GPU intégré (UHD 770) : pas d'import en wgpu 30, donc mode Flux avec anneau borné.
  OpenCL reste le chemin zéro-copie de l'UHD 770.
- memmap (fichier) : import tenté ; s'il est refusé (pages d'un fichier), mode Flux.

### 9.5 Sources et destinations, sans limite

| Source | Appel | Résultat |
|---|---|---|
| `np.ndarray` en RAM | `f(x)` | `gs.Array` |
| `np.memmap` plus gros que la RAM | `f(x, out=np.memmap(...))` | écrit bloc par bloc dans le fichier de sortie |
| générateur / itérable de blocs | `f.stream(gen)` | itérateur de résultats, un par bloc |

- `out=` : tout tableau contigu de la bonne forme et du bon type (évite aussi d'allouer la sortie).
- Sans `out=`, la sortie est allouée en RAM : si elle ne peut pas l'être, `gs.OutOfMemory` suggère `out=`.

### 9.6 Chaînage

- Mode Résident : le `gs.Array` reste en VRAM, le kernel suivant le lit sans transfert.
- Mode Flux : le résultat est en RAM (ou dans `out=`), le kernel suivant le relit. La fusion de kernels (M7)
  supprimera ces allers-retours.
- Mode Direct : pas de transfert dans aucun cas.

### 9.7 Ce qui est découpable (kernels avec index, M3)

On découpe la grille le long de sa **première dimension** (blocs de lignes de la sortie). Pour chaque entrée :
- si elle est lue **seulement à la ligne courante** — premier indice égal à la variable `i` de
  `i, … = gs.index()` (jamais réaffectée), et même première dimension nommée que la sortie — elle est
  découpée de la même façon (`x[i, k]`, `x[i, best]`, `idx[i]`) ;
- sinon, elle est **entière** : envoyée une fois et gardée sur le GPU pendant tout le traitement
  (généralisation des « arguments constants » du prototype). C'est une analyse du source, faite à la
  compilation : `x[i + 1, k]` ou `x[j, k]` gardent `x` entier, sans jamais lire hors du bloc.

| Kernel | Découpé | Entier |
|---|---|---|
| `normalize(x: ["n"]) -> ["n"]` (élément par élément) | `x`, sortie | — |
| `pairwise_dist(series: ["n","t"]) -> ["n","n"]` | sortie | `series` (lue aux lignes `i` et `j`) |
| `matvec(a: ["m","k"], v: ["k"]) -> ["m"]` | `a`, sortie | `v` |
| `stats(x: ["n","t"]) -> (["n"], ["n"])` | `x`, sorties | — |

Taille d'un bloc : 64 Mio par tableau découpé au plus, dans le budget restant après les entrées entières
(deux emplacements en OpenCL) et dans la taille max d'un buffer. Si les entrées entières ne tiennent pas dans
le budget (ou dans un buffer) : `gs.UnsupportedFeature` qui nomme l'entrée en cause.

Dans un bloc, les lancements sont cadencés (§ mesures M3) : aucun ne dure plus de ~0,1 s, quelle que soit
la taille de la grille.

### 9.8 Critères de réussite

| Cas | Machine | Pourquoi |
|---|---|---|
| 4 Gio en OpenCL | Mac | dépasse l'allocation max (3,3 Gio) |
| 2 Gio en Vulkan | PC, UHD 770 | dépasse la liaison max (1 Gio) |
| 24 Gio en Vulkan et en OpenCL | PC, RTX | dépasse la VRAM (16 Gio) |
| 40 Gio en memmap, `out=` memmap | Mac | dépasse la RAM (26 Go) |

- Mémoire en plus : 0 en Direct ; au plus la taille de l'anneau (~0,5 Gio) en Résident et en Flux,
  **indépendante du volume**.
- Débit en mode Flux sur la RTX : proche de la bande passante PCIe, transferts et calcul recouverts.
- Résultats identiques à NumPy dans tous les cas (tests différentiels avec des volumes qui forcent le découpage).

---

## 10. Packaging et interop

- **PyO3 en abi3** + **maturin** : une wheel par plateforme, valable pour toutes les versions de Python supportées.
- Wheels : macOS arm64 et x86_64, Linux manylinux x86_64 et aarch64, Windows x86_64.
- **Aucune dépendance Python obligatoire autre que NumPy.**
- wgpu est compilé dans la wheel ; OpenCL et CUDA sont chargés dynamiquement s'ils sont présents.
- Interop : `__array__`, protocole DLPack (export et import) ; `__cuda_array_interface__` avec le backend CUDA.

---

## 11. Tests et qualité

| Niveau | Contenu |
|---|---|
| Unitaires Rust | frontend (KernelAst → IR), passes de l'IR, interpréteur |
| Code généré | tests de référence (« snapshots ») du texte émis pour chaque cible |
| Différentiel | chaque kernel de test exécuté sur l'interpréteur, sur chaque backend disponible et comparé à NumPy ; génération aléatoire d'entrées (tests par propriétés) |
| Bout en bout Python | API, erreurs (messages compris), Jupyter, async |
| Mémoire unifiée | le zéro-copie est réellement utilisé (même adresse, pic mémoire ×1) ; replis corrects (tableau non aligné, non contigu, backend sans zéro-copie) ; durée de vie des vues |
| Performance | overhead par appel, débit sur gros volumes, out-of-core, pic mémoire ; comparaison au prototype et à pyopencl brut |

**CI sans GPU** : interpréteur + POCL (OpenCL sur CPU) + Vulkan logiciel (lavapipe) pour wgpu. Les GPU réels (Mac, NVIDIA) sont testés manuellement ou sur des runners dédiés.

---

## 12. Feuille de route

| Jalon | Contenu | Critère de fin |
|---|---|---|
| **M0** Squelette | workspace Cargo, maturin, `gs.devices()` liste wgpu + OpenCL | `pip install -e .` puis `gs.devices()` sur le Mac |
| **M1** MVP élément par élément | frontend minimal, IR, interpréteur, émetteurs WGSL + OpenCL C, `gs.Array` synchrone, zéro-copie OpenCL sur mémoire unifiée | `normalize` correct sur Metal et OpenCL ; overhead ≤ 0,3 ms ; tests différentiels verts ; pic mémoire ×1 sur 1 Go en OpenCL sur le Mac |
| **M1.5** Moteur de transferts et choix automatique | blocs et plan d'exécution ; modes Direct, Résident, Flux ; Metal `NoCopy` via wgpu-hal ; anneau épinglé et pipeline à deux files (OpenCL) ; `out=`, `np.memmap`, `.stream()` ; budget VRAM ; f64 via Vulkan (`SHADER_F64`) ; besoins `f64` / `f64_math` ; choix automatique du périphérique par kernel et option `compat` (`auto` : avertissement + repli OpenCL ; `strict` : erreur) (§4.8) | critères du §9.8 sur le Mac et le PC ; un kernel f64 arithmétique tourne sur Vulkan (RTX), un kernel f64 avec `exp` sur OpenCL (RTX), sans `device=` |
| **M2** Index et formes | `gs.index()`, dimensions nommées, retours multiples, boucles, `gs.local`, réductions de `gs.Array` | `pairwise_dist` et `stats` corrects ; ta DTW portée et comparée au prototype |
| **M3** Out-of-core avec index | ✅ découpage des kernels avec index (§9.7) : sortie découpée, entrées résidentes ; lancements cadencés | `pairwise_dist` sur des données dont la sortie dépasse la VRAM (19,3 Gio sur la RTX, 3,8 s) |
| **M4** Async et interop | ✅ `await`, DLPack, `y.save()`, appels multi-threads ; cache disque écarté après mesure | même script en synchrone, async et Jupyter (testé, nbclient) |
| **M5** Mode explicite | ✅ `grid=`, `gs.Out`, atomiques (dont émulation OpenCL 1.2 et WGSL), bornes vérifiées | histogramme correct sur toutes les cibles (6 périphériques) |
| **M6** CUDA | ✅ émetteur CUDA C, backend CUDA (liaisons maison, NVRTC), moteur commun avec OpenCL, cache PTX | parité des tests avec les autres backends sur NVIDIA (61 tests différentiels sur `cuda:0`) |
| **M7** Avancé | multi-GPU, plusieurs streams, fusion de kernels élément par élément | à définir avec des mesures |

---

## 13. Décisions ouvertes

| # | Question | Piste actuelle |
|---|---|---|
| D1 | IR maison ou bâtir sur CubeCL ? | IR maison, car OpenCL C est une cible obligatoire. Vérifier les cibles actuelles de CubeCL avant de trancher |
| D2 | Version minimale de Python | ✅ 3.10 (abi3), décidé à M0 |
| D3 | `f16` dans le MVP ? | non ; pas de besoin constaté à M2, à revoir avec un cas réel |
| D4 | Strides (vues non contiguës) | copie contiguë dans le MVP ; à mesurer |
| D5 | Nom du projet : garder « GPUsync » ? | à décider |
| D6 | Licence | MIT, comme le prototype |
| D7 | Syntaxe du mode explicite (`grid=` + `gs.Out`) | ✅ validée à M5 sur 5 cas ; ajouts : valeur de départ, accumulation avec `out=` ; `unsafe_writes` abandonné |
| D8 | API des réductions : méthodes de `gs.Array` seulement, ou aussi des kernels de réduction écrits par l'utilisateur ? | méthodes d'abord |
| D9 | Backend natif (Metal, Vulkan) en plus de wgpu pour le zéro-copie ? | Metal : inutile, wgpu-hal permet `NoCopy` (M1.5). Vulkan : nécessaire pour importer la mémoire hôte (absent de wgpu 30) ; à décider après M1.5 selon le coût mesuré de l'anneau de transfert, sachant que CUDA (M6) couvre la RTX |
| D10 | Backend par défaut ? | ✅ remplacé par le choix automatique par kernel (§4.8, M1.5) : Metal sur Mac si `NoCopy` tient le critère « 0 en plus », OpenCL pour les GPU intégrés, Vulkan pour les GPU dédiés, OpenCL quand `f64_math` est requis |
