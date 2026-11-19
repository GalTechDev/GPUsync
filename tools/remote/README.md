# gs-remote — exécution distante sur les machines de test

Outil interne pour développer GPUsync en vérifiant chaque étape sur du vrai matériel :
le **client** (sur la machine de dev) synchronise le dépôt vers des **serveurs** (les machines de test),
y lance une commande, affiche la sortie en direct et renvoie le code de retour.

```
Mac (dev) ── client ──┬──► serveur Mac      Apple M5 Pro (Metal / OpenCL)
                      └──► serveur Windows  NVIDIA (CUDA / Vulkan / OpenCL) + GPU intégré
```

Les deux scripts n'utilisent que la bibliothèque standard de Python (≥ 3.9) : rien à installer.

| Fichier | Rôle |
|---|---|
| `gs_remote_server.py` | serveur, à lancer sur chaque machine de test |
| `gs_remote.py` | client, utilisé depuis la machine de dev |

---

## 1. Installer un serveur

Seul `gs_remote_server.py` est nécessaire sur la machine de test (copie-le par clé USB, git, partage réseau…).

### macOS

```bash
caffeinate -i python3 gs_remote_server.py
```

- `caffeinate -i` empêche la mise en veille pendant que le serveur tourne.
- Au premier lancement, macOS demande d'autoriser les connexions entrantes pour Python : accepte.

### Windows

1. Installe Python depuis python.org, avec la case « Add python.exe to PATH » cochée.
   N'utilise pas l'alias du Microsoft Store.
2. Vérifie que le réseau Wi-Fi ou Ethernet est en profil **Privé** (Paramètres → Réseau et Internet → propriétés du réseau).
3. Lance le serveur :

   ```powershell
   py gs_remote_server.py
   ```

4. Quand le pare-feu Windows demande l'autorisation, coche **Réseaux privés uniquement**.
   Ou bien, dans un PowerShell administrateur :

   ```powershell
   New-NetFirewallRule -DisplayName "gs-remote" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
   ```

Pour les étapes suivantes du projet, la machine Windows aura aussi besoin de : `rustup` + Visual Studio Build Tools
(charge de travail C++), le driver NVIDIA (fournit CUDA et OpenCL), le driver Intel du GPU intégré (fournit OpenCL),
et plus tard le CUDA Toolkit. `python gs_remote.py info win` montre ce qui est déjà présent.

### Au démarrage

Le serveur affiche son adresse et la commande à lancer sur le client :

```
gs-remote serveur — PC-WIN (Windows AMD64, Python 3.13.1)
  dossier   : C:\Users\toi\.gpusync-remote-server
  écoute    : 0.0.0.0:8765
  autorisés : IP privées et locales

Sur la machine cliente, ajoute cet hôte avec :
  python gs_remote.py hosts add <nom> http://192.168.1.42:8765 <jeton>
```

Le jeton est créé au premier lancement et conservé dans `~/.gpusync-remote-server/server.json`.
Pour l'afficher : `python gs_remote_server.py --show-token`.

Options utiles :

| Option | Effet |
|---|---|
| `--port 8765` | port d'écoute |
| `--allow 192.168.1.0/24` | n'accepter que ce réseau (répétable). Par défaut : toutes les IP privées et locales |
| `--bind 127.0.0.1` | n'écouter qu'en local (utile si client et serveur sont sur la même machine) |
| `--regen-token` | générer un nouveau jeton (l'ancien est révoqué) |
| `--root DOSSIER` | où stocker workspaces et logs |

---

## 2. Configurer le client

Sur la machine de dev, une fois par serveur :

```bash
python3 tools/remote/gs_remote.py hosts add mac http://127.0.0.1:8765 <jeton-du-mac>
python3 tools/remote/gs_remote.py hosts add win http://192.168.1.42:8765 <jeton-du-pc>
python3 tools/remote/gs_remote.py info all
```

Les hôtes et jetons sont stockés hors du dépôt, dans `~/.gpusync-remote/hosts.json` (droits 600).
Pour utiliser un autre fichier : variable `GS_REMOTE_CONFIG`.

---

## 3. Utilisation

```bash
R="python3 tools/remote/gs_remote.py"

$R info all                                   # OS, Python, Rust, GPU de chaque machine
$R run all -- @python -c "import platform; print(platform.platform())"
$R run win --timeout 1800 -- cargo test       # 30 min max
$R run all --cwd GPUsync-python -- @python -m pytest tests -q
$R run win --env RUST_LOG=debug -- cargo run --example bench
$R run win --shell -- "nvidia-smi && where python"
$R get win bench/results.json                 # récupérer un fichier produit à distance
$R sync all                                   # synchroniser sans exécuter
$R reset win                                  # effacer le workspace distant
```

- **Cible** : un nom, une liste (`mac,win`) ou `all`. Avec plusieurs cibles, les commandes tournent en parallèle,
  chaque ligne est préfixée par `[nom]`, et un récapitulatif affiche le statut, le code et la durée par machine.
- **`--`** sépare les options du client de la commande distante, qui est passée telle quelle.
- **`@python`** est remplacé par l'interpréteur Python du serveur (évite les différences `python` / `py` / `python3`).
- **Code de retour** : celui de la commande distante (ou 1 si une machine a échoué en multi-cibles).
  `Ctrl+C` tue les jobs distants.
- Les sorties de Python distant sont forcées en UTF-8 (`PYTHONUTF8=1`) et non bufferisées.

### Synchronisation

Avant chaque `run` (sauf `--no-sync`), le dossier source est synchronisé vers un **workspace** du même nom :

- dossier source : racine git du dossier courant (ou `--src`) ; workspace : nom du dossier (ou `--ws`) ;
- seuls les fichiers modifiés sont envoyés (comparaison par SHA-256) ; les fichiers supprimés localement sont
  supprimés à distance ;
- **les fichiers créés à distance sont conservés** (`target/`, `.venv/`, résultats) : les compilations Rust
  restent incrémentales d'un run à l'autre ;
- exclusions par défaut : `.git`, `target`, `__pycache__`, `.venv`, `venv`, `node_modules`, `dist`, `build`, … ;
  ajoute les tiennes avec `--exclude MOTIF` ou dans un fichier `.gsremoteignore` à la racine (un motif par ligne).

---

## 4. Sécurité

Ce serveur **exécute n'importe quelle commande** envoyée par un client qui possède le jeton. Protections en place :

| Risque | Protection |
|---|---|
| Quelqu'un sur le réseau envoie des commandes | chaque requête est signée HMAC-SHA256 avec le jeton ; sans le jeton, tout est refusé (401) |
| Le jeton est intercepté | il ne circule jamais : seule une signature est envoyée |
| Une requête capturée est renvoyée plus tard | horodatage (± 5 min) + nonce à usage unique → refusée |
| Une requête est modifiée en route | la signature couvre la méthode, le chemin, l'horodatage et le hash du corps → refusée |
| Accès depuis Internet | seules les IP privées / locales sont acceptées par défaut (`--allow` pour restreindre encore) |
| Un fichier envoyé écrase quelque chose hors du workspace | chemins absolus, `..`, lettres de lecteur et liens symboliques sortants refusés |
| Une commande ne s'arrête jamais | timeout (600 s par défaut) puis arrêt du processus **et de ses sous-processus** |

Limites à connaître :

- le trafic **n'est pas chiffré** : le code envoyé et les sorties sont lisibles sur le réseau local.
  À utiliser sur un réseau de confiance (chez toi), pas sur un Wi-Fi public ;
- les horloges des machines doivent être à l'heure (± 5 min), ce que la synchronisation automatique de Windows
  et macOS assure normalement ;
- le serveur tourne avec les droits de l'utilisateur qui l'a lancé : ne le lance pas en administrateur.
  Arrête-le (`Ctrl+C`) quand tu ne t'en sers pas.

---

## 5. API (pour référence)

Toutes les requêtes portent les en-têtes `X-GS-Time`, `X-GS-Nonce`, `X-GS-Sig`
(`HMAC-SHA256(jeton, "MÉTHODE\nchemin\nhorodatage\nnonce\nsha256(corps)")`).

| Méthode et route | Rôle |
|---|---|
| `GET /info` | informations sur la machine |
| `POST /manifest/<ws>` | `{files: {chemin: [taille, sha256]}}` → `{need, delete}` |
| `POST /sync/<ws>` | zip des fichiers demandés + `__gs_meta__.json` |
| `POST /run/<ws>` | `{cmd, shell, env, timeout, cwd}` → job |
| `GET /jobs/<id>?offset=N` | statut + sortie à partir de l'octet N (base64) |
| `POST /jobs/<id>/kill` | tuer le job |
| `GET /files/<ws>/<chemin>` | télécharger un fichier du workspace |
| `POST /reset/<ws>` | supprimer le workspace |
