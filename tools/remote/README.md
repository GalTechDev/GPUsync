# gs-remote — exécution distante sur les machines de test

Outil interne pour développer GPUsync en vérifiant chaque étape sur du vrai matériel.

```
                 ┌───────────── Mac (dev) ─────────────┐
                 │  CLI ──(localhost)──► HUB :8765      │
                 └──────────────────────▲──────▲───────┘
                     connexion sortante │      │ connexion sortante
                ┌───────────────────────┴┐   ┌─┴──────────────────────┐
                │ agent « mac »          │   │ agent « win »           │
                │ Apple M5 Pro           │   │ NVIDIA + GPU intégré    │
                │ (même Mac, localhost)  │   │ (PC Windows)            │
                └────────────────────────┘   └─────────────────────────┘
```

- Le **hub** tourne sur le Mac. C'est le seul programme qui écoute un port.
- Les **agents** tournent sur les machines de test. **Ils n'écoutent aucun port** : ils se connectent au hub,
  attendent leurs tâches, et se reconnectent tout seuls si le hub redémarre ou si le réseau coupe.
- La **CLI** (sur le Mac) envoie les tâches au hub, affiche la sortie en direct et renvoie le code de retour.

Tout est dans un seul fichier, `gs_remote.py`, qui n'utilise que la bibliothèque standard de Python (≥ 3.9).

---

## 1. Mise en place

### Sur le Mac : le hub

```bash
caffeinate -i python3 tools/remote/gs_remote.py hub
```

À laisser tourner dans un terminal. `caffeinate -i` empêche la mise en veille. Au premier lancement,
macOS demande d'autoriser les connexions entrantes pour Python : accepte (c'est ce qui permet au PC de joindre le hub).

### Sur le Mac : enregistrer les agents

Dans un autre terminal, une fois par machine de test :

```bash
python3 tools/remote/gs_remote.py agents add mac
python3 tools/remote/gs_remote.py agents add win
```

Chaque commande crée un jeton secret propre à l'agent et affiche la commande exacte à lancer sur la machine de test.

### Sur le Mac : l'agent local

Avec la commande affichée par `agents add mac`, en remplaçant l'IP par `127.0.0.1` :

```bash
python3 tools/remote/gs_remote.py agent --hub http://127.0.0.1:8765 --name mac --token <jeton>
```

### Sur le PC Windows : l'agent

1. Installe Python depuis python.org, avec la case « Add python.exe to PATH » cochée
   (pas l'alias du Microsoft Store).
2. Copie `gs_remote.py` sur le PC (clé USB, partage réseau…).
3. Lance la commande affichée par `agents add win` :

   ```powershell
   py gs_remote.py agent --hub http://192.168.1.113:8765 --name win --token <jeton>
   ```

**Aucune règle de pare-feu à créer sur le PC** : l'agent ne fait que des connexions sortantes.

Les fois suivantes, `py gs_remote.py agent` suffit : l'URL du hub, le nom et le jeton sont mémorisés
dans `%USERPROFILE%\.gpusync-remote\agent.json`.

Pour la suite du projet, le PC aura aussi besoin de : `rustup` et Visual Studio Build Tools (charge de travail C++),
le driver NVIDIA (qui fournit CUDA et OpenCL), le driver Intel du GPU intégré (qui fournit OpenCL),
et plus tard le CUDA Toolkit. `gs_remote.py info` montre ce qui est déjà installé.

---

## 2. Utilisation (sur le Mac)

```bash
R="python3 tools/remote/gs_remote.py"

$R info                                       # agents en ligne, OS, Python, Rust, GPU
$R run all -- @python -c "import platform; print(platform.platform())"
$R run win --timeout 1800 -- cargo test       # 30 min max
$R run all --cwd GPUsync-python -- @python -m pytest tests -q
$R run win --env RUST_LOG=debug -- cargo run --example bench
$R run win --shell -- "nvidia-smi && where python"
$R get win bench/results.json                 # récupérer un fichier produit à distance
$R sync all                                   # synchroniser sans exécuter
$R reset win                                  # effacer le workspace distant
$R agents list                                # agents enregistrés
$R agents rm win                              # révoquer un agent (son jeton n'est plus accepté)
```

- **Cible** : un nom, une liste (`mac,win`) ou `all`. Avec plusieurs cibles, les commandes tournent
  en parallèle, chaque ligne est préfixée par `[nom]`, et un récapitulatif affiche statut, code et durée.
  Avec `all`, les agents hors ligne sont signalés et ignorés.
- **`--`** sépare les options de la commande distante, qui est passée telle quelle.
- **`@python`** est remplacé par l'interpréteur Python de l'agent (évite les différences `python` / `py` / `python3`).
- **Code de retour** : celui de la commande distante (1 si une machine a échoué en multi-cibles).
  `Ctrl+C` arrête les tâches distantes.
- Les sorties de Python distant sont forcées en UTF-8 et non bufferisées.

### Synchronisation

Avant chaque `run` (sauf `--no-sync`), le dépôt est synchronisé vers un **workspace** du même nom sur l'agent :

- source : racine git du dossier courant (ou `--src`) ; workspace : nom du dossier (ou `--ws`) ;
- seuls les fichiers modifiés sont transférés (comparaison SHA-256) ; les fichiers supprimés localement
  le sont aussi à distance ;
- **les fichiers créés à distance sont conservés** (`target/`, `.venv/`, résultats) : les compilations Rust
  restent incrémentales d'un run à l'autre ;
- exclusions par défaut : `.git`, `target`, `__pycache__`, `.venv`, `venv`, `node_modules`, `dist`, `build`… ;
  ajoute les tiennes avec `--exclude MOTIF` ou dans un fichier `.gsremoteignore` à la racine (un motif par ligne).

---

## 3. Sécurité

| Risque | Protection |
|---|---|
| Quelqu'un contacte une machine de test pour lui faire exécuter du code | **impossible** : les agents n'écoutent aucun port |
| Un faux hub (IP usurpée sur le réseau) envoie une tâche à un agent | chaque réponse du hub est signée HMAC-SHA256 avec le jeton de l'agent : sans le jeton, l'agent refuse tout |
| Une ancienne tâche, capturée, est renvoyée à un agent | chaque réponse est liée au nonce unique de la requête de l'agent : une réponse rejouée est refusée |
| Quelqu'un envoie des tâches au hub | l'API de contrôle n'accepte que des requêtes **locales au Mac**, signées avec un jeton stocké dans `~/.gpusync-remote/hub.json` (droits 600) |
| Un agent se fait passer pour un autre | un jeton par agent ; un agent n'accède qu'à ses propres tâches |
| Un agent lit des fichiers du Mac | il ne peut demander que les fichiers listés dans sa tâche (le dépôt synchronisé) |
| Une requête capturée est rejouée au hub | horodatage (± 5 min) + nonce à usage unique |
| Une requête est modifiée en route | la signature couvre l'identité, la méthode, le chemin, l'horodatage et le hash du corps |
| Accès au hub depuis Internet | seules les IP privées et locales sont acceptées (`--allow 192.168.1.0/24` pour restreindre encore) |
| Un fichier reçu écrase quelque chose hors du workspace | chemins absolus, `..`, lettres de lecteur et liens symboliques sortants refusés |
| Une commande ne s'arrête jamais | timeout (600 s par défaut), puis arrêt du processus **et de ses sous-processus** |

Testé : faux hub sans signature, signé avec un autre jeton, et signé avec le bon jeton mais rejoué → l'agent refuse
dans les trois cas et n'exécute rien.

Limites à connaître :

- le trafic **n'est pas chiffré** : le code envoyé et les sorties sont lisibles sur le réseau local.
  À utiliser sur un réseau de confiance (chez toi), pas sur un Wi-Fi public ;
- **un jeton d'agent permet de faire exécuter du code à cet agent** par quiconque le détient et peut joindre l'agent
  en se faisant passer pour le hub. Garde-le secret ; en cas de doute, `agents rm` puis `agents add` ;
- les horloges des machines doivent être à l'heure (± 5 min) ;
- les commandes s'exécutent avec les droits de l'utilisateur qui a lancé l'agent : ne le lance pas en administrateur.
  Arrête-le (`Ctrl+C`) quand tu ne t'en sers pas.

---

## 4. Fichiers

| Machine | Fichier | Contenu |
|---|---|---|
| Mac | `~/.gpusync-remote/hub.json` | jeton de contrôle, jetons des agents, port |
| Mac | `~/.gpusync-remote/hub/jobs/` | sorties des tâches |
| machine de test | `~/.gpusync-remote/agent.json` | URL du hub, nom, jeton |
| machine de test | `~/.gpusync-remote/agent/workspaces/` | dépôts synchronisés et fichiers produits |

`GS_REMOTE_HOME` permet de changer le dossier `~/.gpusync-remote`.

---

## 5. Protocole (pour référence)

Requêtes : en-têtes `X-GS-Who` (nom d'agent, ou `@ctl` pour la CLI), `X-GS-Time`, `X-GS-Nonce`,
`X-GS-Sig = HMAC(jeton, "req\nwho\nMÉTHODE\nchemin\nhorodatage\nnonce\nsha256(corps)")`.
Réponses : `X-GS-Resp-Sig = HMAC(jeton, "resp\nnonce-de-la-requête\nstatut\nsha256(corps)")`.

| Route | Appelant | Rôle |
|---|---|---|
| `POST /a/hello` | agent | annonce et infos machine |
| `POST /a/poll` | agent | attend une tâche (long-polling, 25 s max) |
| `POST /a/files` | agent | fichiers d'une tâche de synchronisation (zip) |
| `POST /a/output` | agent | envoie la sortie ; la réponse indique si la tâche doit être arrêtée |
| `POST /a/upload?job=` | agent | fichier demandé par `get` |
| `POST /a/done` | agent | fin de tâche (statut, code) |
| `GET /c/agents` | CLI (locale) | état des agents |
| `POST /c/jobs` | CLI (locale) | soumettre une tâche |
| `GET /c/jobs/<id>?offset=` | CLI (locale) | statut + sortie à partir de l'octet N |
| `POST /c/jobs/<id>/kill` | CLI (locale) | arrêter une tâche |
| `GET /c/jobs/<id>/result` | CLI (locale) | fichier récupéré par `get` |
