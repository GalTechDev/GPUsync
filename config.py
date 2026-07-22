# -*- coding: utf-8 -*-
"""
Configuration des variables pour l'application d'analyse
"""

# Chemins et dossiers
ROMANE_DATA_PATH = 'RomaneData'  # Préfixe pour les chemins de données
FILE_SUFFIX = ''  # Suffixe pour les noms de fichiers (avant l'extension)
UPLOAD_FOLDER = 'uploads'
PDF_FOLDER = 'pdfs'
RESULTS_FOLDER = 'results'

# Configuration de la base de données
DATABASE = 'database.db'

# Configuration pour l'analyse
ALIM_TEST = False  # Variable de contrôle pour les tests
DEFAULT_NB_CLUSTERS = 15  # Nombre de clusters par défaut
DEFAULT_W = 5  # Paramètre de fenêtre pour la distance DTW
NB_INSTANCE_MINIMUM = 5  # Nombre minimum d'instances par cluster
MAX_ITERATIONS = 5  # Nombre maximum d'itérations

# Colonnes et variables importantes
ID_COLUMN = 'elev_id'  # Colonne utilisée comme identifiant partout dans le code
DEFAULT_COLUMNS = [
    'Set_TEMP',
    'Set_OXYGENE'
]

# Colonnes additionnelles qui peuvent être utilisées dans l'analyse
ADDITIONAL_COLUMNS = [
    'SetDay_TEMP',
    'SetDay_OXYGENE',
    'Set_Delta_OXYGENE',
    'SetDay_Delta_TEMP',
    'SetDay_Delta_OXYGENE',
    'Set_Delta_RENVLT',
    'Set_RENVLT',
    'Set_alim',
    'Set_Delta_alim'
]

# Colonnes statistiques générées
STATS_COLUMNS = ['min', 'max', 'mean', '25', '50', '75']

# Variables initialisées dans le processus
INITIAL_VALUES = {
    'cluster': -1,
    'MEAS': -1.0,
    'MEAS_r': -1.0
}

# Configuration matplotlib
MATPLOTLIB_CONFIG = {
    'font.size': 45,
    'font.weight': 'bold'
}

# Taille des figures
FIGURE_SIZE = (60.0, 100.0)
FIGURE_DPI = 80
