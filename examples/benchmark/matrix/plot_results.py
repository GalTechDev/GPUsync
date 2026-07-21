import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def main():
    # Chemins des fichiers
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'results.csv')
    
    if not os.path.exists(csv_file):
        print(f"Erreur : Le fichier {csv_file} n'existe pas.")
        return
        
    # Chargement des données
    df = pd.read_csv(csv_file)
    
    # Filtrer les éventuels échecs s'il en reste
    df = df[df['Status'] == 'SUCCESS']
    
    # Configuration du style
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({'font.size': 10})
    
    # Couleurs personnalisées pour bien différencier
    algorithms = df['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    # Création de la figure (2 lignes, 3 colonnes)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    metrics = [
        ('Time_s', 'Temps d\'exécution (s)', axes[0,0], False),
        ('TFLOPS', 'Débit (TFLOPS)', axes[0,1], False),
        ('VRAM_GB', 'VRAM Utilisée (Go)', axes[0,2], False),
        ('RAM_GB', 'RAM CPU Utilisée (Go)', axes[1,0], False),
        ('Max_Error', 'Erreur Mathématique Maximale', axes[1,1], True) # Echelle logarithmique pour l'erreur
    ]
    
    for metric_col, title, ax, log_y in metrics:
        for i, algo in enumerate(algorithms):
            data = df[df['Algorithm'] == algo]
            ax.plot(data['Matrix_Size'], data[metric_col], marker='o', linewidth=2, label=algo, color=colors[i])
            
        ax.set_title(f'{title} vs Taille Matrice', fontsize=14, fontweight='bold')
        ax.set_xlabel('Taille de la Matrice (N x N)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xscale('log', base=2)
        if log_y:
            ax.set_yscale('log')
            
        ax.set_xticks(df['Matrix_Size'].unique())
        ax.set_xticklabels([str(int(x)) for x in df['Matrix_Size'].unique()], rotation=45)
        
        # Ajouter les barres rouges pour la mémoire
        if metric_col == 'VRAM_GB':
            ax.axhline(y=16.0, color='r', linestyle='--', linewidth=2, label='Limite VRAM (16 Go)')
            
    # Le 6ème graphique (en bas à droite) servira de légende globale pour ne pas surcharger les graphes
    axes[1,2].axis('off')
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[1,2].legend(handles, labels, loc='center', fontsize=14, title="Algorithmes", title_fontsize=16, framealpha=0.9)
    
    plt.tight_layout()
    
    # Sauvegarde de l'image
    plot_file = os.path.join(base_dir, 'benchmark_results_complet.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {plot_file}")
    
    # Affichage interactif
    plt.show()

if __name__ == "__main__":
    main()
