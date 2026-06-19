import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'pipeline_tuning.csv')
    
    if not os.path.exists(csv_file):
        print(f"Fichier introuvable : {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    # Nettoyer les éventuelles lignes mal formatées (dues à l'exécution simultanée)
    df = df.dropna()
    
    # Création d'une colonne combinée pour le plot
    df['Config'] = 'Chunk:' + df['MAX_CHUNK'].astype(str) + ' | In-Flight:' + df['IN_FLIGHT_LIMIT'].astype(str)
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trier par débit décroissant
    df_sorted = df.sort_values('TFLOPS', ascending=False)
    
    # 1. Plot TFLOPS
    sns.barplot(data=df_sorted, x='TFLOPS', y='Config', ax=axes[0], palette='viridis')
    axes[0].set_title('Débit par Configuration de Pipeline (TFLOPS)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('TFLOPS (Plus grand = Meilleur)', fontsize=12)
    axes[0].set_ylabel('Configuration', fontsize=12)
    
    # 2. Plot VRAM
    sns.barplot(data=df_sorted, x='VRAM_GB', y='Config', ax=axes[1], palette='magma')
    axes[1].set_title('Consommation VRAM (Go)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('VRAM Utilisée (Go) (Plus petit = Meilleur)', fontsize=12)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    
    plot_file = os.path.join(base_dir, 'pipeline_tuning_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous : {plot_file}")
    
    plt.show()

if __name__ == '__main__':
    main()
