import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_file = os.path.join(os.path.dirname(__file__), 'results.csv')
    if not os.path.exists(csv_file):
        print(f"Fichier {csv_file} introuvable. Lancez le benchmark d'abord.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Configuration du style
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "figure.facecolor": "#121212"})
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Benchmark K-Means Out-Of-Core : GPUsync vs Scikit-Learn', fontsize=16, color='white')
    
    # 1. Temps d'exécution
    ax1 = axes[0]
    sns.lineplot(data=df, x='N', y='Time_s', hue='Impl', marker='o', ax=ax1, palette=['#ff5555', '#55ff55'])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Temps d\'exécution (Plus bas = Meilleur)', color='white')
    ax1.set_xlabel('Nombre de points (N)', color='white')
    ax1.set_ylabel('Temps (secondes)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    # 2. Utilisation Mémoire (Seulement GPUsync pour montrer le Out-Of-Core)
    ax2 = axes[1]
    df_gpu = df[df['Impl'] == 'GPUsync']
    ax2.plot(df_gpu['N'], df_gpu['VRAM_GB'], marker='o', color='#55aaff', label='VRAM Peak (Go)')
    ax2.plot(df_gpu['N'], df_gpu['RAM_GB'], marker='s', color='#ffaa55', label='RAM Peak (Go)')
    ax2.set_xscale('log')
    ax2.set_title('Utilisation Mémoire de GPUsync (Streaming VRAM)', color='white')
    ax2.set_xlabel('Nombre de points (N)', color='white')
    ax2.set_ylabel('Mémoire (Go)', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'perf_plot.png')
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Graphique sauvegardé sous : {output_path}")

if __name__ == "__main__":
    main()
