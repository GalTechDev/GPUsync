import os
import sys
import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gr_core import gr_soft_dtw_loss, compute_cost_matrix, get_alignment_matrix_connected

def generate_sensor_data(D=3, N=20, noise_std=0.4):
    """
    Génère des séries temporelles synthétiques avec déphasage physique (retards).
    D: nombre de capteurs.
    N: longueur de la séquence.
    """
    t = np.linspace(0, 3 * np.pi, N)
    
    # 1. Signaux de référence propres (Y)
    Y_cpu = np.zeros((D, N), dtype=np.float32)
    # Capteur 0 : référence
    Y_cpu[0] = np.sin(t)
    # Capteur 1 : en retard de 2 pas de temps
    Y_cpu[1] = np.sin(t - 0.4)
    # Capteur 2 : en retard de 4 pas de temps
    Y_cpu[2] = np.sin(t - 0.8)
    
    # 2. Ajout de bruit et distorsion locale pour les signaux observés (X)
    X_cpu = Y_cpu.copy()
    np.random.seed(42)
    
    # Ajout d'un bruit important indépendant
    X_cpu += np.random.normal(0, noise_std, size=X_cpu.shape).astype(np.float32)
    
    # Ajout d'une anomalie locale (bruit extrême) sur le capteur 1 à l'index 10
    X_cpu[1, 9:12] += 2.0
    
    return torch.tensor(X_cpu), torch.tensor(Y_cpu)

def optimize_signals(X_obs, Y, edges, lambd=0.0, lr=0.05, epochs=300):
    """
    Optimise un signal estimé X_est pour minimiser la perte GR-S-DTW avec Y,
    tout en restant proche des observations X_obs.
    """
    X_est = X_obs.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([X_est], lr=lr)
    
    gamma = 0.1
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Perte d'alignement (GR-S-DTW)
        loss_align = gr_soft_dtw_loss(X_est, Y, gamma, edges, lambd)
        
        # Perte de fidélité aux observations
        loss_fidelity = torch.sum((X_est - X_obs) ** 2)
        
        # Perte totale
        loss = loss_fidelity + 0.1 * loss_align
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"    [Optimisation] Époque {epoch + 1}/{epochs} | Perte: {loss.item():.4f}", flush=True)
        
    return X_est.detach()

def compute_alignment_error(X_est, Y_clean, gamma=0.1):
    """
    Calcule l'erreur de chemin d'alignement par rapport à la matrice identité (alignement parfait).
    """
    D, N = X_est.shape
    X_est_grad = X_est.clone().detach().requires_grad_(True)
    I = torch.eye(N)
    
    total_error = 0.0
    errors_per_sensor = []
    
    for v in range(D):
        C_v = compute_cost_matrix(X_est_grad[v], Y_clean[v])
        _, A_v = get_alignment_matrix_connected(C_v, gamma)
        
        # Erreur quadratique par rapport à l'alignement parfait (Identité)
        err = torch.sum((A_v - I) ** 2).item()
        errors_per_sensor.append(err)
        total_error += err
        
    return total_error, errors_per_sensor

def main():
    print("=== SIMULATION DE RÉSEAU DE CAPTEURS : QUALITÉ D'ALIGNEMENT ===")
    
    D = 3  # 3 capteurs
    N = 20
    
    # Définition des arêtes avec les décalages physiques (delays)
    # Capteur 0 et 1 : décalage de +2
    # Capteur 1 et 2 : décalage de +2
    edges = [
        (0, 1, 2),
        (1, 2, 2)
    ]
    
    # Génération des données
    X_obs, Y_clean = generate_sensor_data(D=D, N=N, noise_std=0.4)
    
    print("\n--- Optimisation 1 : Sans Régularisation de Graphe (Independent Soft-DTW, lambda = 0.0) ---")
    X_est_indep = optimize_signals(X_obs, Y_clean, edges, lambd=0.0, lr=0.05, epochs=300)
    err_indep_total, err_indep_sensors = compute_alignment_error(X_est_indep, Y_clean)
    
    print("\n--- Optimisation 2 : Avec Régularisation de Graphe (GR-S-DTW, lambda = 5.0) ---")
    X_est_graph = optimize_signals(X_obs, Y_clean, edges, lambd=5.0, lr=0.05, epochs=300)
    err_graph_total, err_graph_sensors = compute_alignment_error(X_est_graph, Y_clean)
    
    print("\n--- COMPARAISON DES PERFORMANCES D'ALIGNEMENT (Erreurs de chemin vs Identité) ---")
    improvement = (err_indep_total - err_graph_total) / err_indep_total * 100
    print(f"Erreur totale de chemin d'alignement (Norme de Frobenius vs Identité) :")
    print(f"  - Independent Soft-DTW : {err_indep_total:.4f}")
    print(f"  - GR-S-DTW (Proposé)   : {err_graph_total:.4f}")
    print(f"Réduction de l'erreur d'alignement : {improvement:.2f}%")
    
    print("\nDétail par capteur :")
    for v in range(D):
        imp_v = (err_indep_sensors[v] - err_graph_sensors[v]) / err_indep_sensors[v] * 100
        print(f"  * Capteur {v} :")
        print(f"    - Independent : {err_indep_sensors[v]:.4f}")
        print(f"    - GR-S-DTW    : {err_graph_sensors[v]:.4f}")
        print(f"    - Amélioration : {imp_v:.2f}%")
        
    # Sauvegarde des résultats pour le rapport
    os.makedirs("examples/results/reports", exist_ok=True)
    report_path = "examples/results/reports/simulation_sensor_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== RÉSULTATS DE SIMULATION D'ALIGNEMENT GR-S-DTW ===\n")
        f.write(f"Erreur totale Independent Soft-DTW : {err_indep_total:.4f}\n")
        f.write(f"Erreur totale GR-S-DTW (lambda=5.0) : {err_graph_total:.4f}\n")
        f.write(f"Réduction globale de l'erreur d'alignement : {improvement:.2f}%\n\n")
        f.write("Détails des erreurs par capteur :\n")
        for v in range(D):
            f.write(f"  Capteur {v} : Indep={err_indep_sensors[v]:.4f} | Graph={err_graph_sensors[v]:.4f} | Imp={((err_indep_sensors[v] - err_graph_sensors[v]) / err_indep_sensors[v] * 100):.2f}%\n")
        
    print(f"\nRapport écrit dans : {report_path}")

if __name__ == "__main__":
    main()
