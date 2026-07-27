import subprocess
import sys
import os
import threading

def read_stream(stream, prefix):
    """
    Lit un flux de sortie (stdout/stderr) et l'affiche ligne par ligne avec un préfixe.
    """
    for line in iter(stream.readline, ''):
        print(f"[{prefix}] {line.strip()}", flush=True)
    stream.close()

def run_script(script_name, python_exe, prefix):
    """
    Lance un script Python en arrière-plan et démarre un thread pour lire sa sortie.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    # Configuration des variables d'environnement (PYTHONPATH)
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env["PYTHONPATH"] = os.path.join(root_dir, "src") + os.pathsep + env.get("PYTHONPATH", "")
    
    process = subprocess.Popen(
        [python_exe, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )
    
    # Démarrage des threads pour capturer stdout et stderr en temps réel
    t_out = threading.Thread(target=read_stream, args=(process.stdout, prefix))
    t_err = threading.Thread(target=read_stream, args=(process.stderr, f"{prefix}-ERR"))
    
    t_out.start()
    t_err.start()
    
    return process, t_out, t_err

def main():
    python_exe = r"C:\Users\moreaum\Documents\Code\Python\python.exe"
    
    scripts = {
        "test_phase_a_b.py": "TEST A-B",
        "test_phase_c.py": "TEST C",
        "test_phase_d_gpusync.py": "TEST D-GPUsync",
        "simulate_sensor_network.py": "SIMULATION"
    }
    
    print("=== LANCEMENT DES SCRIPTS DE TEST EN PARALLÈLE ===", flush=True)
    
    running_processes = []
    
    # 1. Lancement de tous les processus en parallèle
    for script, prefix in scripts.items():
        print(f"[Master] Démarrage de {script}...", flush=True)
        proc, t_out, t_err = run_script(script, python_exe, prefix)
        running_processes.append((script, proc, t_out, t_err))
        
    print("[Master] Tous les processus ont été lancés. Attente de la fin de l'exécution...\n", flush=True)
    
    # 2. Attente de tous les processus et de leurs threads
    exit_codes = {}
    for script, proc, t_out, t_err in running_processes:
        code = proc.wait()
        t_out.join()
        t_err.join()
        exit_codes[script] = code
        
    print("\n=== BILAN DE L'EXÉCUTION ===", flush=True)
    for script, code in exit_codes.items():
        status = "OK" if code == 0 else f"ERREUR (Code {code})"
        print(f"  * {script} : {status}", flush=True)

if __name__ == "__main__":
    main()
