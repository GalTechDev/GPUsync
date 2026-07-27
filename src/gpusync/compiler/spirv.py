import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def compile_opencl_to_spirv(opencl_source: str, kernel_name: str) -> bytes:
    """
    Tente de compiler le code source OpenCL C en binaire SPIR-V (.spv)
    en utilisant en priorité le compilateur clspv local (vendored) s'il existe,
    ou clspv installé globalement sur le système.

    Retourne les octets du fichier .spv compilé ou lève une exception en cas d'échec.
    """
    # Résoudre le chemin potentiel du compilateur clspv vendored localement
    gpusync_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clspv_filename = "clspv.exe" if os.name == "nt" else "clspv"
    local_clspv_path = os.path.join(gpusync_dir, "bin", clspv_filename)

    if os.path.exists(local_clspv_path):
        clspv_cmd = local_clspv_path
    else:
        clspv_cmd = "clspv"

    # 1. Vérifier si clspv est disponible (localement ou globalement)
    try:
        subprocess.run([clspv_cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError(
            f"Le compilateur hors-ligne '{clspv_cmd}' (Google clspv) est introuvable. "
            "Assurez-vous qu'il est installé, dans le PATH de votre système, ou packagé dans gpusync/bin/."
        ) from None

    # 2. Créer des fichiers temporaires pour l'entrée et la sortie
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "kernel.cl")
        output_file = os.path.join(tmpdir, "kernel.spv")

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(opencl_source)

        # 3. Exécuter clspv
        cmd = [clspv_cmd, input_file, "-o", output_file, f"-entrypoint={kernel_name}"]

        logger.info(f"Compilation SPIR-V : {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Échec de la compilation SPIR-V par clspv :\n{error_msg}")

        if not os.path.exists(output_file):
            raise RuntimeError("La compilation clspv s'est terminée mais aucun fichier .spv n'a été généré.")

        with open(output_file, "rb") as f:
            spirv_bytes = f.read()

        return spirv_bytes
