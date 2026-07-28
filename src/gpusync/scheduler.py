import logging
from contextlib import contextmanager
from typing import Any, List

from .devices import ALL_DEVICE_MANAGERS
from .utils.devices import list_gpus
from gpusync.utils.imports import cp, cl

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, num_queues_per_device: int = 4, backend: str | None = None, gpu: Any = None):
        """
        Initialise un Scheduler.

        Args:
            num_queues_per_device (int): Nombre de files de commande (streams/queues) par GPU.
            backend (str, optional): "opencl" ou "cuda". Si None et gpu est fourni, on utilise le backend du gpu.
            gpu (GPUDeviceInfo ou raw device, optional): GPU spécifique à utiliser.
        """
        self.gpu = gpu
        if gpu is not None and hasattr(gpu, "backend"):
            self.backend = backend.lower() if backend is not None else gpu.backend
        else:
            self.backend = backend.lower() if backend is not None else "opencl"

        self.devices_managers: List[Any] = []
        self.num_queues_per_device = num_queues_per_device
        self._next_device_idx = 0

    def initialize(self, device_name_filter: str | None = None):
        """
        Initialise le(s) périphérique(s) de calcul.
        Si un `gpu` spécifique a été passé dans le constructeur, il est configuré directement.
        Sinon, effectue une découverte automatique filtrée par `device_name_filter`.
        """
        lookup_backend = "cuda" if self.backend == "llvm_cuda" else self.backend
        manager_cls: Any = next((c for c in ALL_DEVICE_MANAGERS if getattr(c, "backend", None) == lookup_backend), None)

        if manager_cls is None:
            raise ValueError(
                f"Backend '{self.backend}' non supporté. Backends disponibles: {[getattr(c, 'backend', None) for c in ALL_DEVICE_MANAGERS]}"
            )

        if self.gpu is not None:
            # 1. Utilisation du GPU spécifié
            raw_device = getattr(self.gpu, "raw_device", self.gpu)
            if hasattr(self.gpu, "backend") and self.gpu.backend != lookup_backend:
                logger.warning(
                    f"GPU backend mismatch: specified {self.gpu.backend} but requested {self.backend}. Attempting to resolve."
                )
                devs = manager_cls.discover_devices()
                if self.gpu.index < len(devs):
                    raw_device = devs[self.gpu.index]
                elif len(devs) > 0:
                    raw_device = devs[0]
                else:
                    raise ValueError(
                        f"Specified GPU has backend '{self.gpu.backend}', but requested backend '{self.backend}' has no compatible devices."
                    )

            if lookup_backend == "cuda" and isinstance(raw_device, int):
                raw_device = cp.cuda.Device(raw_device)

            manager = manager_cls(raw_device, num_queues=self.num_queues_per_device)
            manager.backend = self.backend
            self.devices_managers.append(manager)
            logger.info(f"Périphérique {self.backend.upper()} initialisé (spécifié) : {manager.name}")
        else:
            # 2. Découverte automatique dynamique
            devs = manager_cls.discover_devices(name_filter=device_name_filter)
            for dev in devs:
                try:
                    manager = manager_cls(dev, num_queues=self.num_queues_per_device)
                    manager.backend = self.backend
                    self.devices_managers.append(manager)
                    logger.info(f"Périphérique {self.backend.upper()} initialisé : {manager.name}")
                except Exception as e:
                    dev_name = getattr(dev, "name", dev)
                    logger.error(f"Impossible d'initialiser le GPU {self.backend.upper()} {dev_name}: {e}")

            if not self.devices_managers:
                raise RuntimeError(f"Le Scheduler {self.backend.upper()} n'a pu initialiser aucun périphérique.")

        logger.info(
            f"Scheduler {self.backend.upper()} prêt. {len(self.devices_managers)} périphérique(s) disponible(s)."
        )

    def get_best_device(self):
        """
        Répartit la charge entre les différents GPUs physiques détectés (Round-Robin).
        """
        if not self.devices_managers:
            raise RuntimeError("Scheduler non initialisé ou aucun périphérique détecté.")

        manager = self.devices_managers[self._next_device_idx]
        self._next_device_idx = (self._next_device_idx + 1) % len(self.devices_managers)
        return manager


class SwarmScheduler:
    """
    Le Quartier Général pour l'architecture ABA / Swarm.
    Initialise automatiquement CUDA pour les cartes Nvidia, et OpenCL pour le reste (Intel, AMD).
    Unifie tous les managers de périphériques en un seul pool de calcul.
    """

    def __init__(self, num_queues_per_device: int = 4, gpus: List[Any] | None = None):
        """
        Initialise un SwarmScheduler.

        Args:
            num_queues_per_device (int): Nombre de files de commande par GPU.
            gpus (List, optional): Liste restreinte de périphériques (GPUDeviceInfo ou bruts) à ajouter à l'essaim.
                                   Si None, découvre automatiquement tous les GPUs du système.
        """
        self.num_queues_per_device = num_queues_per_device
        self.gpus = gpus
        self.devices_managers: List[Any] = []
        self._next_device_idx = 0

    def initialize(self):
        # 1. Si une liste spécifique de périphériques est fournie
        if self.gpus is not None:
            for gpu in self.gpus:
                backend = getattr(gpu, "backend", None)
                raw_device = getattr(gpu, "raw_device", gpu)

                # Détection automatique du backend si non spécifié (heuristic)
                if backend is None:
                    try:
                        if isinstance(raw_device, cl.Device):
                            backend = "opencl"
                    except Exception:
                        pass
                    if backend is None:
                        backend = "cuda"  # fallback par défaut

                # Recherche du manager de périphérique associé au backend
                manager_cls = next((c for c in ALL_DEVICE_MANAGERS if c.backend == backend), None)
                if manager_cls is None:
                    logger.error(f"[Swarm] Aucun manager trouvé pour le backend '{backend}'")
                    continue

                try:
                    if backend == "cuda" and isinstance(raw_device, int):
                        raw_device = cp.cuda.Device(raw_device)
                    manager = manager_cls(raw_device, num_queues=self.num_queues_per_device)
                    self.devices_managers.append(manager)
                    logger.info(f"[Swarm] Périphérique {backend.upper()} initialisé (spécifié) : {manager.name}")
                except Exception as e:
                    logger.error(f"[Swarm] Impossible d'initialiser {backend.upper()} pour {raw_device}: {e}")
        else:
            # 2. Découverte automatique sur l'ensemble des managers enregistrés
            for manager_cls in ALL_DEVICE_MANAGERS:
                devs = manager_cls.discover_devices()
                for dev in devs:
                    # En mode découverte auto sur Swarm, on évite le doublon Nvidia sous OpenCL
                    if manager_cls.backend == "opencl":
                        dev_name = getattr(dev, "name", "")
                        dev_vendor = getattr(dev, "vendor", "")
                        if "nvidia" in dev_name.lower() or "nvidia" in dev_vendor.lower():
                            continue  # Déjà géré par CUDA en principe

                    try:
                        manager = manager_cls(dev, num_queues=self.num_queues_per_device)
                        self.devices_managers.append(manager)
                        logger.info(f"[Swarm] {manager_cls.backend.upper()} détecté et ajouté : {manager.name}")
                    except Exception as e:
                        dev_name = getattr(dev, "name", dev)
                        logger.error(f"[Swarm] Erreur lors de l'initialisation de {dev_name}: {e}")

        if not self.devices_managers:
            raise RuntimeError("SwarmScheduler n'a pu initialiser AUCUN périphérique de calcul !")

        logger.info(f"SwarmScheduler PRÊT. {len(self.devices_managers)} périphérique(s) dans l'essaim.")

    def get_best_device(self):
        if not self.devices_managers:
            raise RuntimeError("SwarmScheduler non initialisé.")
        manager = self.devices_managers[self._next_device_idx]
        self._next_device_idx = (self._next_device_idx + 1) % len(self.devices_managers)
        return manager

    def get_device(self, identifier: str):
        """
        identifier: ex "cuda:0", "opencl:1", ou "intel"
        """
        identifier = identifier.lower()
        if ":" in identifier:
            backend, idx_str = identifier.split(":")
            gpu_idx = int(idx_str)
            # Filtre par backend
            managers = [m for m in self.devices_managers if hasattr(m, "backend") and m.backend == backend]
            if gpu_idx < len(managers):
                return managers[gpu_idx]
        else:
            # Filtre par nom
            for m in self.devices_managers:
                if hasattr(m, "name") and identifier in m.name.lower():
                    return m

        # Fallback si non trouvé
        logger.warning(f"[Swarm] Périphérique '{identifier}' non trouvé, fallback sur Round-Robin.")
        return self.get_best_device()


_default_scheduler = None


def get_default_scheduler() -> Scheduler:
    """
    Récupère ou initialise un scheduler par défaut avec le premier GPU disponible.
    """
    global _default_scheduler
    if _default_scheduler is None:
        gpus = list_gpus()
        if not gpus:
            raise RuntimeError("Aucun GPU disponible pour initialiser le Scheduler par défaut.")
        _default_scheduler = Scheduler(gpu=gpus[0])
        _default_scheduler.initialize()
    return _default_scheduler


@contextmanager
def use_gpu(backend: str = "opencl", gpu_idx: int = 0):
    """
    Context manager pour initialiser et utiliser un Scheduler GPUsync proprement.

    Exemple :
        with use_gpu(backend="opencl") as sched:
            res = mon_kernel(data)
    """
    gpus = list_gpus()
    if not gpus:
        raise RuntimeError("Aucun GPU disponible.")

    # Sélectionner le GPU par rapport à l'index et au backend demandé
    selected_gpu = None
    if gpu_idx < len(gpus):
        selected_gpu = gpus[gpu_idx]
    else:
        selected_gpu = gpus[0]

    sched = Scheduler(gpu=selected_gpu, backend=backend)
    sched.initialize()

    global _default_scheduler
    old_default = _default_scheduler
    _default_scheduler = sched

    try:
        yield sched
    finally:
        _default_scheduler = old_default
        for manager in getattr(sched, "devices_managers", []):
            if hasattr(manager, "_buffer_pool"):
                manager._buffer_pool.clear()
