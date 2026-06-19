import pyopencl as cl
import logging
from typing import List
from .device import DeviceManager, CUDADeviceManager
from .bg_loop import auto_sync

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, num_queues_per_device: int = 4, backend: str = "opencl"):
        self.backend = backend.lower()
        self.devices_managers = []
        self.num_queues_per_device = num_queues_per_device
        self._next_device_idx = 0

    @auto_sync
    async def initialize(self, device_name_filter: str = None):
        """
        Découvre les cartes graphiques sur l'ordinateur et initialise les DeviceManagers.
        Si device_name_filter est renseigné, seul le périphérique contenant ce nom sera utilisé.
        Doit être appelé avant toute exécution asynchrone.
        """
        if self.backend in ["cuda", "cublas"]:
            import cupy as cp
                
            num_devices = cp.cuda.runtime.getDeviceCount()
            if num_devices == 0:
                raise RuntimeError("Aucune carte CUDA trouvée sur ce système.")

            for i in range(num_devices):
                device = cp.cuda.Device(i)
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode('utf-8')
                
                if device_name_filter and device_name_filter.lower() not in name.lower():
                    continue
                    
                try:
                    manager = CUDADeviceManager(device, num_queues=self.num_queues_per_device)
                    self.devices_managers.append(manager)
                    logger.info(f"Périphérique CUDA initialisé : {name}")
                except Exception as e:
                    logger.error(f"Impossible d'initialiser le GPU CUDA {name}: {e}")

            if not self.devices_managers:
                raise RuntimeError(f"Le Scheduler CUDA n'a pu initialiser aucun périphérique.")
                
            logger.info(f"Scheduler CUDA prêt. {len(self.devices_managers)} périphérique(s) disponible(s).")

        else:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("Aucune plateforme OpenCL trouvée sur ce système.")

            for platform in platforms:
                # On cherche prioritairement des GPUs
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if not devices:
                    # Fallback sur les CPUs ou accélérateurs si pas de GPU
                    devices = platform.get_devices() 
                
                for device in devices:
                    if device_name_filter and device_name_filter.lower() not in device.name.lower():
                        continue
                        
                    try:
                        manager = DeviceManager(device, num_queues=self.num_queues_per_device)
                        self.devices_managers.append(manager)
                        logger.info(f"Périphérique OpenCL initialisé : {device.name}")
                    except Exception as e:
                        logger.error(f"Impossible d'initialiser le GPU OpenCL {device.name}: {e}")

            if not self.devices_managers:
                available_devices = [d.name for p in platforms for d in p.get_devices()]
                raise RuntimeError(f"Le Scheduler OpenCL n'a pu initialiser aucun périphérique correspondant à '{device_name_filter}'.\nPériphériques disponibles : {available_devices}")
                
            logger.info(f"Scheduler OpenCL prêt. {len(self.devices_managers)} périphérique(s) disponible(s).")

    def get_best_device(self) -> DeviceManager:
        """
        Répartit la charge entre les différents GPUs physiques détectés (Round-Robin).
        """
        if not self.devices_managers:
            raise RuntimeError("Scheduler non initialisé ou aucun périphérique détecté.")
        
        manager = self.devices_managers[self._next_device_idx]
        self._next_device_idx = (self._next_device_idx + 1) % len(self.devices_managers)
        return manager
