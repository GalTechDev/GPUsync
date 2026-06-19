import pyopencl as cl
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self, device: cl.Device, num_queues: int = 16):
        self.device = device
        # Création d'un unique contexte pour ce composant matériel (GPU)
        self.context = cl.Context([device])
        
        # Pour forcer la concurrence, nous créons un 'Pool' de CommandQueues.
        # En assignant différentes tâches à différentes queues d'un même contexte, 
        # le driver matériel répartit l'exécution sur les unités de calcul (Compute Units) disponibles.
        properties = cl.command_queue_properties.PROFILING_ENABLE
        self.queues = [
            cl.CommandQueue(self.context, self.device, properties=properties) 
            for _ in range(num_queues)
        ]
        
        self.next_queue_idx = 0
        logger.info(f"DeviceManager initialisé : {device.name} avec {num_queues} Command Queues.")

    def get_queue(self) -> cl.CommandQueue:
        """
        Retourne une file de commande selon une stratégie Round-Robin.
        """
        queue = self.queues[self.next_queue_idx]
        self.next_queue_idx = (self.next_queue_idx + 1) % len(self.queues)
        return queue

class CUDADeviceManager:
    def __init__(self, device, num_queues: int = 16):
        import cupy as cp
        self.device = device
        
        # Rend le contexte actif
        self.device.use()
        
        # Pour forcer la concurrence CUDA, nous créons un Pool de Streams.
        self.queues = [
            cp.cuda.Stream(non_blocking=True)
            for _ in range(num_queues)
        ]
        
        self.next_queue_idx = 0
        
        # Récupère le nom du périphérique pour les logs
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props['name'].decode('utf-8')
        logger.info(f"CUDADeviceManager initialisé : {name} avec {num_queues} Streams.")

    def get_queue(self):
        """
        Retourne un stream CUDA selon une stratégie Round-Robin.
        """
        queue = self.queues[self.next_queue_idx]
        self.next_queue_idx = (self.next_queue_idx + 1) % len(self.queues)
        return queue
