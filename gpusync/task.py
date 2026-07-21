class AsyncTask:
    """
    Représente une tâche de calcul asynchrone (Kernel JIT) qui peut être exécutée
    sur n'importe quel périphérique disponible (CUDA ou OpenCL).
    """
    def __init__(self, kernel_code_opencl: str, kernel_name_opencl: str, kernel_code_cuda: str, kernel_name_cuda: str):
        self.kernel_code_opencl = kernel_code_opencl
        self.kernel_name_opencl = kernel_name_opencl
        self.kernel_code_cuda = kernel_code_cuda
        self.kernel_name_cuda = kernel_name_cuda
        self._compiled_programs = {}

    async def execute(self, scheduler, global_size, local_size, *args, stream=None, fire_and_forget=False):
        """
        Orchestre l'exécution du kernel en déléguant l'exécution matérielle
        directement au manager du périphérique cible.
        """
        if not args:
            device_manager = scheduler.get_best_device()
        else:
            device_manager = args[0].device_manager

        return await device_manager.execute_kernel(
            self, global_size, local_size, *args, stream=stream, fire_and_forget=fire_and_forget
        )
