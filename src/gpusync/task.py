class AsyncTask:
    """
    Représente une tâche de calcul asynchrone (Kernel JIT) qui peut être exécutée
    sur n'importe quel périphérique disponible (CUDA, OpenCL, LLVM ou LLVM NVPTX).
    """

    def __init__(
        self,
        kernel_code_opencl: str,
        kernel_name_opencl: str,
        kernel_code_cuda: str,
        kernel_name_cuda: str,
        kernel_code_llvm: str | None = None,
        kernel_name_llvm: str | None = None,
        kernel_code_ptx: str | None = None,
        kernel_name_ptx: str | None = None,
    ):
        self.kernel_code_opencl = kernel_code_opencl
        self.kernel_name_opencl = kernel_name_opencl
        self.kernel_code_cuda = kernel_code_cuda
        self.kernel_name_cuda = kernel_name_cuda
        self.kernel_code_llvm = kernel_code_llvm
        self.kernel_name_llvm = kernel_name_llvm
        self.kernel_code_ptx = kernel_code_ptx
        self.kernel_name_ptx = kernel_name_ptx
        self._compiled_programs: dict = {}

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
