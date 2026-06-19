"""
Mock module for Python IDE typing and autocomplete for OpenCL.
These functions are intercepted by the AST compiler and transpiled into OpenCL C.
"""

CLK_LOCAL_MEM_FENCE = "CLK_LOCAL_MEM_FENCE"
CLK_GLOBAL_MEM_FENCE = "CLK_GLOBAL_MEM_FENCE"

def get_local_id(dim: int) -> int:
    """Returns the local ID of the thread within its work-group."""
    pass

def get_group_id(dim: int) -> int:
    pass
    
def get_global_id(dim: int) -> int:
    pass

def local_memory(shape: tuple, dtype: str):
    """
    Allocates __local memory shared across the work-group.
    Translates to `__local dtype name[shape[0]][shape[1]]...`
    """
    pass

def barrier(flags):
    """
    Synchronizes threads within a work-group.
    """
    pass
