import numpy as np
import asyncio
from gpusync.compiler import gpusync_kernel
from gpusync.scheduler import Scheduler

@gpusync_kernel
def add_kernel(ctx: int, a: list, b: list) -> tuple:
    return a[ctx] + b[ctx]

async def main():
    print('Testing LLVM backend...')
    sched = Scheduler(backend='llvm')
    sched.initialize()
    
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    
    res = await add_kernel(a, b, global_size=4, scheduler=sched)
    print('Result:', res)
    assert np.allclose(res, a + b), 'Computation failed!'
    print('LLVM execution successful!')

asyncio.run(main())
