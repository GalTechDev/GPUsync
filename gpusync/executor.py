import asyncio
import pyopencl as cl
import logging

logger = logging.getLogger(__name__)

async def await_event(event_or_stream):
    """
    Suspend l'exécution de la coroutine Python jusqu'à ce que l'événement matériel 
    (transfert PCIe ou Kernel GPU) soit terminé, que ce soit OpenCL ou CUDA.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # Si c'est un Stream CUDA (CuPy)
    if hasattr(event_or_stream, 'add_callback'):
        stream = event_or_stream
        def _stream_callback(stream, status, arg):
            loop.call_soon_threadsafe(future.set_result, stream)
        try:
            stream.add_callback(_stream_callback, None)
        except Exception as e:
            logger.warning(f"Impossible d'ajouter une callback CUDA : {e}")
            # Fallback en polling
            while not stream.done:
                await asyncio.sleep(0.001)
            return stream
        return await future

    # Si c'est un événement OpenCL
    elif hasattr(event_or_stream, 'command_execution_status'):
        event = event_or_stream
        def _event_callback(status):
            import pyopencl as cl
            if status == cl.command_execution_status.COMPLETE:
                loop.call_soon_threadsafe(future.set_result, event)
            else:
                err = RuntimeError(f"Erreur d'exécution GPU (status: {status})")
                loop.call_soon_threadsafe(future.set_exception, err)

        try:
            import pyopencl as cl
            event.set_callback(cl.command_execution_status.COMPLETE, _event_callback)
        except Exception as e:
            import pyopencl as cl
            logger.warning(f"Impossible d'utiliser set_callback OpenCL, fallback en polling : {e}")
            while event.command_execution_status != cl.command_execution_status.COMPLETE:
                if event.command_execution_status < 0:
                    raise RuntimeError(f"Erreur OpenCL durant le polling : status {event.command_execution_status}")
                await asyncio.sleep(0.001)
            return event

        return await future
    else:
        raise ValueError("Objet d'attente matériel non reconnu.")
