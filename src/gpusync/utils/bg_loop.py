import asyncio
import functools
import threading


class BackgroundLoop:
    """
    Démon en tâche de fond qui gère une boucle d'événements asyncio
    dédiée à l'exécution asynchrone purement non-bloquante du matériel.
    """

    _loop = None
    _thread = None
    _lock = threading.Lock()

    @classmethod
    def get_loop(cls):
        """Récupère ou crée la boucle d'événements asyncio globale en arrière-plan."""
        with cls._lock:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()

                def run_loop(loop):
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_forever()
                    finally:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

                cls._thread = threading.Thread(target=run_loop, args=(cls._loop,), daemon=True)
                cls._thread.start()
            return cls._loop


def auto_sync(coro_func):
    """
    Décorateur intelligent qui permet d'appeler une fonction asynchrone (coroutine) de manière synchrone.
    - S'il est appelé depuis un code standard (sans event loop), il envoie la tâche dans le thread
      d'arrière-plan et attend le résultat de manière synchrone.
    - S'il est appelé depuis une fonction `async def` (avec un event loop actif), il retourne
      la coroutine pour que l'utilisateur puisse utiliser `await`.
    """

    @functools.wraps(coro_func)
    def wrapper(*args, **kwargs):
        try:
            # Vérifie si une boucle asyncio est en train de tourner dans le thread courant
            asyncio.get_running_loop()
            is_async = True
        except RuntimeError:
            is_async = False

        coro = coro_func(*args, **kwargs)

        if is_async:
            return coro
        else:
            loop = BackgroundLoop.get_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()

    return wrapper
