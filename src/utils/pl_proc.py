'''
    Universal settings for parallel processing across scripts.
'''

def init_worker(threads):
    """Runs once in every worker process: limits its BLAS/OpenMP threads so that the workers do not oversubscribe the CPU."""
    global _thread_limits
    from threadpoolctl import threadpool_limits
    _thread_limits = threadpool_limits(threads)