import os

from .lightning import (SimpleDataModule,
                        SimpleModule)

def rec_num_workers():

    """Based on 
    https://github.com/pytorch/pytorch/blob/fb0f285638338da93960d2b654a59c9639671fc0/torch/utils/data/dataloader.py#L478-L505
    """

    max_num_worker_suggest = None
    cpuset_checked = False
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
            cpuset_checked = True
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count

    return max_num_worker_suggest
