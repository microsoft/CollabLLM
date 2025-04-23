import os
import torch
import torch.distributed as dist
from datetime import timedelta


def init_distributed_mode():
    # Ensure the script is being run with distributed launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        raise RuntimeError("Please set RANK, WORLD_SIZE, and LOCAL_RANK in the environment variables.")

    # Initialize the process group with a timeout of one day
    timeout = timedelta(days=1)  # 1 day timeout

    dist.init_process_group(
        backend='nccl',  
        init_method=None,
        timeout=timeout
    )

    torch.cuda.set_device(local_rank)
    dist.barrier()

    print(f"Distributed initialized. Rank: {rank}, World Size: {world_size}")
