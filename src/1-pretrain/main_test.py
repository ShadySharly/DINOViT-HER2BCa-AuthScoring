import sys, os, logging

# Set parent root to make it visible for this file
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))

# Set logging with debug options to gain more information
logging.basicConfig(level=logging.DEBUG)

# Set the number of OpenMP threads
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    logging.debug(f"Setting up process group with rank {rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    logging.debug("Destroying process group")
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    logging.debug(f"Creating model for rank {rank}")
    model = torch.nn.Linear(10, 10).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    logging.debug(f"Cleaning up for rank {rank}")
    cleanup()

if __name__ == "__main__":
    available_gpus = torch.cuda.device_count()
    world_size = available_gpus  # or the number of GPUs available
    logging.debug(f"Spawning processes with world size {world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True) # type: ignore
