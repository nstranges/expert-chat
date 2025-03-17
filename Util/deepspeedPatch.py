import os
import subprocess
from mpi4py import MPI
import deepspeed.comm as dist

# Custom to fix deepSpeeds bug
def mpi_discovery(distributed_port=29500, verbose=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        import shlex
        try:
            hostname_cmd = shlex.split("hostname -i")
            result = subprocess.check_output(hostname_cmd)
            master_addr = result.decode('utf-8').split()[0]
        except subprocess.CalledProcessError:  # If `hostname -I` fails, use socket fallback
            import socket
            master_addr = socket.gethostbyname(socket.gethostname())

    # Broadcast master_addr to all ranks
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    # Set environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        print(f"[MPI] world_rank={rank}, local_rank={local_rank}, world_size={world_size}, master_addr={master_addr}, master_port={distributed_port}")

# Trying to fix DeepSpeed bug
dist.mpi_discovery = mpi_discovery