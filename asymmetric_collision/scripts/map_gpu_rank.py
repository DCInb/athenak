#!/usr/bin/env python3
"""Pin each node-local MPI rank to a distinct CPU near its Kokkos-selected GPU."""
import csv
import ctypes
import hashlib
import io
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def expand_cpus(value):
    result = set()
    for part in value.split(','):
        bounds = part.split('-')
        result.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def allocated_cpus(job, hostname):
    """Read only this node's CPUs, including compressed Slurm node lists."""
    allowed = set()
    for nodes, cpus in re.findall(r'\bNodes=(\S+)\s+CPU_IDs=([0-9,-]+)', job):
        names = subprocess.check_output(['scontrol', 'show', 'hostnames', nodes], text=True)
        if hostname.split('.')[0] in {name.split('.')[0] for name in names.split()}:
            allowed.update(expand_cpus(cpus))
    if not allowed:
        raise RuntimeError(f'No Slurm CPU allocation found for node {hostname}')
    return allowed


def gpu_cpu_plan():
    # Query CUDA's visible-device ordering directly. Physical nvidia-smi indices
    # need not match CUDA_VISIBLE_DEVICES order or the Slurm allocation indices.
    runtime = ctypes.CDLL('libcudart.so')
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        raise RuntimeError('CUDA could not enumerate the allocated GPUs')
    ranks = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
    local_ranks = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', str(ranks)))
    if count.value < local_ranks:
        raise RuntimeError('This mapper requires one visible GPU per node-local MPI rank')
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    near = {}
    for line in topology.splitlines():
        cells = line.split()
        if cells and re.fullmatch(r'GPU\d+', cells[0]):
            near[int(cells[0][3:])] = expand_cpus(cells[-3])
    allowed = set(os.sched_getaffinity(0))
    if 'SLURM_JOB_ID' in os.environ:
        job = subprocess.check_output(['scontrol', 'show', 'job', '-dd',
                                       os.environ['SLURM_JOB_ID']], text=True)
        allowed &= allocated_cpus(job, socket.gethostname())
    if len(allowed) < local_ranks:
        raise RuntimeError('Not enough allocated CPUs; disable MPI binding before this mapper')
    devices = []
    for rank in range(local_ranks):
        bus = ctypes.create_string_buffer(32)
        if runtime.cudaDeviceGetPCIBusId(bus, len(bus), rank) != 0:
            raise RuntimeError(f'CUDA could not identify visible GPU {rank}')
        rows = subprocess.check_output(
            ['nvidia-smi', '-i', bus.value.decode(), '--query-gpu=index,uuid,pci.bus_id',
             '--format=csv,noheader'], text=True)
        index, uuid, pci = next(csv.reader(io.StringIO(rows), skipinitialspace=True))
        devices.append(dict(index=int(index), uuid=uuid.strip(), pci=pci.strip()))
    chosen, used = {}, set()
    for rank in sorted(range(local_ranks), key=lambda i: len(allowed & near[devices[i]['index']])):
        candidates = (allowed & near[devices[rank]['index']]) - used
        if not candidates:
            raise RuntimeError(f'No distinct allocated CPU near GPU {devices[rank]["uuid"]}')
        cpu = min(candidates)
        chosen[rank] = dict(cpu=cpu, **devices[rank])
        used.add(cpu)
    return chosen


def main():
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', str(rank)))
    executable = str(Path(os.environ['ATHENA_REAL_EXE']).resolve(strict=True))
    args = sys.argv[1:]
    device_flags = [arg for arg in args if arg.startswith('--kokkos-map-device-id-by=')]
    if device_flags and device_flags != ['--kokkos-map-device-id-by=mpi_rank']:
        raise RuntimeError('GPU-local CPU mapping requires --kokkos-map-device-id-by=mpi_rank')
    if not device_flags:
        args.append('--kokkos-map-device-id-by=mpi_rank')
    chosen = gpu_cpu_plan()[local_rank]
    os.sched_setaffinity(0, {chosen['cpu']})
    if rank == 0:
        digest = hashlib.sha256()
        with open(executable, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        print(f'# GPU-local executable SHA-256: {digest.hexdigest()}', flush=True)
    print(f'# GPU-local rank: rank={rank} local_rank={local_rank} node={socket.gethostname()} '
          f'pid={os.getpid()} cpu={chosen["cpu"]} '
          f'gpu_uuid={chosen["uuid"]} pci={chosen["pci"]} executable={executable}', flush=True)
    os.execv(executable, [executable, *args])


if __name__ == '__main__':
    main()
