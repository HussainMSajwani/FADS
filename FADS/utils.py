from nvsmi import get_available_gpus
from pathlib import Path
from itertools import product
from os import listdir

def mkoutdir(path,**kwargs):
    keys = kwargs.keys()
    outdir = Path(path) / "results"
    
    items = [pair[1] for pair in kwargs.items()]
    all_poss = product(*items)
    
    for poss in all_poss:
        outdir_poss = outdir
        for key, parameter in zip(keys, poss):
            outdir_poss /= f"{key}_{parameter}"
        Path(outdir_poss).mkdir(parents=True, exist_ok=True) 


def mkoutdir_params(path, params_path):
    items = {}
    for param in listdir(params_path):
        with open(params_path + param, 'r') as f:
            items[param] = f.readlines()

    keys = items.keys()
    outdir = Path(path) / "results"
    
    items = [pair[1] for pair in items.items()]
    all_poss = product(*items)
    
    for poss in all_poss:
        outdir_poss = outdir
        for key, parameter in zip(keys, poss):
            outdir_poss /= f"{key}_{parameter[:-1]}"
        Path(outdir_poss).mkdir(parents=True, exist_ok=True) 


def which_GPU():
    key = lambda gpu: gpu.mem_used

    GPUs = list(get_available_gpus(gpu_util_max=0.9, mem_util_max=0.9))

    if len(GPUs) == 0:
        while len(GPUs) == 0:
            GPUs = list(get_available_gpus(gpu_util_max=0.9)) 
            
    avail = min(GPUs, key=key)
    print("Using GPU", avail)
    if "GPU-af44f744-db9e-7732-4364-d11c06f70ef8" in [gpu.uuid for gpu in GPUs]:
        return int(avail.id)
    else: 
        return int(avail.id)-1  