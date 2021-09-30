import subprocess
from tqdm.auto import tqdm, trange

H2S = [0.05, 0.25, 0.5, 0.75, 1]

for h2s in tqdm(H2S, desc="h2s", ):  
    for i in trange(1, 6, desc="sim"):
        clean = subprocess.Popen(["make", "clean"], stdout=subprocess.PIPE, cwd="/home/shussain/make")
        clean.communicate()
        
        args = ["make", "sim", "chr=22", f"h2s={h2s}", "dc=10", "n=5000", "d=20000"]
        make = subprocess.Popen(args, stdout=subprocess.PIPE, cwd="/home/shussain/make")
        r=make.communicate()
        
        move = subprocess.Popen(["make", "move", f"name={h2s}/simulation_output{i}", f"log={r[0].decode('utf-8')}"], cwd="/home/shussain/make")
        r=move.communicate()
