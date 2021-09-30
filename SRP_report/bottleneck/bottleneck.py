import numpy as np
import matplotlib.pyplot as plt
from FADS_SRP.models import supervised_autoencoder, compiled_autoencoder
from FADS_SRP.input import get_dsets_
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from pandas import DataFrame

def get_dset(h2s, sim_i):
    P = f"/home/shussain/Simulated_data/15042021/{h2s}/simulation_output{sim_i}"
    return get_dsets_(P)

out = []
mse = []
wd = []
sizes = [1000, 750, 500, 250, 100, 50, 20, 10, 5]

dset = get_dset(0.5, 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
plt.hist(dset["X_train"].ravel())
plt.xlabel("Occurence of major allele")
plt.savefig("/home/shussain/final_report/bottleneck/results/original.jpeg")

for size in tqdm(sizes):
    hist, model = supervised_autoencoder(dset["X_train"], dset["y_train"], sizes=[size], input_shape=1000, reconstruction_weight=0.8)
    out_ = model(dset["X_train"])[1].numpy()
    out.append(out_)
    wd.append(wasserstein_distance(dset["X_train"].ravel(), out_.ravel()))
    mse.append(hist.history["loss"][-1])
    
DataFrame({
    "size": sizes,
    "wasserstein": wd,
    "mse": mse
}).to_csv("/home/shussain/final_report/bottleneck/results/result.csv")    

fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, constrained_layout=True)
#plt.tight_layout()
for size, X_train_hat, m, w, axes in zip(sizes, out, mse, wd, ax.ravel()):
    axes.hist(X_train_hat.ravel())
    axes.set_title(f"k={size}, WD={str(w)[:6]}, MSE={str(m)[:6]}")

plt.savefig("/home/shussain/final_report/bottleneck/results/result.jpeg")