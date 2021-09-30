from FADS_SRP.models import supervised_autoencoder, lr
from FADS_SRP.input import get_dsets_
from FADS_SRP.auc import calculate_auc
from FADS_SRP.thresholding import *

import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from sklearn.svm import SVC
import silence_tensorflow.auto
from pandas import DataFrame
from numpy.random import normal
from time import sleep

import argparse

wait = normal(1, 0.25)
print(f"sleeping {wait} seconds")
sleep(wait)

parser = argparse.ArgumentParser(description="Train SAE on dataset")

parser.add_argument('h2s', type=float, help='h2s')
parser.add_argument('i', type=int, help='simulation_output')
parser.add_argument('rho', type=float, help='seed')

args = parser.parse_args()

def get_dset(h2s, sim_i):
    if sim_i == 1.0:
        sim_i = int(sim_i)
    P = f"/home/shussain/Simulated_data/01072021/{h2s}/simulation_output{sim_i}/PS/output"
    return get_dsets_(P)

def get_learnt_rep(ae, X):
    out = X
    for L in ae.layers:
        out = L(out)
        if L.name == "latent_dim":
            return out.numpy()

        
h2s = args.h2s

if h2s == 1.0 or h2s == 0.0:
    h2s = int(h2s) 

rho = args.rho if args.rho != 0 else int(args.rho)
i = args.i

K = [5, 25, 100, 300]    

path = "/home/shussain/experiments/different_intervals_chr22/results"

dset = get_dset(h2s, i)

hist, ae = supervised_autoencoder(dset["X_train"], dset["y_train"], sizes=[300], 
                                  input_shape=1000, reconstruction_weight=rho, 
                                  dropout=False, activation="relu", l1=0)

h2s = float(h2s)

DataFrame(hist.history).to_csv(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/loss.csv")

dset["ct"] = ae_ct(dset["ct"], ae)        

plt.clf()
manhattan_plot(dset["ct"])
plt.savefig(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/p_manhattan.jpeg")
plt.clf()
ae_thresh_plot(dset["ct"])
plt.savefig(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/ae_manhattan.jpeg")
plt.close()

dset["X_train_learnt"] = get_learnt_rep(ae, dset["X_train"])
dset["X_test_learnt"] = get_learnt_rep(ae, dset["X_test"])
#train SVM on learnt representation
learnt_svm = SVC(probability=True)
learnt_svm.fit(dset["X_train_learnt"], dset["y_train"][:, 1])

learnt_svm_auc = calculate_auc(
    lambda X: learnt_svm.predict_proba(X)[:, 1],
    dset["X_test_learnt"],
    dset["y_test"][:, 1]
)
#train LR on learnt representation
hist, learnt_lr = lr(dset["X_train_learnt"], dset["y_train"])

learnt_lr_auc = calculate_auc(
    lambda X: learnt_lr.predict(X)[:, 1],
    dset["X_test_learnt"],
    dset["y_test"][:, 1]
)

with open(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/learnt_auc.csv", 'w') as t:
    t.writelines(f"{h2s},{i},{learnt_svm_auc},{learnt_lr_auc}\n")

with open(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/thresh_auc.csv", 'w') as t:
    pass
    
for k in K:
    k_p_SNPs_index = get_n_SNPs(dset["ct"], k)
    k_ae_SNPs_index = ae_get_n_SNPs(dset["ct"], k)

    #train SVM on p thresholded data
    p_thresh_svm = SVC(probability=True)
    p_thresh_svm.fit(dset["X_train"][:, k_p_SNPs_index], dset["y_train"][:, 1])

    p_thresh_svm_auc = calculate_auc(
        lambda X: p_thresh_svm.predict_proba(X)[:, 1],
        dset["X_test"][:, k_p_SNPs_index],
        dset["y_test"][:, 1]
    )
    #train SVM on ae thresholded data
    ae_thresh_svm = SVC(probability=True)
    ae_thresh_svm.fit(dset["X_train"][:, k_ae_SNPs_index], dset["y_train"][:, 1])

    ae_thresh_svm_auc = calculate_auc(
        lambda X: ae_thresh_svm.predict_proba(X)[:, 1],
        dset["X_test"][:, k_ae_SNPs_index],
        dset["y_test"][:, 1]
    )
    #train LR on p thresh data
    hist, p_thresh_lr = lr(dset["X_train"][:, k_p_SNPs_index], dset["y_train"])

    p_thresh_lr_auc = calculate_auc(
        lambda X: p_thresh_lr.predict(X)[:, 1],
        dset["X_test"][:, k_p_SNPs_index],
        dset["y_test"][:, 1]
    )
    #train LR on ae thresh data
    hist, ae_thresh_lr = lr(dset["X_train"][:, k_ae_SNPs_index], dset["y_train"])

    ae_thresh_lr_auc = calculate_auc(
        lambda X: ae_thresh_lr.predict(X)[:, 1],
        dset["X_test"][:, k_ae_SNPs_index],
        dset["y_test"][:, 1]
    )
    
    with open(f"{path}/rho_{rho}/h2s_{h2s}/sim_{i}/thresh_auc.csv", 'a') as t:
        t.writelines(f"{h2s},{i},{k},{p_thresh_svm_auc},{ae_thresh_svm_auc},{p_thresh_lr_auc},{ae_thresh_lr_auc}\n")