from FADS.input import get_dsets_
from FADS.thresholding import get_n_SNPs, manhattan_plot
from FADS.auc import calculate_auc
from FADS.models import lr

from lassonet import LassoNetClassifier
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.svm import SVC
import pickle
from pandas import DataFrame

ppath = "/home/shussain/experiments/lassonet/results"

parser = argparse.ArgumentParser(description="Train SAE on dataset")

parser.add_argument('h2s', type=float, help='h2s')
parser.add_argument('i', type=int, help='simulation_output')

args = parser.parse_args()

h2s = args.h2s
if h2s == 1.0 or h2s == 0.0:
    h2s = int(h2s) 
i = args.i

def lassonet_manhattan(ct, model):
    causals_index = ct.query("causal==1").index.values
    ncausals_index = ct.query("causal==0").index.values
    
    x = np.array(list(range(ct.shape[0])))
    
    plt.scatter(x[ncausals_index], model.feature_importances_.cpu().numpy()[ncausals_index])
    plt.scatter(x[causals_index], model.feature_importances_.cpu().numpy()[causals_index])
    plt.xlabel("SNP id")
    plt.ylabel("lambda")

    plt.savefig(f"{ppath}/h2s_{h2s}/sim_{i}/lassonet.jpeg")


ds = get_dsets_(f"/home/shussain/Simulated_data/13072021/{h2s}/sim_{i}/PS/output/")
gpu = np.random.randint(0, 2)
model = LassoNetClassifier(verbose=False, hidden_dims=(500, 200, 10), device=f"cuda:{gpu}")
path = model.path(ds["X_train"], ds["y_train"][:, 1])

selected_SNPs = [hi.selected.cpu().numpy() for hi in path]
plt.plot([np.sum(s) for s in selected_SNPs])
plt.savefig(f"{ppath}/h2s_{h2s}/sim_{i}/selected.jpeg")

with open(f'{ppath}/h2s_{h2s}/sim_{i}/selected.pkl', 'wb') as f:
    pickle.dump(selected_SNPs, f)

ds["ct"]["lambda"] = model.feature_importances_.cpu().numpy()
ds["ct"].to_csv(f"{ppath}/h2s_{h2s}/sim_{i}/ct.csv")
sorted_lambdas = list(reversed(np.argsort(model.feature_importances_.cpu().numpy())))

plt.clf()
lassonet_manhattan(ds["ct"], model)

with open(f"{ppath}/h2s_{h2s}/sim_{i}/thresh_auc.csv", 'w') as t:
    pass
for k in [5, 25, 100, 300]:
    k_p_SNPs_index = get_n_SNPs(ds["ct"], k)
    k_top_lambdas = sorted_lambdas[:k]

    #train SVM on p thresholded data
    p_thresh_svm = SVC(probability=True)
    p_thresh_svm.fit(ds["X_train"][:, k_p_SNPs_index], ds["y_train"][:, 1])

    p_thresh_svm_auc = calculate_auc(
        lambda X: p_thresh_svm.predict_proba(X)[:, 1],
        ds["X_test"][:, k_p_SNPs_index],
        ds["y_test"][:, 1]
    )
    #train SVM on ae thresholded data
    ae_thresh_svm = SVC(probability=True)
    ae_thresh_svm.fit(ds["X_train"][:, k_top_lambdas], ds["y_train"][:, 1])

    ae_thresh_svm_auc = calculate_auc(
        lambda X: ae_thresh_svm.predict_proba(X)[:, 1],
        ds["X_test"][:, k_top_lambdas],
        ds["y_test"][:, 1]
    )
    #train LR on p thresh data
    hist, p_thresh_lr = lr(ds["X_train"][:, k_p_SNPs_index], ds["y_train"])

    p_thresh_lr_auc = calculate_auc(
        lambda X: p_thresh_lr.predict(X)[:, 1],
        ds["X_test"][:, k_p_SNPs_index],
        ds["y_test"][:, 1]
    )
    #train LR on ae thresh data
    hist, ae_thresh_lr = lr(ds["X_train"][:, k_top_lambdas], ds["y_train"])

    ae_thresh_lr_auc = calculate_auc(
        lambda X: ae_thresh_lr.predict(X)[:, 1],
        ds["X_test"][:, k_top_lambdas],
        ds["y_test"][:, 1]
    )
    
    with open(f"{ppath}/h2s_{h2s}/sim_{i}/thresh_auc.csv", 'a') as t:
        t.writelines(f"{h2s},{i},{k},{p_thresh_svm_auc},{ae_thresh_svm_auc},{p_thresh_lr_auc},{ae_thresh_lr_auc}\n")
