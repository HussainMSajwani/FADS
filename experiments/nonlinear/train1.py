from FADS.input import get_dsets_
from FADS.thresholding import get_n_SNPs, manhattan_plot, ae_get_n_SNPs, ae_ct, ae_manhattan
from FADS.auc import calculate_auc
from FADS.models import lr, supervised_autoencoder

from lassonet import LassoNetClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pickle
from pandas import DataFrame
import argparse
from torch.cuda import empty_cache

ppath = "/home/shussain/experiments/nonlinear/results1"

parser = argparse.ArgumentParser(description="Train SAE on dataset")

parser.add_argument('h2s', type=float, help='h2s')
parser.add_argument('i', type=int, help='simulation_output')
parser.add_argument('base', type=float, help='base')


args = parser.parse_args()

h2s = args.h2s
if h2s == 1.0 or h2s == 0.0:
    h2s = int(h2s)
i = args.i
base = args.base
if base == int(base):
    base = int(base)

print(h2s, i, base)

def lassonet_manhattan(ct, model):
    causals_index = ct.query("causal==1").index.values
    ncausals_index = ct.query("causal==0").index.values

    x = np.array(list(range(ct.shape[0])))

    plt.scatter(x[ncausals_index],
                model.feature_importances_.cpu().numpy()[ncausals_index])
    plt.scatter(x[causals_index],
                model.feature_importances_.cpu().numpy()[causals_index])
    plt.xlabel("SNP id")
    plt.ylabel("lambda")

    plt.savefig(f"{ppath}/{h2s}/base_{base}/sim_{i}/lassonet.jpeg")


# LassoNet
ds = get_dsets_(
    f"/home/shussain/Simulated_data/14072021/{h2s}/base_{base}/sim_{i}/sim_{i}/PS/output/")

"""
gpu = np.random.randint(0, 2)
model = LassoNetClassifier(verbose=False, hidden_dims=(
    500, 200, 10), device=f"cuda:{gpu}")
path = model.path(ds["X_train"], ds["y_train"][:, 1])
print("here 1")
selected_SNPs = [hi.selected.cpu().numpy() for hi in path]
plt.plot([np.sum(s) for s in selected_SNPs])
plt.savefig(f"{ppath}/{h2s}/base_{base}/sim_{i}/selected.jpeg")
plt.clf()

with open(f'{ppath}/{h2s}/base_{base}/sim_{i}/selected.pkl', 'wb') as f:
    pickle.dump(selected_SNPs, f)
print("here 2")
ds["ct"]["lambda"] = model.feature_importances_.cpu().numpy()
sorted_lambdas = list(
    reversed(np.argsort(model.feature_importances_.cpu().numpy())))

#AE
print("done with pytorch")

lassonet_manhattan(ds["ct"], model)
plt.savefig(f"{ppath}/{h2s}/base_{base}/sim_{i}/lasso_manhattan.jpeg")

del model
"""
empty_cache()
print("right before ae")
hist, ae = supervised_autoencoder(ds["X_train"], ds["y_train"], sizes=[300],
                                  input_shape=1000, reconstruction_weight=0.3,
                                  dropout=True, activation="relu", l1=1e-2)
#generate sum of weights and add it to ct
ds["ct"] = ae_ct(ds["ct"], ae)
ds["ct"].to_csv(f"{ppath}/{h2s}/base_{base}/sim_{i}/ct.csv")
print("after ae")

plt.clf()
manhattan_plot(ds["ct"])
plt.savefig(f"{ppath}/{h2s}/base_{base}/sim_{i}/p_manhattan.jpeg")

plt.clf()
ae_manhattan(ds["ct"])
plt.savefig(f"{ppath}/{h2s}/base_{base}/sim_{i}/ae_manhattan.jpeg")


with open(f"{ppath}/{h2s}/base_{base}/sim_{i}/thresh_auc.csv", 'w') as t:
    pass
del ae
print("deleted ae")

for k in [5, 25, 50, 100, 300]:
    print("k=", k)
    k_p_SNPs_index = get_n_SNPs(ds["ct"], k)
    k_top_lambdas = sorted_lambdas[:k]
    k_ae_SNPs_index = ae_get_n_SNPs(ds["ct"], k)


    #train SVM on p thresholded data
    p_thresh_svm = SVC(probability=True)
    p_thresh_svm.fit(ds["X_train"][:, k_p_SNPs_index], ds["y_train"][:, 1])

    p_thresh_svm_auc = calculate_auc(
        lambda X: p_thresh_svm.predict_proba(X)[:, 1],
        ds["X_test"][:, k_p_SNPs_index],
        ds["y_test"][:, 1]
    )
    #train SVM on ae thresholded data
    ln_thresh_svm = SVC(probability=True)
    ln_thresh_svm.fit(ds["X_train"][:, k_top_lambdas], ds["y_train"][:, 1])

    ln_thresh_svm_auc = calculate_auc(
        lambda X: ln_thresh_svm.predict_proba(X)[:, 1],
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
    hist, ln_thresh_lr = lr(ds["X_train"][:, k_top_lambdas], ds["y_train"])

    ln_thresh_lr_auc = calculate_auc(
        lambda X: ln_thresh_lr.predict(X)[:, 1],
        ds["X_test"][:, k_top_lambdas],
        ds["y_test"][:, 1]
    )
    #AE
    ae_thresh_svm = SVC(probability=True)
    ae_thresh_svm.fit(ds["X_train"][:, k_ae_SNPs_index], ds["y_train"][:, 1])

    ae_thresh_svm_auc = calculate_auc(
        lambda X: ae_thresh_svm.predict_proba(X)[:, 1],
        ds["X_test"][:, k_ae_SNPs_index],
        ds["y_test"][:, 1]
    )

    hist, ae_thresh_lr = lr(ds["X_train"][:, k_ae_SNPs_index], ds["y_train"])

    ae_thresh_lr_auc = calculate_auc(
        lambda X: ae_thresh_lr.predict(X)[:, 1],
        ds["X_test"][:, k_ae_SNPs_index],
        ds["y_test"][:, 1]
    )
    with open(f"{ppath}/{h2s}/base_{base}/sim_{i}/thresh_auc.csv", 'a') as t:
        t.writelines(
            f"{h2s},{i},{k},{p_thresh_svm_auc},{ae_thresh_svm_auc},{ln_thresh_svm_auc},{p_thresh_lr_auc},{ae_thresh_lr_auc},{ln_thresh_lr_auc}\n")
