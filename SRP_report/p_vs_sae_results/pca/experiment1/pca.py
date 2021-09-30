from FADS_SRP.models import supervised_autoencoder, nn, lr
from FADS_SRP.input import get_dsets_
from FADS_SRP.auc import calculate_auc
from FADS_SRP.thresholding import *

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import silence_tensorflow.auto


def get_dset(h2s, sim_i):
    P = f"/home/shussain/Simulated_data/15042021/{h2s}/simulation_output{sim_i}"
    return get_dsets_(P)

H2S = [0.05, 0.25, 0.5, 0.75, 1]
K = [5, 25, 100, 300]

with open("/home/shussain/final_report/p_vs_sae_results/pca/experiment1/pca.csv", 'w') as t:
    t.writelines("h2s,simulation,k,pca_svm,pca_lr,pca_nn\n")

for h2s in tqdm(H2S, desc="h2s"):
    for k in tqdm(K):
        for i in range(1, 11):
            #get dataset
            dset = get_dset(h2s, i)
            #train pca
            pca = PCA(n_components=k)
            pca_out_train = pca.fit_transform(dset["X_train"])
            pca_out_test = pca.transform(dset["X_test"])
            #train SVM on pca representation
            pca_svm = SVC(probability=True)
            pca_svm.fit(pca_out_train, dset["y_train"][:, 1])
            
            pca_svm_auc = calculate_auc(
                lambda X: pca_svm.predict_proba(X)[:, 1],
                pca_out_test,
                dset["y_test"][:, 1]
            )
            #train LR on pca representation
            hist, pca_lr = lr(pca_out_train, dset["y_train"])
            
            pca_lr_auc = calculate_auc(
                lambda X: pca_lr.predict(X)[:, 1],
                pca_out_test,
                dset["y_test"][:, 1]
            )
            #train NN on pca representation
            hist, model = nn(pca_out_train, dset["y_train"])
            
            pca_nn_auc = calculate_auc(
                lambda X: model.predict(X)[:, 1],
                pca_out_test,
                dset["y_test"][:, 1]
            )
            
            with open("/home/shussain/final_report/p_vs_sae_results/pca/experiment1/pca.csv", 'a') as t:
                t.writelines(f"{h2s},{i},{k},{pca_svm_auc},{pca_lr_auc},{pca_nn_auc}\n")
        
        