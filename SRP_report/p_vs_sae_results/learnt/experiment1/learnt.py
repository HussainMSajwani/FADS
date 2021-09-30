from FADS_SRP.models import supervised_autoencoder, nn, lr
from FADS_SRP.input import get_dsets_
from FADS_SRP.auc import calculate_auc
from FADS_SRP.thresholding import *

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.svm import SVC
import silence_tensorflow.auto


def get_dset(h2s, sim_i):
    P = f"/home/shussain/Simulated_data/15042021/{h2s}/simulation_output{sim_i}"
    return get_dsets_(P)

def get_learnt_rep(ae, X):
    out = X
    for L in ae.layers:
        out = L(out)
        if L.name == "latent_dim":
            return out.numpy()

H2S = [0.05, 0.25, 0.5, 0.75, 1]
K = [5, 25, 100, 300]

learnt = False

with open("/home/shussain/final_report/p_vs_sae_results/learnt/experiment1/learnt_representation.csv", 'w') as t:
    t.writelines("h2s,simulation,k,sae,learnt_svm,learnt_lr,learnt_nn\n")

for h2s in tqdm(H2S, desc="h2s"):
    for k in tqdm(K):
        for i in range(1, 11):
            #get dataset
            dset = get_dset(h2s, i)
            #train sae
            hist, ae = supervised_autoencoder(dset["X_train"], dset["y_train"], sizes=[k], 
                                            input_shape=1000, reconstruction_weight=0.3, 
                                            dropout=True, activation="relu", l1=1e-2)
            #generate sum of weights and add it to ct
            dset["ct"] = ae_ct(dset["ct"], ae)        
            #calculate auc of supervised part of sae (logistic regression)
            sae_auc = calculate_auc(
                lambda X: ae.predict(X)[0][:, 1],
                dset["X_test"],
                dset["y_test"][:, 1]
            )
            #obtain the latent representation of the dataset. this will have d=300
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
            #train NN on learnt representation
            hist, model = nn(dset["X_train_learnt"], dset["y_train"])
            
            learnt_nn_auc = calculate_auc(
                lambda X: model.predict(X)[:, 1],
                dset["X_test_learnt"],
                dset["y_test"][:, 1]
            )
            
            with open("p_vs_sae_results/learnt_representation.csv", 'a') as t:
                t.writelines(f"{h2s},{i},{k},{sae_auc},{learnt_svm_auc},{learnt_lr_auc},{learnt_nn_auc}\n")
        
        