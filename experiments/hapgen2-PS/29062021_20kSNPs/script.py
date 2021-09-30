from FADS_SRP.models import supervised_autoencoder, nn, lr
from FADS_SRP.input import get_dsets_
from FADS_SRP.auc import calculate_auc
from FADS_SRP.thresholding import *

import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from sklearn.svm import SVC
import silence_tensorflow.auto
from pandas import DataFrame

def get_dset(h2s, sim_i):
    P = f"/home/shussain/Simulated_data/29062021/{h2s}/simulation_output{sim_i}/PS/output/"
    return get_dsets_(P)

path = "/home/shussain/experiments/hapgen2-PS/29062021_20kSNPs/"
with open(f"{path}thresholding.csv", 'w') as t:
    t.writelines("h2s,simulation,k,p_svm,ae_svm,p_lr,ae_lr,p_nn,ae_nn\n")

H2S = [0.05, 0.25, 0.5, 0.75, 1]
K = [5, 25, 100, 300]    

for h2s in H2S:
    print(f"h2s={h2s}")
    for i in trange(1, 6, desc="sim_out"):
        #get dataset
        dset = get_dset(h2s, i)
        #train sae
        hist, ae = supervised_autoencoder(dset["X_train"], dset["y_train"], sizes=[300], 
                                         input_shape=20000, reconstruction_weight=0.3, 
                                         dropout=True, activation="relu", l1=1e-2)
        #generate sum of weights and add it to ct
        dset["ct"] = ae_ct(dset["ct"], ae)        
        
        plt.clf()
        manhattan_plot(dset["ct"])
        plt.savefig(f"{path}/plots/simulation_output{i}/{h2s}/p_manhattan.jpeg")
        plt.clf()
        ae_thresh_plot(dset["ct"])
        plt.savefig(f"{path}plots/simulation_output{i}/{h2s}/ae_manhattan.jpeg")
        plt.close()
        for k in tqdm(K, desc="k"):
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

            
            with open(f"{path}thresholding.csv", 'a') as t:
                t.writelines(f"{h2s},{i},{k},{p_thresh_svm_auc},{ae_thresh_svm_auc},{p_thresh_lr_auc},{ae_thresh_lr_auc}\n")

