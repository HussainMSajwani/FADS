import argparse
from FADS.FeatureSelector import *
from FADS.input import get_dset
from functools import partial
from tqdm import tqdm
from matplotlib.pyplot import savefig, clf

parser = argparse.ArgumentParser(description="train")

parser.add_argument('i', type=int, help='simulation_output')
parser.add_argument('base', type=float, help='base')
parser.add_argument('method', type=str, help='method to use')
parser.add_argument('h2s', type=float, help='h2s')

args = parser.parse_args()

h2s = args.h2s
if h2s == 1.0 or h2s == 0.0:
    h2s = int(h2s)
i = args.i
base = args.base
if base == int(base):
    base = int(base)

print(args)

dset = get_dset(f"/home/shussain/Simulated_data/14072021/{h2s}/base_{base}/sim_{i}/sim_{i}/PS/output")

methods = {
    'ae': partial(SAE, sizes=[800, 300]),
    'ln': partial(LassoNet, architecture=(1200, 800, 200,)),
    'p' : pValue,
    'hsic': HSICLasso
}


method = methods[args.method]
method = method(dset)

#method.manhattan()
#savefig(f"/home/shussain/FADS/experiments/nonlinear/results_comp/base_{base}/h2s_{h2s}/sim_{i}/{args.method}.jpeg")


for k in tqdm([5, 25, 100, 300]):
    #svm_auc = method.train_svm(k)
    #lr_auc = method.train_lr(k)

    n_causals = method.n_causals_top_k(k)
    
    with open(f"/home/shussain/FADS/experiments/nonlinear/results_comp/base_{base}/h2s_{h2s}/sim_{i}/n_causals.csv", 'a') as f:
        #print(f"{base},{h2s},{i},{k},{args.method},{svm_auc},{lr_auc}\n")
        #f.writelines(f"{base},{h2s},{i},{k},{args.method},{svm_auc},{lr_auc}\n")
        f.writelines(f"{base},{h2s},{i},{k},{args.method},{n_causals}\n")
