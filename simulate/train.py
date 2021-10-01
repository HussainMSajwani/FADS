import argparse
from FADS.FeatureSelector import *
from FADS.input import get_dset
from functools import partial
from tqdm import tqdm
from matplotlib.pyplot import savefig, clf
from numpy import save
from os import getcwd



parser = argparse.ArgumentParser(description="train")

parser.add_argument('genvar', type=float, help='genvar')
parser.add_argument('i', type=int, help='simulati----------------------------------on_output')
parser.add_argument('method', type=str, help='method to use')
parser.add_argument('h2s', type=float, help='h2s')
parser.add_argument('d', type=int, help='d')
parser.add_argument('indir', type=str, help='input dir')
parser.add_argument('outdir', type=str, help='output dir')

args = parser.parse_args()

h2s = args.h2s
if h2s == 1.0 or h2s == 0.0:
    h2s = int(h2s)
i = args.i
d = args.d
genvar = args.genvar



if args.indir == '.':
    indir = getcwd()
else:
    indir = args.indir

if args.outdir == '.':
    outdir = getcwd()
else:
    outdir = args.outdir


print(args)

dset = get_dset(f"{indir}/PS/output")

#d = dset["X_train"].shape[1]

methods = {
    'ae': partial(SAE, sizes=[800, 300]),
    'ln': partial(LassoNet, architecture=(1200, 800, 200,)),
    'p' : pValue,
    'hsic': HSIClasso
}


method = methods[args.method]
method = method(dset)

method.train()

#save(f"/home/shussain/FADS/Conf/results/results_3000/h2s_{h2s}/sim_{i}/importance.npy", method.importance)
method.manhattan()
savefig(f"{outdir}/{args.method}.jpeg")




for k in tqdm([5, 25, 100, 300]):
    if k > method.max_k:
        continue
    svm_auc = method.train_svm(k)
    lr_auc = method.train_lr(k)

    n_causals = method.n_causals_top_k(k)
    
    with open(f"{outdir}/n_causals.csv", 'a') as f:
        #print(f"{base},{h2s},{i},{k},{args.method},{svm_auc},{lr_auc}\n")
        #f.writelines(f"{base},{h2s},{i},{k},{args.method},\n")
        f.writelines(f"{h2s},{i},{k},{genvar}.{args.method},{n_causals},{svm_auc},{lr_auc},{d}\n")

