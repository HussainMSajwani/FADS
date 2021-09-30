#!/bin/bash

h2s=$1
i=$2
r2=$3

path=/home/shussain/Simulated_data/01072021/$1/simulation_output$2/prelim

mkdir -p $path/pruned

~/plink/plink-1.9/plink --bfile $path/prelim_plink_chr22\
                        --indep-pairwise 50 5 $r2\
                        --out $path/pruned/r2_$r2\
